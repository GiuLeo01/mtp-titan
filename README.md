# MTP-TITAN

## Overview


## Code Guide

Il framework di training alla base del progetto è torchtitan, pinnato alla versione X. è stato scelto di integrare il codice, come modulo esterno, senza toccare la repo originale del framework.

Questo è stato possibile perché torchtitan espone un punto di estensione generico: il `ConfigManager`
accetta in `--module` un module path qualsiasi e ne importa il `config_registry`. Tutto il 
codice sta quindi in un pacchetto a parte, `mtp_titan`.

 Le due varianti MTP sono sottoclassi di `Llama3Model`, la loss è una
sottoclasse di `BaseLoss`, le metriche per testa di `MetricsProcessor`.

Quindi le run possono essere semplicemente lanciate per esempio con:

```bash
MODULE=mtp_titan CONFIG=gloeckle_57m NGPU=1 ./run_train.sh
```


### architectures.py

In questo script sono definiti i diversi modelli utilizzati per questi esperimenti:
- baseline: variante di llama3
- Gloeckle: trunk di llama3 accorciato, più n mtp head parallele, implementate come in X
- Deepseek: trunk di llama3 accorciato, più D mtp module sequenziali, implementati come in X

Le taglie stanno in `SHAPES` (`debug`, `17m`, `57m`, `101m`): si danno `dim` e `n_layers`, il
resto (`n_heads`, `hidden_dim`, conteggio dei parametri) è derivato. Gli esperimenti usano solo
`57m`.

`model_config()` costruisce una configurazione per Llama3, quindi utilizzabile per la baseline, e
poi riutilizzata anche per costruire le model_config di Gloeckle e Deepseek. Entrambe partono dalla
config densa e ne affettano i layer: `trunk_depth = L - n` per Gloeckle, `L - D` per Deepseek, e i
blocchi rimanenti diventano le teste o i moduli. Questa convenzione, ispirata al paper di Gloeckle et al. è necessaria per rendere le run comparabili.

`model_spec()` impacchetta la config con `parallelize_fn`, `state_dict_adapter`, necessari per il training.

### config_registry.py

In questo script si trovano, e costruiscono, le effettive ricette per ogni run, tutte costruite a
partire dalla funzione `_recipe()` che fissa tutto ciò che deve restare uguale fra i bracci:
ottimizzatore, scheduler, dati, batch globale, validazione, checkpoint e metriche. Ogni variante ha il suo wrapper, `_baseline()`, `_gloeckle()` e `_deepseek()`, che cambiano il `model_spec` e la loss.


Le ricette sono una funzione per esperimento, e seed
(`baseline_57m`, `gloeckle_57m_n3`, `deepseek_57m_efficient`, i `..._seed1`, i `..._profile` per il
profiling). Servono funzioni separate perché il `ConfigManager`
chiama `config_fn()` senza argomenti.

### gloeckle.py

In questo file si trova l'implementazione del modello MTP classico (Gleoeckle et al). la classe GloeckleModel, ereditando da LLama3Model, usa prima il metodo trunk_hidden_states, che semplicemente itera sui layer del trunk, fino a ottenere gli hidden states; successivamente, il forward itera sulle teste mtp, producendo la tupla degli output. Un metodo fondamentale è preprocess_inputs, che è in gran parte analogo al default di llama3, a parte la gestione delle labels; infatti, per poter calcolare la loss di MTP, ogni testa ha la sua label, che è shiftata di 1 rispetto alla testa precedente. La funzione che si occupa del calcolo delle labels shiftate è mtp_labels(), definita in targets.py


### deepseek.py

In questo file si trova l'implementazione del modello MTP sequenziale (DeepSeek-V3, §2.2). La differenza rispetto a Gloeckle è che le teste non sono indipendenti ma sequenziali: ogni modulo riceve la rappresentazione prodotta dal precedente, più l'embedding del token futuro vero. La classe DeepSeekMtpModule implementa una singola testa di predizione, ed è fatta da due RMSNorm, una proiezione M_k (una Linear da 2d a d) e un blocco transformer: il forward normalizza separatamente la rappresentazione in ingresso e l'embedding del token futuro, li concatena, li proietta con M_k e passa il risultato al blocco. La classe DeepSeekModel, che come GloeckleModel eredita da LLama3Model, calcola prima gli hidden states del trunk, e poi itera sui moduli riassegnando hidden_states a ogni giro, che è esattamente ciò che rende la catena sequenziale; l'embedding del token futuro si ottiene shiftando a sinistra gli embedding già calcolati per il trunk. Su ogni uscita viene poi chiamato decode(), che applica la norm finale e l'lm_head. preprocess_inputs() è identico a quello di Gloeckle e usa la stessa mtp_labels().

Nel file c'è anche deepseek_head_weights(), che costruisce i pesi della loss dell'eq. 25: 1 per il main model e λ/D per ciascuna profondità.

### loss.py

In questo script si trovano le due implementazioni della loss MTP, entrambe condivise dalle due varianti. MtpLoss è la versione Naive, che può essere condivisa da entrambe le versioni di MTP, in quando deepseek differisce solo per il modo in cui i termini della loss vengono pesati.
 Dentro la classe c'è anche un accumulatore delle CE per testa, che serve a loggarle separatamente: il dizionario di metriche che la loss restituisce viene scartato dal trainer, quindi il passaggio avviene attraverso questo accumulatore, che MtpMetricsProcessor drena a ogni logging.  Le CE accumulate sono sempre quelle grezze, mai pesate, altrimenti la CE della testa 1 non sarebbe più confrontabile con la baseline.

MtpMemoryEfficientLoss è la versione di Gloeckle §2, quella che evita di tenere tutti i logit in memoria insieme, facendo forward e backward singolarmente su ogni testa, accumulando il gradiente. Eredita da ChunkedLossWrapper solo per un motivo pratico: il trainer riconosce quel tipo e si occupa da solo di chiamare set_lm_head e di mettere _skip_lm_head sul modello, che così restituisce hidden states invece di logits. La loss cicla allora sulle teste, e per ognuna applica l'lm_head, calcola la CE, fa subito la backward e accumula il gradiente rispetto all'ingresso della testa: a quel punto i logit di quella testa possono essere liberati prima di passare alla successiva. I gradienti raccolti vengono infine restituiti al modello da _HeadGradientBridge, una autograd.Function custom il cui forward non fa nulla e il cui backward si limita a consegnare i gradienti già calcolati. Serve perché le teste condividono il grafo del trunk: n backward separate lo attraverserebbero n volte, e PyTorch si fermerebbe al secondo giro.

### targets.py
Contiene la funzione di shifting delle labels precedentemente citata. Olte a occuparsi dello shifting, gestisce anche il masking delle posizioni cross-documento, in modo da evitare che la loss delle mtp head venga calcolata su token appartenenti alla sequenza successiva.

### speculative.py

In questo script si trova l'implementazione del self-speculative decoding, cioè l'uso delle predizioni MTP come draft da verificare. greedy_decode è la versione di riferimento, un forward per token, e serve sia da confronto sia da oracolo per il test T8. speculative_decode è il loop vero: a ogni iterazione fa un solo forward, che serve contemporaneamente a verificare i token draftati al giro prima e a produrre quelli del giro dopo. La verifica sta in accepted_prefix_length, che confronta i token draftati con quelli che il greedy avrebbe prodotto. 

Il drafting è l'unica parte che cambia fra le due varianti, e sta in due classi con la stessa interfaccia, scelte da make_drafter. GloeckleDrafter non fa quasi nulla: le teste sono indipendenti, quindi il forward normale del modello contiene già tutte le predizioni che servono e il draft è una semplice indicizzazione. DeepSeekDrafter invece deve ricostruire la catena a mano, perché serve l'embedding del token futuro vero, che in inferenza non esiste ancora; si alimenta allora la catena con le proprie predizioni, sovrascrivendo le ultime righe degli embedding shiftati con i token appena draftati. DecodeStats tiene infine i conti: token per forward e acceptance rate.

### metrics.py

MtpMetricsProcessor è una sottoclasse di MetricsProcessor che, a ogni logging, drena l'accumulatore di MtpLoss e aggiunge le CE per testa alle metriche, sia di training che di validation. Si occupa anche di dare il nome alle run su wandb, a partire dalla ricetta.


# profiler.py

Definisce una sottoclasse del profiler, per abilitare il tracking della memoria, non sarebbe un parametro accessibile esternamente.




## Experiment Runs

57M non-embedding parameters, 16k code-BPE vocabulary, `bigcode/starcoderdata` (Python),
1.13B tokens (Chinchilla, 8641 steps), 2 seeds per arm, 1 GPU per run.
Validation on a held-out shard. Checkpoints at `step-8641`.

| Arm | Config | Non-emb params | Seed | Val next-token CE | Val CE @ k=2 | Val CE @ k=3 | Val objective | tok/s | MFU | Peak reserved |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | `baseline_57m` | 56.64M | 0 | 1.3467 | — | — | 1.3467 | 112,765 | 41.2% | 6.07 GiB |
| Baseline | `baseline_57m` | 56.64M | 1 | 1.3486 | — | — | 1.3486 | 111,069 | 40.6% | 6.07 GiB |
| | | | **mean** | **1.3477** | | | | **111,917** | **40.9%** | |
| Gloeckle n=2 | `gloeckle_57m` | 56.64M | 0 | 1.3902 | 2.0359 | — | 3.4237 | 92,993 | 34.0% | 13.32 GiB |
| Gloeckle n=2 | `gloeckle_57m` | 56.64M | 1 | 1.3954 | 2.0446 | — | 3.4375 | 93,786 | 34.3% | 13.32 GiB |
| | | | **mean** | **1.3928** | **2.0403** | | | **93,390** | **34.2%** | |
| DeepSeek D=2 | `deepseek_57m` | 59.00M | 0 | 1.3840 | 1.5433 | 1.5536 | 1.8477 | 80,892 | 30.3% | 16.20 GiB |
| DeepSeek D=2 | `deepseek_57m` | 59.00M | 1 | 1.3822 | 1.5431 | 1.5510 | 1.8455 | 79,984 | 30.0% | 16.20 GiB |
| | | | **mean** | **1.3831** | **1.5432** | **1.5523** | | **80,438** | **30.2%** | |

Notes on reading the table:

- **Val next-token CE** is the only cross-arm quality comparison: head 1 for Gloeckle, the main
  model for DeepSeek. The **val objective** column is each arm's own training loss (a sum over
  heads for Gloeckle, `L_main + (λ/D) Σ L_k` for DeepSeek) and is *not* comparable across arms.
- **CE @ k=2, k=3 are not comparable between the two MTP variants.** Gloeckle's head k predicts
  `t+k` from position `t` alone; DeepSeek's depth-k module is given the true intermediate tokens
  (teacher forcing, eq. 21), so it solves an easier conditional problem.
- DeepSeek carries +4.2% non-embedding parameters over the other two arms: both variants shorten
  the trunk to `L − n`, but DeepSeek additionally has the `M_k ∈ R^{d×2d}` projections, which have
  no equivalent to remove.
- **Peak reserved** is the allocator pool reported by torchtitan, not live memory.

## Profiling


## Speculative Decoding

Self-speculative decoding: the MTP predictions are the draft, the next forward is the
verification. Greedy throughout, so the verified output is exactly the greedy autoregressive one.

Measured with `scripts/benchmark_speculative.py` on the `step-8641` checkpoints: 5 Python
prompts × 128 generated tokens, batch size 1, one A6000.

| Arm | Draft slots | Acceptance @1 | Acceptance @2 | Tokens per forward | Forwards saved | tok/s greedy | tok/s speculative | Output ≡ greedy |
|---|---|---|---|---|---|---|---|---|
| Baseline | 0 | — | — | 1.000 | 1.00x | 169.7 | 164.8 | yes |
| Gloeckle n=2 | 1 | 91.3% | — | 1.893 | **1.89x** | 167.5 | 304.0 | yes |
| DeepSeek D=2 | 2 | 90.0% | 86.6% | 2.712 | **2.71x** | 162.3 | 413.6 | yes |

Notes on reading the table:

- **Tokens per forward** is the metric that transfers to a served model: it depends only on the
  acceptance rate. torchtitan is a training framework and has **no KV cache**, so every decoding
  step reruns the forward over the whole prefix; the **tok/s columns are demonstrative only**
  (batch 1, 57M model) and are reported as a ratio, not as absolute serving throughput.
- The two arms do not draft the same number of tokens. Gloeckle `n=2` has 2 heads, hence 1
  draftable slot; DeepSeek `D=2` has a main model plus 2 modules, hence 2 draftable slots. Both
  follow their own paper's convention for the same nominal setting.
- The baseline row exercises the same speculative loop with nothing to draft, and is the control
  that the loop itself adds no overhead.
- `Output ≡ greedy` is checked token-by-token against `greedy_decode` on every prompt, and is
  additionally a unit test (T8).

## Testing


## Conclusion