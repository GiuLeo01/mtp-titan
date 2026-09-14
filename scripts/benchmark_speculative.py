"""Measure self-speculative decoding against greedy autoregressive decoding.

torchtitan is a training framework and has no KV cache, so every step reruns
the forward over the whole prefix. The number that transfers to a served model
is therefore tokens per verification forward, which depends only on the
acceptance rate; wall-clock here is demonstrative, at batch size 1.

    python scripts/benchmark_speculative.py \
        --config gloeckle_57m \
        --checkpoint ../e1-runs/<stamp>/gloeckle_57m/checkpoint/step-8641
"""

import argparse
import importlib

import torch
import torch.distributed.checkpoint as dcp

from torchtitan.components.tokenizer import HuggingFaceTokenizer

from mtp_titan.speculative import greedy_decode, speculative_decode

PROMPTS = [
    "def binary_search(values, target):\n    low = 0\n",
    "class Rectangle:\n    def __init__(self, width, height):\n",
    "import json\n\n\ndef load_config(path):\n    with open(path) as handle:\n",
    "def fibonacci(n):\n    if n < 2:\n        return n\n",
    "for index, line in enumerate(lines):\n    if line.startswith('#'):\n",
]


def build_model(config_name: str, checkpoint: str | None, device: torch.device):
    registry = importlib.import_module("mtp_titan.config_registry")
    trainer_config = getattr(registry, config_name)()
    spec = trainer_config.model_spec

    with torch.device(device):
        model = spec.model.build()
        model.init_states(buffer_device=device)
    model.eval()

    if checkpoint is not None:
        state_dict = model.state_dict()
        dcp.load(state_dict, checkpoint_id=checkpoint)
        model.load_state_dict(state_dict)

    tokenizer = trainer_config.tokenizer.build(
        tokenizer_path=trainer_config.hf_assets_path
    )
    return model, tokenizer, spec.max_context_length


def report(name: str, stats, matched: bool) -> None:
    print(f"\n{name}")
    print(f"  generated tokens          {stats.generated_tokens}")
    print(f"  verification forwards     {stats.verification_forwards}")
    print(f"  tokens per forward        {stats.tokens_per_forward:.3f}")
    print(f"  tokens/s (demonstrative)  {stats.tokens_per_second:.1f}")
    if stats.proposed_per_slot:
        rates = ", ".join(
            f"slot {slot + 1}: {rate:6.1%}"
            for slot, rate in enumerate(stats.acceptance_rates())
        )
        print(f"  acceptance per slot       {rates}")
    print(f"  output == greedy          {'yes' if matched else 'NO'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda")
    model, tokenizer, max_context_length = build_model(
        args.config, args.checkpoint, device
    )

    sample = None
    for _ in range(args.repeats):
        greedy_total = None
        speculative_total = None
        all_matched = True

        for prompt_text in PROMPTS:
            encoded = tokenizer.encode(prompt_text)[: args.prompt_tokens]
            prompt = torch.tensor(encoded, dtype=torch.int64, device=device)

            greedy_tokens, greedy_stats = greedy_decode(
                model, prompt, args.new_tokens, max_context_length
            )
            speculative_tokens, speculative_stats = speculative_decode(
                model, prompt, args.new_tokens, max_context_length
            )

            all_matched &= bool(torch.equal(greedy_tokens, speculative_tokens))
            greedy_total = _merge(greedy_total, greedy_stats)
            speculative_total = _merge(speculative_total, speculative_stats)
            if sample is None:
                sample = tokenizer.decode(greedy_tokens.tolist())

    print(f"config: {args.config}")
    print(f"checkpoint: {args.checkpoint or 'none (random weights)'}")
    print(f"prompts: {len(PROMPTS)} x {args.new_tokens} generated tokens")
    report("greedy autoregressive", greedy_total, True)
    report("speculative decoding", speculative_total, all_matched)

    speedup = speculative_total.tokens_per_forward / greedy_total.tokens_per_forward
    print(f"\nforwards saved: {speedup:.2f}x")
    print(f"\nsample generation (greedy):\n---\n{sample}\n---")
    if not all_matched:
        raise SystemExit("ERROR: speculative output differs from greedy")


def _merge(total, stats):
    if total is None:
        return stats
    total.generated_tokens += stats.generated_tokens
    total.verification_forwards += stats.verification_forwards
    total.seconds += stats.seconds
    while len(total.proposed_per_slot) < len(stats.proposed_per_slot):
        total.proposed_per_slot.append(0)
        total.accepted_per_slot.append(0)
    for slot, (proposed, accepted) in enumerate(
        zip(stats.proposed_per_slot, stats.accepted_per_slot)
    ):
        total.proposed_per_slot[slot] += proposed
        total.accepted_per_slot[slot] += accepted
    return total


if __name__ == "__main__":
    main()
