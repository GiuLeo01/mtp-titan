import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tokenizers import decoders, pre_tokenizers, Tokenizer, trainers
from tokenizers.models import BPE

BOS_TOKEN = "<|begin_of_text|>"
EOS_TOKEN = "<|end_of_text|>"
PAD_TOKEN = "<|pad|>"


def texts_up_to(dataset, text_field, max_characters):
    seen = 0
    for sample in dataset:
        text = sample[text_field]
        seen += len(text)
        yield text
        if seen >= max_characters:
            return


def train_tokenizer(dataset, text_field, vocab_size, max_characters):
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[BOS_TOKEN, EOS_TOKEN, PAD_TOKEN],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(
        texts_up_to(dataset, text_field, max_characters), trainer=trainer
    )
    return tokenizer


def save_tokenizer(tokenizer, output_dir, model_max_length):
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_dir / "tokenizer.json"))
    config = {
        "bos_token": BOS_TOKEN,
        "eos_token": EOS_TOKEN,
        "pad_token": PAD_TOKEN,
        "add_bos_token": True,
        "add_eos_token": True,
        "model_max_length": model_max_length,
        "tokenizer_class": "PreTrainedTokenizerFast",
    }
    (output_dir / "tokenizer_config.json").write_text(json.dumps(config, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="bigcode/starcoderdata")
    parser.add_argument("--data-dir", default="python")
    parser.add_argument("--text-field", default="content")
    parser.add_argument("--vocab-size", type=int, default=16384)
    parser.add_argument("--max-characters", type=int, default=500_000_000)
    parser.add_argument("--model-max-length", type=int, default=2048)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dataset = load_dataset(
        args.dataset, data_dir=args.data_dir, split="train", streaming=True
    )
    tokenizer = train_tokenizer(
        dataset, args.text_field, args.vocab_size, args.max_characters
    )
    save_tokenizer(tokenizer, Path(args.output), args.model_max_length)

    print(f"vocab size: {tokenizer.get_vocab_size()}")
    print(f"saved to: {args.output}")


if __name__ == "__main__":
    main()
