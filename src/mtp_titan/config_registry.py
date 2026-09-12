from pathlib import Path

import torchtitan
from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    GrainDataLoader,
    HuggingFaceRandomAccessSource,
    HuggingFaceStreamingSource,
    SingleDatasetConfig,
)
from torchtitan.components.loss import BaseLoss, ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optimizer import default_adamw, LRSchedulersContainer
from torchtitan.components.validate import Validator
from torchtitan.config import DebugConfig, ParallelismConfig, TrainingConfig
from torchtitan.hf_datasets.text_datasets import TextProcessor
from torchtitan.tools.profiler import Profiler
from torchtitan.trainer import Trainer

from torchtitan.protocols.model_spec import ModelSpec

from .architectures import gloeckle_model_spec, model_spec, SHAPES
from .loss import GloeckleLoss
from .metrics import MtpMetricsProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TORCHTITAN_ROOT = Path(torchtitan.__file__).resolve().parents[1]

SEQ_LEN = 2048
VOCAB_SIZE = 16384
TOKENIZER_PATH = str(PROJECT_ROOT / "assets" / "tokenizer-code-16k")

TOKENS_PER_TRAIN_STEP = 131072
TOKENS_PER_MICROBATCH_PER_DP_RANK = 16384

CHINCHILLA_TOKENS_PER_PARAMETER = 20

LR_ANCHOR_DIM = 256
LR_ANCHOR_VALUE = 8e-4

DEBUG_TOKENIZER_PATH = str(TORCHTITAN_ROOT / "tests" / "assets" / "tokenizer")
DEBUG_VOCAB_SIZE = 2048

GLOECKLE_NUM_HEADS = 2

PROFILE_STEPS = 30
PROFILE_FREQ = 10

STARCODER_SHARD_COUNT = 59
STARCODER_VALIDATION_SHARDS = (58,)


def _starcoder_content(sample):
    return sample["content"]


def _starcoder_python(shard_indices) -> SingleDatasetConfig:
    return SingleDatasetConfig(
        source=HuggingFaceStreamingSource.Config(
            path="bigcode/starcoderdata",
            split="train",
            load_dataset_kwargs={
                "data_files": [
                    f"python/train-{index:05d}-of-{STARCODER_SHARD_COUNT:05d}.parquet"
                    for index in shard_indices
                ]
            },
        ),
        processor=TextProcessor.Config(text_fn=_starcoder_content),
        post_filters=(lambda sample: sample is not None,),
    )


STARCODER_PYTHON_TRAIN = _starcoder_python(
    [i for i in range(STARCODER_SHARD_COUNT) if i not in STARCODER_VALIDATION_SHARDS]
)
STARCODER_PYTHON_VALIDATION = _starcoder_python(STARCODER_VALIDATION_SHARDS)

SMOKE_CORPUS = SingleDatasetConfig(
    source=HuggingFaceRandomAccessSource.Config(
        path="json",
        split="train",
        load_dataset_kwargs={
            "data_files": str(TORCHTITAN_ROOT / "tests" / "assets" / "c4_test" / "data.json"),
        },
    ),
    processor=TextProcessor.Config(),
    post_filters=(lambda sample: sample is not None,),
)


def scaled_learning_rate(dim: int) -> float:
    return LR_ANCHOR_VALUE * (LR_ANCHOR_DIM / dim) ** 0.5


def chinchilla_steps(shape_name: str) -> int:
    tokens = CHINCHILLA_TOKENS_PER_PARAMETER * SHAPES[shape_name].non_embedding_parameters
    return tokens // TOKENS_PER_TRAIN_STEP


def _recipe(
    shape_name: str,
    *,
    spec: ModelSpec,
    loss: BaseLoss.Config,
    run_name: str,
    hf_assets_path: str,
    train_dataset: SingleDatasetConfig,
    validation_dataset: SingleDatasetConfig,
    steps: int,
    seed: int,
) -> Trainer.Config:
    packed_train = ConcatThenSplitPackingConfig(dataset=train_dataset)
    packed_validation = ConcatThenSplitPackingConfig(dataset=validation_dataset)
    return Trainer.Config(
        model_spec=spec,
        hf_assets_path=hf_assets_path,
        loss=loss,
        optimizer=default_adamw(lr=scaled_learning_rate(SHAPES[shape_name].dim)),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=max(steps // 100, 1),
            decay_ratio=0.9,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=TOKENS_PER_MICROBATCH_PER_DP_RANK,
            num_tokens_per_train_step=TOKENS_PER_TRAIN_STEP,
            max_context_length=SEQ_LEN,
            steps=steps,
        ),
        debug=DebugConfig(seed=seed),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=8,
            data_parallel_shard_degree=1,
        ),
        dataloader=GrainDataLoader.Config(dataset=packed_train, shuffle=True, seed=seed),
        validator=Validator.Config(
            enable=True,
            freq=max(steps // 20, 1),
            steps=20,
            dataloader=GrainDataLoader.Config(
                dataset=packed_validation, shuffle=False, seed=seed
            ),
        ),
        metrics=MtpMetricsProcessor.Config(
            log_freq=10, enable_wandb=True, run_name=run_name
        ),
        checkpoint=CheckpointManager.Config(
            enable=True, interval=max(steps // 4, 1)
        ),
        activation_checkpoint=None,
    )


def _baseline(
    shape_name: str,
    *,
    vocab_size: int,
    hf_assets_path: str,
    train_dataset: SingleDatasetConfig,
    validation_dataset: SingleDatasetConfig,
    steps: int,
    seed: int,
) -> Trainer.Config:
    return _recipe(
        shape_name,
        spec=model_spec(shape_name, vocab_size=vocab_size, seq_len=SEQ_LEN),
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(global_vocab_size=vocab_size),
        ),
        run_name=f"baseline-{shape_name}-seed{seed}",
        hf_assets_path=hf_assets_path,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        steps=steps,
        seed=seed,
    )


def _gloeckle(
    shape_name: str,
    *,
    num_heads: int,
    vocab_size: int,
    hf_assets_path: str,
    train_dataset: SingleDatasetConfig,
    validation_dataset: SingleDatasetConfig,
    steps: int,
    seed: int,
) -> Trainer.Config:
    return _recipe(
        shape_name,
        spec=gloeckle_model_spec(
            shape_name,
            num_heads=num_heads,
            vocab_size=vocab_size,
            seq_len=SEQ_LEN,
        ),
        loss=GloeckleLoss.Config(global_vocab_size=vocab_size),
        run_name=f"gloeckle-{shape_name}-n{num_heads}-seed{seed}",
        hf_assets_path=hf_assets_path,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        steps=steps,
        seed=seed,
    )


def baseline_17m(seed: int = 0) -> Trainer.Config:
    return _baseline(
        "17m",
        vocab_size=VOCAB_SIZE,
        hf_assets_path=TOKENIZER_PATH,
        train_dataset=STARCODER_PYTHON_TRAIN,
        validation_dataset=STARCODER_PYTHON_VALIDATION,
        steps=chinchilla_steps("17m"),
        seed=seed,
    )


def baseline_57m(seed: int = 0) -> Trainer.Config:
    return _baseline(
        "57m",
        vocab_size=VOCAB_SIZE,
        hf_assets_path=TOKENIZER_PATH,
        train_dataset=STARCODER_PYTHON_TRAIN,
        validation_dataset=STARCODER_PYTHON_VALIDATION,
        steps=chinchilla_steps("57m"),
        seed=seed,
    )


def baseline_101m(seed: int = 0) -> Trainer.Config:
    return _baseline(
        "101m",
        vocab_size=VOCAB_SIZE,
        hf_assets_path=TOKENIZER_PATH,
        train_dataset=STARCODER_PYTHON_TRAIN,
        validation_dataset=STARCODER_PYTHON_VALIDATION,
        steps=chinchilla_steps("101m"),
        seed=seed,
    )


def baseline_smoke(seed: int = 0) -> Trainer.Config:
    config = _baseline(
        "debug",
        vocab_size=DEBUG_VOCAB_SIZE,
        hf_assets_path=DEBUG_TOKENIZER_PATH,
        train_dataset=SMOKE_CORPUS,
        validation_dataset=SMOKE_CORPUS,
        steps=20,
        seed=seed,
    )
    config.parallelism = ParallelismConfig()
    config.metrics.log_freq = 1
    config.metrics.enable_wandb = False
    return config


def gloeckle_57m(seed: int = 0) -> Trainer.Config:
    return _gloeckle(
        "57m",
        num_heads=GLOECKLE_NUM_HEADS,
        vocab_size=VOCAB_SIZE,
        hf_assets_path=TOKENIZER_PATH,
        train_dataset=STARCODER_PYTHON_TRAIN,
        validation_dataset=STARCODER_PYTHON_VALIDATION,
        steps=chinchilla_steps("57m"),
        seed=seed,
    )


def gloeckle_57m_n4(seed: int = 0) -> Trainer.Config:
    return _gloeckle(
        "57m",
        num_heads=4,
        vocab_size=VOCAB_SIZE,
        hf_assets_path=TOKENIZER_PATH,
        train_dataset=STARCODER_PYTHON_TRAIN,
        validation_dataset=STARCODER_PYTHON_VALIDATION,
        steps=chinchilla_steps("57m"),
        seed=seed,
    )


def gloeckle_smoke(seed: int = 0) -> Trainer.Config:
    config = _gloeckle(
        "debug",
        num_heads=GLOECKLE_NUM_HEADS,
        vocab_size=DEBUG_VOCAB_SIZE,
        hf_assets_path=DEBUG_TOKENIZER_PATH,
        train_dataset=SMOKE_CORPUS,
        validation_dataset=SMOKE_CORPUS,
        steps=20,
        seed=seed,
    )
    config.parallelism = ParallelismConfig()
    config.metrics.log_freq = 1
    config.metrics.enable_wandb = False
    return config


def _for_profiling(config: Trainer.Config) -> Trainer.Config:
    config.parallelism = ParallelismConfig()
    config.training.steps = PROFILE_STEPS
    config.profiler = Profiler.Config(
        enable_profiling=True,
        profile_freq=PROFILE_FREQ,
        enable_memory_snapshot=True,
    )
    config.metrics.log_freq = 1
    config.metrics.enable_wandb = False
    config.checkpoint.enable = False
    config.validator.enable = False
    return config


def baseline_57m_profile(seed: int = 0) -> Trainer.Config:
    return _for_profiling(baseline_57m(seed=seed))


def gloeckle_57m_profile(seed: int = 0) -> Trainer.Config:
    return _for_profiling(gloeckle_57m(seed=seed))


def gloeckle_57m_n4_profile(seed: int = 0) -> Trainer.Config:
    return _for_profiling(gloeckle_57m_n4(seed=seed))


def gloeckle_smoke_profile(seed: int = 0) -> Trainer.Config:
    return _for_profiling(gloeckle_smoke(seed=seed))
