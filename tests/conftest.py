import pytest
import torch

from torchtitan.config import ParallelismConfig
from torchtitan.distributed.parallel_dims import ParallelDims

from mtp_titan.architectures import gloeckle_model_config, model_config
from mtp_titan.loss import MtpLoss

SHAPE = "debug"
VOCAB_SIZE = 512
SEGMENT_LENGTH = 32
NUM_SEGMENTS = 3
SEED = 0

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="no attention backend has a CPU backward; see CLAUDE.md",
)


@pytest.fixture(autouse=True)
def clear_gloeckle_loss_accumulators():
    MtpLoss.head_loss_accumulators.clear()
    yield
    MtpLoss.head_loss_accumulators.clear()


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda")


@pytest.fixture
def parallel_dims() -> ParallelDims:
    return ParallelDims(
        dp_replicate=1, dp_shard=1, cp=1, tp=1, pp=1, ep=1, world_size=1
    )


@pytest.fixture
def parallelism() -> ParallelismConfig:
    return ParallelismConfig(spmd_backend="partial_dtensor")


def _build(config, device: torch.device):
    torch.manual_seed(SEED)
    with torch.device(device):
        model = config.build()
        model.init_states(buffer_device=device)
    return model


@pytest.fixture
def build_baseline(device):
    def factory():
        return _build(
            model_config(
                SHAPE,
                vocab_size=VOCAB_SIZE,
                seq_len=SEGMENT_LENGTH,
                attn_backend="flex",
            ),
            device,
        )

    return factory


@pytest.fixture
def build_gloeckle(device):
    def factory(num_heads: int):
        return _build(
            gloeckle_model_config(
                SHAPE,
                num_heads=num_heads,
                vocab_size=VOCAB_SIZE,
                seq_len=SEGMENT_LENGTH,
                attn_backend="flex",
            ),
            device,
        )

    return factory


@pytest.fixture
def packed_batch(device) -> dict[str, torch.Tensor]:
    torch.manual_seed(SEED + 1)
    num_tokens = NUM_SEGMENTS * SEGMENT_LENGTH
    stream = torch.randint(0, VOCAB_SIZE, (num_tokens + 1,), device=device)
    positions = torch.arange(SEGMENT_LENGTH, device=device).repeat(NUM_SEGMENTS)
    return {
        "input": stream[:-1],
        "labels": stream[1:],
        "positions": positions,
    }
