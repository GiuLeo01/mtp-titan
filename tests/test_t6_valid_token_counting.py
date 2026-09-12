"""T6: valid-token counting for loss normalization. The last i positions of
every packed segment have no target for head i, and must not be counted."""

from mtp_titan.loss import GloeckleLoss, TRAIN_HEAD_LOSSES

from .conftest import requires_cuda, NUM_SEGMENTS, SEGMENT_LENGTH, VOCAB_SIZE
from .test_t1_baseline_equivalence import run_forward_backward


@requires_cuda
def test_valid_token_counts(build_gloeckle, packed_batch, parallel_dims, parallelism):
    num_heads = 3
    model = build_gloeckle(num_heads=num_heads)
    loss_fn = GloeckleLoss.Config(global_vocab_size=VOCAB_SIZE).build()

    run_forward_backward(model, loss_fn, packed_batch, parallel_dims, parallelism)

    _loss_sums, valid_counts = GloeckleLoss.drain_head_losses(TRAIN_HEAD_LOSSES)

    num_tokens = NUM_SEGMENTS * SEGMENT_LENGTH
    for head_index in range(num_heads):
        expected = num_tokens - head_index * NUM_SEGMENTS
        assert int(valid_counts[head_index]) == expected, (
            f"head {head_index}: expected {expected} valid tokens, "
            f"got {int(valid_counts[head_index])}"
        )
