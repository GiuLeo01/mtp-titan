"""T6: valid-token counting for loss normalization. The last k positions of
every packed segment have no target for prediction k, and must not be counted."""

from mtp_titan.loss import MtpLoss, TRAIN_HEAD_LOSSES

from .conftest import requires_cuda, NUM_SEGMENTS, SEGMENT_LENGTH, VOCAB_SIZE
from .test_t1_baseline_equivalence import run_forward_backward


@requires_cuda
def test_valid_token_counts(build_mtp, packed_batch, parallel_dims, parallelism):
    num_predictions = 3
    model = build_mtp(num_predictions)
    loss_fn = MtpLoss.Config(global_vocab_size=VOCAB_SIZE).build()

    run_forward_backward(model, loss_fn, packed_batch, parallel_dims, parallelism)

    _loss_sums, valid_counts = MtpLoss.drain_head_losses(TRAIN_HEAD_LOSSES)

    num_tokens = NUM_SEGMENTS * SEGMENT_LENGTH
    for prediction in range(num_predictions):
        expected = num_tokens - prediction * NUM_SEGMENTS
        assert int(valid_counts[prediction]) == expected, (
            f"prediction {prediction}: expected {expected} valid tokens, "
            f"got {int(valid_counts[prediction])}"
        )
