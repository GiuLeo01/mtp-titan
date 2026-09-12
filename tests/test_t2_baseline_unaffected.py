"""T2: the baseline path is unaffected by exercising the Gloeckle path."""

import torch

from torchtitan.components.loss import CrossEntropyLoss

from mtp_titan.loss import GloeckleLoss

from .conftest import requires_cuda, VOCAB_SIZE
from .test_t1_baseline_equivalence import run_forward_backward


@requires_cuda
def test_baseline_unaffected_by_gloeckle_usage(
    build_baseline, build_gloeckle, packed_batch, parallel_dims, parallelism
):
    baseline_before = build_baseline()
    loss_before, gradients_before = run_forward_backward(
        baseline_before,
        CrossEntropyLoss.Config(global_vocab_size=VOCAB_SIZE).build(),
        packed_batch,
        parallel_dims,
        parallelism,
    )

    gloeckle = build_gloeckle(num_heads=2)
    run_forward_backward(
        gloeckle,
        GloeckleLoss.Config(global_vocab_size=VOCAB_SIZE).build(),
        packed_batch,
        parallel_dims,
        parallelism,
    )

    baseline_after = build_baseline()
    loss_after, gradients_after = run_forward_backward(
        baseline_after,
        CrossEntropyLoss.Config(global_vocab_size=VOCAB_SIZE).build(),
        packed_batch,
        parallel_dims,
        parallelism,
    )

    assert torch.equal(loss_before, loss_after)
    assert set(gradients_before) == set(gradients_after)
    for name, gradient in gradients_before.items():
        assert torch.equal(gradient, gradients_after[name])
