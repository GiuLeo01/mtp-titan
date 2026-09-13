"""T5: the memory-efficient loss must match the naive one, loss and gradients."""

import pytest
import torch

from mtp_titan.loss import MtpLoss, MtpMemoryEfficientLoss

from .conftest import requires_cuda, VOCAB_SIZE


def run_forward_backward(
    model, loss_fn, batch, parallel_dims, parallelism, global_valid_tokens
):
    model.zero_grad(set_to_none=True)
    inputs, labels, extra_kwargs = model.preprocess_inputs(
        dict(batch), parallel_dims=parallel_dims, parallelism=parallelism
    )
    predictions = model(inputs, **extra_kwargs)
    loss, _ = loss_fn(predictions, labels, global_valid_tokens)
    loss.backward()
    gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return loss.detach(), gradients


@requires_cuda
@pytest.mark.parametrize("head_weights", [None, (1.0, 0.15)])
def test_memory_efficient_matches_naive(
    build_mtp, head_weights, packed_batch, parallel_dims, parallelism, device
):
    naive = build_mtp(2)
    efficient = build_mtp(2)
    efficient.load_state_dict(naive.state_dict(), strict=True)

    global_valid_tokens = torch.tensor(
        float(packed_batch["input"].numel()), device=device
    )

    naive_loss, naive_gradients = run_forward_backward(
        naive,
        MtpLoss.Config(
            global_vocab_size=VOCAB_SIZE, head_weights=head_weights
        ).build(),
        packed_batch,
        parallel_dims,
        parallelism,
        global_valid_tokens,
    )

    efficient._skip_lm_head = True
    efficient_loss_fn = MtpMemoryEfficientLoss.Config(
        global_vocab_size=VOCAB_SIZE, head_weights=head_weights
    ).build()
    efficient_loss_fn.set_lm_head(efficient.lm_head)

    efficient_loss, efficient_gradients = run_forward_backward(
        efficient,
        efficient_loss_fn,
        packed_batch,
        parallel_dims,
        parallelism,
        global_valid_tokens,
    )

    assert torch.allclose(naive_loss, efficient_loss)
    assert set(naive_gradients) == set(efficient_gradients)
    for name, naive_gradient in naive_gradients.items():
        assert torch.allclose(
            naive_gradient, efficient_gradients[name], atol=1e-6
        ), name
