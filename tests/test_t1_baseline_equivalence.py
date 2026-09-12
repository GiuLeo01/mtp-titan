"""T1: with n=1, Gloeckle MTP must reproduce the baseline exactly."""

import torch

from torchtitan.components.loss import CrossEntropyLoss

from mtp_titan.loss import GloeckleLoss

from .conftest import requires_cuda, VOCAB_SIZE


def run_forward_backward(model, loss_fn, batch, parallel_dims, parallelism):
    model.zero_grad(set_to_none=True)
    inputs, labels, extra_kwargs = model.preprocess_inputs(
        dict(batch), parallel_dims=parallel_dims, parallelism=parallelism
    )
    predictions = model(inputs, **extra_kwargs)
    loss, _ = loss_fn(predictions, labels)
    loss.backward()
    gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return loss.detach(), gradients


def gloeckle_state_dict_from_baseline(baseline_state_dict, trunk_depth, num_heads=1):
    gloeckle_state_dict = dict()
    head_prefixes = {
        f"layers.{trunk_depth + i}.": f"heads.{i}." for i in range(num_heads)
    }

    for key, value in baseline_state_dict.items():
        new_key = key
        for baseline_prefix, head_prefix in head_prefixes.items():
            if key.startswith(baseline_prefix):
                new_key = head_prefix + key[len(baseline_prefix):]
                break
        gloeckle_state_dict[new_key] = value

    return gloeckle_state_dict


@requires_cuda
def test_gloeckle_n1_matches_baseline(
    build_baseline, build_gloeckle, packed_batch, parallel_dims, parallelism
):
    baseline = build_baseline()
    gloeckle = build_gloeckle(num_heads=1)

    trunk_depth = len(gloeckle.layers)
    gloeckle.load_state_dict(
        gloeckle_state_dict_from_baseline(baseline.state_dict(), trunk_depth),
        strict=True,
    )

    baseline_loss, baseline_gradients = run_forward_backward(
        baseline,
        CrossEntropyLoss.Config(global_vocab_size=VOCAB_SIZE).build(),
        packed_batch,
        parallel_dims,
        parallelism,
    )
    gloeckle_loss, gloeckle_gradients = run_forward_backward(
        gloeckle,
        GloeckleLoss.Config(global_vocab_size=VOCAB_SIZE).build(),
        packed_batch,
        parallel_dims,
        parallelism,
    )

    assert torch.equal(baseline_loss, gloeckle_loss)

    expected_gradients = gloeckle_state_dict_from_baseline(
        baseline_gradients, trunk_depth
    )
    assert set(expected_gradients) == set(gloeckle_gradients)
    for name, expected in expected_gradients.items():
        assert torch.equal(expected, gloeckle_gradients[name])
