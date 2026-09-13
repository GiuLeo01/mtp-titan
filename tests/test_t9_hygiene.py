"""T9: shape/dtype, determinism, and checkpoint save/reload."""

import torch

from .conftest import (
    requires_cuda,
    mtp_model_config,
    NUM_SEGMENTS,
    SEED,
    SEGMENT_LENGTH,
    VOCAB_SIZE,
)


@requires_cuda
def test_output_shape_and_dtype(build_mtp, packed_batch, parallel_dims, parallelism):
    model = build_mtp(2)
    inputs, _labels, extra_kwargs = model.preprocess_inputs(
        dict(packed_batch), parallel_dims=parallel_dims, parallelism=parallelism
    )
    with torch.no_grad():
        logits = model(inputs, **extra_kwargs)

    num_tokens = NUM_SEGMENTS * SEGMENT_LENGTH
    parameter_dtype = next(model.parameters()).dtype
    assert len(logits) == 2
    for prediction_logits in logits:
        assert prediction_logits.shape == (num_tokens, VOCAB_SIZE)
        assert prediction_logits.dtype == parameter_dtype


@requires_cuda
def test_determinism(build_mtp, packed_batch, parallel_dims, parallelism):
    model_a = build_mtp(2)
    model_b = build_mtp(2)

    inputs, _labels, extra_kwargs = model_a.preprocess_inputs(
        dict(packed_batch), parallel_dims=parallel_dims, parallelism=parallelism
    )
    with torch.no_grad():
        logits_a = model_a(inputs, **extra_kwargs)
        logits_b = model_b(inputs, **extra_kwargs)

    for prediction_a, prediction_b in zip(logits_a, logits_b):
        assert torch.equal(prediction_a, prediction_b)


@requires_cuda
def test_checkpoint_round_trip(
    build_mtp, mtp_variant, packed_batch, parallel_dims, parallelism, device, tmp_path
):
    model = build_mtp(2)
    inputs, _labels, extra_kwargs = model.preprocess_inputs(
        dict(packed_batch), parallel_dims=parallel_dims, parallelism=parallelism
    )
    with torch.no_grad():
        expected_logits = model(inputs, **extra_kwargs)

    checkpoint_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint_path)

    torch.manual_seed(SEED + 1)
    with torch.device(device):
        reloaded = mtp_model_config(mtp_variant, 2).build()
        reloaded.init_states(buffer_device=device)

    with torch.no_grad():
        before_load_logits = reloaded(inputs, **extra_kwargs)
    assert not torch.equal(before_load_logits[0], expected_logits[0])

    reloaded.load_state_dict(torch.load(checkpoint_path, weights_only=True))

    with torch.no_grad():
        reloaded_logits = reloaded(inputs, **extra_kwargs)

    for expected, actual in zip(expected_logits, reloaded_logits):
        assert torch.equal(expected, actual)
