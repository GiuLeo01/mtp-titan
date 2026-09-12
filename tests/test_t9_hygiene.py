"""T9: shape/dtype, determinism, and checkpoint save/reload."""

import torch

from mtp_titan.architectures import gloeckle_model_config

from .conftest import (
    requires_cuda,
    NUM_SEGMENTS,
    SEED,
    SEGMENT_LENGTH,
    SHAPE,
    VOCAB_SIZE,
)


@requires_cuda
def test_output_shape_and_dtype(build_gloeckle, packed_batch, parallel_dims, parallelism):
    model = build_gloeckle(num_heads=2)
    inputs, _labels, extra_kwargs = model.preprocess_inputs(
        dict(packed_batch), parallel_dims=parallel_dims, parallelism=parallelism
    )
    with torch.no_grad():
        logits = model(inputs, **extra_kwargs)

    num_tokens = NUM_SEGMENTS * SEGMENT_LENGTH
    parameter_dtype = next(model.parameters()).dtype
    assert len(logits) == 2
    for head_logits in logits:
        assert head_logits.shape == (num_tokens, VOCAB_SIZE)
        assert head_logits.dtype == parameter_dtype


@requires_cuda
def test_determinism(build_gloeckle, packed_batch, parallel_dims, parallelism):
    model_a = build_gloeckle(num_heads=2)
    model_b = build_gloeckle(num_heads=2)

    inputs, _labels, extra_kwargs = model_a.preprocess_inputs(
        dict(packed_batch), parallel_dims=parallel_dims, parallelism=parallelism
    )
    with torch.no_grad():
        logits_a = model_a(inputs, **extra_kwargs)
        logits_b = model_b(inputs, **extra_kwargs)

    for head_a, head_b in zip(logits_a, logits_b):
        assert torch.equal(head_a, head_b)


@requires_cuda
def test_checkpoint_round_trip(
    build_gloeckle, packed_batch, parallel_dims, parallelism, device, tmp_path
):
    model = build_gloeckle(num_heads=2)
    inputs, _labels, extra_kwargs = model.preprocess_inputs(
        dict(packed_batch), parallel_dims=parallel_dims, parallelism=parallelism
    )
    with torch.no_grad():
        expected_logits = model(inputs, **extra_kwargs)

    checkpoint_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint_path)

    torch.manual_seed(SEED + 1)
    with torch.device(device):
        reloaded = gloeckle_model_config(
            SHAPE, num_heads=2, vocab_size=VOCAB_SIZE, seq_len=SEGMENT_LENGTH
        ).build()
        reloaded.init_states(buffer_device=device)

    with torch.no_grad():
        before_load_logits = reloaded(inputs, **extra_kwargs)
    assert not torch.equal(before_load_logits[0], expected_logits[0])

    reloaded.load_state_dict(torch.load(checkpoint_path, weights_only=True))

    with torch.no_grad():
        reloaded_logits = reloaded(inputs, **extra_kwargs)

    for expected_head, reloaded_head in zip(expected_logits, reloaded_logits):
        assert torch.equal(expected_head, reloaded_head)
