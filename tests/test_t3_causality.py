"""T3: causality. d loss(target t+k) / d input(pos > i) must be exactly 0."""

import torch

from .conftest import requires_cuda, SEGMENT_LENGTH


class _FixedEmbedding(torch.nn.Module):
    def __init__(self, embeddings):
        super().__init__()
        self.embeddings = embeddings

    def forward(self, tokens):
        return self.embeddings


@requires_cuda
def test_no_future_leakage(build_gloeckle, packed_batch, parallel_dims, parallelism):
    model = build_gloeckle(num_heads=2)
    inputs, _labels, extra_kwargs = model.preprocess_inputs(
        dict(packed_batch), parallel_dims=parallel_dims, parallelism=parallelism
    )
    position = SEGMENT_LENGTH // 2

    for head_index in range(2):
        embeddings = model.tok_embeddings(inputs).detach().clone().requires_grad_(True)
        model.tok_embeddings = _FixedEmbedding(embeddings)
        logits = model(inputs, **extra_kwargs)

        scalar = logits[head_index][position].sum()
        grad, = torch.autograd.grad(scalar, embeddings)

        future = grad[position + 1:]
        past = grad[:position + 1]

        assert torch.equal(future, torch.zeros_like(future))
        assert past.abs().sum() > 0


@requires_cuda
def test_no_leakage_beyond_teacher_forcing_horizon(
    build_deepseek, packed_batch, parallel_dims, parallelism
):
    """DeepSeek-V3 §2.2, eq. 21: depth k legitimately reads Emb(t_{i+k}), so its
    horizon is i+k -- and nothing beyond it may be read."""
    num_modules = 2
    model = build_deepseek(num_modules=num_modules)
    inputs, _labels, extra_kwargs = model.preprocess_inputs(
        dict(packed_batch), parallel_dims=parallel_dims, parallelism=parallelism
    )
    position = SEGMENT_LENGTH // 2

    for depth in range(num_modules + 1):
        embeddings = model.tok_embeddings(inputs).detach().clone().requires_grad_(True)
        model.tok_embeddings = _FixedEmbedding(embeddings)
        logits = model(inputs, **extra_kwargs)

        scalar = logits[depth][position].sum()
        grad, = torch.autograd.grad(scalar, embeddings)

        horizon = position + depth
        beyond = grad[horizon + 1:]

        assert torch.equal(beyond, torch.zeros_like(beyond))
        assert grad[:horizon + 1].abs().sum() > 0
        assert grad[horizon].abs().sum() > 0
