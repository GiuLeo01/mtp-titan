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
