"""T10: fidelity to DeepSeek-V3 §2.2 -- the M_k projection, the shared
embedding and output head, the sequential chain, and the λ/D loss weights."""

import torch

from mtp_titan.config_registry import DEEPSEEK_LAMBDA, DEEPSEEK_NUM_MODULES
from mtp_titan.deepseek import deepseek_head_weights
from mtp_titan.loss import MtpLoss, TRAIN_HEAD_LOSSES

from .conftest import requires_cuda, VOCAB_SIZE
from .test_t5_memory_efficient import run_forward_backward


def predict(model, batch, parallel_dims, parallelism):
    inputs, _labels, extra_kwargs = model.preprocess_inputs(
        dict(batch), parallel_dims=parallel_dims, parallelism=parallelism
    )
    with torch.no_grad():
        return model(inputs, **extra_kwargs)


@requires_cuda
def test_projection_is_d_by_two_d(build_deepseek):
    model = build_deepseek(num_modules=2)
    dim = model.config.dim
    for mtp_module in model.mtp_modules.values():
        assert mtp_module.projection.weight.shape == (dim, 2 * dim)
        assert mtp_module.projection.bias is None


@requires_cuda
def test_embedding_and_output_head_are_shared(build_deepseek):
    model = build_deepseek(num_modules=2)
    assert model.tok_embeddings.weight is model.lm_head.weight
    for mtp_module in model.mtp_modules.values():
        owned = dict(mtp_module.named_modules())
        assert not any("tok_embeddings" in name or "lm_head" in name for name in owned)


@requires_cuda
def test_both_branches_of_the_projection_are_used(
    build_deepseek, packed_batch, parallel_dims, parallelism
):
    """Eq. 21 concatenates RMSNorm(h^{k-1}) with RMSNorm(Emb(t_{i+k})): perturbing
    either normalization must move the logits, or half the input is being ignored."""
    model = build_deepseek(num_modules=1)
    reference = predict(model, packed_batch, parallel_dims, parallelism)

    for branch in ("hidden_norm", "embedding_norm"):
        with torch.no_grad():
            getattr(model.mtp_modules["0"], branch).weight.add_(0.1)
        perturbed = predict(model, packed_batch, parallel_dims, parallelism)
        with torch.no_grad():
            getattr(model.mtp_modules["0"], branch).weight.sub_(0.1)

        assert torch.equal(perturbed[0], reference[0]), branch
        assert not torch.equal(perturbed[1], reference[1]), branch


@requires_cuda
def test_chain_is_sequential(
    build_deepseek, packed_batch, parallel_dims, parallelism
):
    """Eq. 22 feeds h^k into depth k+1, so depth 1 must reach depth 2's logits.
    With Gloeckle's independent heads it would not."""
    model = build_deepseek(num_modules=2)
    reference = predict(model, packed_batch, parallel_dims, parallelism)

    with torch.no_grad():
        model.mtp_modules["0"].projection.weight.add_(0.1)
    perturbed = predict(model, packed_batch, parallel_dims, parallelism)

    assert torch.equal(perturbed[0], reference[0])
    assert not torch.equal(perturbed[1], reference[1])
    assert not torch.equal(perturbed[2], reference[2])


@requires_cuda
def test_chain_matches_a_hand_built_reference(
    build_deepseek, packed_batch, parallel_dims, parallelism
):
    """Eq. 22 feeds h^k to depth k+1 raw, because self.norm belongs to OutHead
    (eq. 23). Normalizing it on the way is invisible to shapes and to every
    other check here, so pin the whole chain against an independent rebuild."""
    model = build_deepseek(num_modules=2)
    inputs, _labels, extra_kwargs = model.preprocess_inputs(
        dict(packed_batch), parallel_dims=parallel_dims, parallelism=parallelism
    )
    positions = extra_kwargs.get("positions")
    attention_masks = extra_kwargs.get("attention_masks")

    with torch.no_grad():
        outputs = model(inputs, **extra_kwargs)

        embeddings = model.tok_embeddings(inputs)
        hidden = model.trunk_hidden_states(embeddings, positions, attention_masks)
        assert torch.equal(model.decode(hidden), outputs[0])

        for depth, mtp_module in enumerate(model.mtp_modules.values(), start=1):
            shifted = torch.cat(
                [embeddings[depth:], torch.zeros_like(embeddings[:depth])]
            )
            hidden = mtp_module(hidden, shifted, attention_masks, positions)
            assert torch.equal(model.decode(hidden), outputs[depth]), depth


def test_head_weights_follow_lambda_over_d():
    assert deepseek_head_weights(2, 0.3) == (1.0, 0.15, 0.15)
    assert deepseek_head_weights(1, 0.3) == (1.0, 0.3)
    assert deepseek_head_weights(DEEPSEEK_NUM_MODULES, DEEPSEEK_LAMBDA)[0] == 1.0


@requires_cuda
def test_loss_applies_lambda_over_d(
    build_deepseek, packed_batch, parallel_dims, parallelism, device
):
    """Eq. 25: L = L_main + (λ/D) Σ_k L^k_MTP, the main model unweighted."""
    num_modules = 2
    model = build_deepseek(num_modules=num_modules)
    head_weights = deepseek_head_weights(num_modules, DEEPSEEK_LAMBDA)
    global_valid_tokens = torch.tensor(
        float(packed_batch["input"].numel()), device=device
    )

    loss_fn = MtpLoss.Config(
        global_vocab_size=VOCAB_SIZE, head_weights=head_weights
    ).build()
    loss, _ = run_forward_backward(
        model, loss_fn, packed_batch, parallel_dims, parallelism, global_valid_tokens
    )

    cross_entropy_sums, _valid_counts = MtpLoss.drain_head_losses(TRAIN_HEAD_LOSSES)
    expected = sum(
        weight * float(cross_entropy_sums[index])
        for index, weight in enumerate(head_weights)
    ) / float(global_valid_tokens)

    assert abs(float(loss) - expected) < 1e-3 * abs(expected)
