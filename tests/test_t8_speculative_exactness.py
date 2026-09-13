"""T8: speculative decoding must be exact. The verified output has to be the
greedy autoregressive output, token for token, for every variant."""

import pytest
import torch

from mtp_titan.speculative import (
    accepted_prefix_length,
    greedy_decode,
    speculative_decode,
)

from .conftest import requires_cuda, SEGMENT_LENGTH

NUM_PREDICTIONS = 3
PROMPT_LENGTH = 8
NEW_TOKENS = 16

assert PROMPT_LENGTH + NEW_TOKENS + NUM_PREDICTIONS <= SEGMENT_LENGTH, (
    "generation must stay inside the RoPE table the test model was built with"
)


@pytest.mark.parametrize(
    "drafted, greedy, expected",
    [
        ([], [5], 0),
        ([7], [7, 1], 1),
        ([7, 3], [7, 3, 9], 2),
        ([7, 3], [7, 4, 9], 1),
        ([7, 3], [2, 3, 9], 0),
        ([7, 3, 5], [7, 3, 5, 1], 3),
        ([7, 3, 5], [7, 9, 5, 1], 1),
    ],
)
def test_accepted_prefix_length(drafted, greedy, expected):
    """Verification stops at the first mismatch even if a later draft token
    would have matched: it was conditioned on a token greedy did not pick."""
    assert (
        accepted_prefix_length(torch.tensor(drafted), torch.tensor(greedy)) == expected
    )


@requires_cuda
def test_speculative_output_equals_greedy(build_mtp, device):
    model = build_mtp(NUM_PREDICTIONS)
    torch.manual_seed(7)
    prompt = torch.randint(0, 64, (PROMPT_LENGTH,), device=device)

    greedy_tokens, greedy_stats = greedy_decode(
        model, prompt, NEW_TOKENS, SEGMENT_LENGTH
    )
    speculative_tokens, speculative_stats = speculative_decode(
        model, prompt, NEW_TOKENS, SEGMENT_LENGTH
    )

    assert torch.equal(greedy_tokens, speculative_tokens)
    assert greedy_stats.generated_tokens == speculative_stats.generated_tokens
    assert speculative_stats.verification_forwards <= greedy_stats.verification_forwards


@requires_cuda
def test_baseline_speculative_degrades_to_autoregressive(build_baseline, device):
    """With no MTP predictions there is nothing to draft, so the same loop must
    fall back to one token per forward."""
    model = build_baseline()
    torch.manual_seed(7)
    prompt = torch.randint(0, 64, (PROMPT_LENGTH,), device=device)

    greedy_tokens, _ = greedy_decode(model, prompt, NEW_TOKENS, SEGMENT_LENGTH)
    speculative_tokens, stats = speculative_decode(
        model, prompt, NEW_TOKENS, SEGMENT_LENGTH
    )

    assert torch.equal(greedy_tokens, speculative_tokens)
    assert stats.tokens_per_forward == 1.0
