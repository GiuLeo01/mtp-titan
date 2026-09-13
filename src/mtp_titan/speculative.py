"""Self-speculative decoding: the MTP predictions are the draft, the next
forward is the verification. Greedy throughout, so the verified output is
exactly the greedy autoregressive one."""

from dataclasses import dataclass, field

import torch

from .deepseek import DeepSeekModel


@dataclass
class DecodeStats:
    generated_tokens: int = 0
    verification_forwards: int = 0
    proposed_per_slot: list[int] = field(default_factory=list)
    accepted_per_slot: list[int] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def tokens_per_forward(self) -> float:
        return self.generated_tokens / max(self.verification_forwards, 1)

    @property
    def tokens_per_second(self) -> float:
        return self.generated_tokens / self.seconds if self.seconds else 0.0

    def acceptance_rates(self) -> list[float]:
        return [
            accepted / proposed if proposed else 0.0
            for accepted, proposed in zip(self.accepted_per_slot, self.proposed_per_slot)
        ]

    def record_draft(self, num_proposed: int, num_accepted: int) -> None:
        while len(self.proposed_per_slot) < num_proposed:
            self.proposed_per_slot.append(0)
            self.accepted_per_slot.append(0)
        for slot in range(num_proposed):
            self.proposed_per_slot[slot] += 1
            if slot < num_accepted:
                self.accepted_per_slot[slot] += 1


def accepted_prefix_length(
    drafted: torch.Tensor, greedy: torch.Tensor
) -> int:
    """Leading drafted tokens that match what greedy would have produced.
    Verification stops at the first mismatch: everything after it was
    conditioned on a token the model would not have chosen."""
    matches = drafted == greedy[: drafted.numel()]
    mismatches = (~matches).nonzero()
    return int(mismatches[0]) if mismatches.numel() else int(drafted.numel())


def _decode_inputs(model, tokens: torch.Tensor, max_context_length: int):
    positions = torch.arange(tokens.numel(), device=tokens.device)
    attention_masks = model.get_attention_masks(
        positions=positions,
        padding_mask=None,
        max_num_documents=None,
        max_context_length=max_context_length,
    )
    return positions, attention_masks


class GloeckleDrafter:
    """Gloeckle et al., ICML 2024, §3: the heads are independent, so the
    training forward already holds every prediction the draft needs."""

    def __init__(self, model, sequence, positions, attention_masks):
        predictions = model(sequence, positions, attention_masks)
        self.predictions = (
            predictions if isinstance(predictions, tuple) else (predictions,)
        )
        self.main_logits = self.predictions[0]
        self.no_draft = sequence.new_empty(0)

    def draft(self, position: int) -> torch.Tensor:
        if len(self.predictions) == 1:
            return self.no_draft
        return torch.stack(
            [
                self.predictions[k][position].argmax()
                for k in range(1, len(self.predictions))
            ]
        )


class DeepSeekDrafter:
    """DeepSeek-V3 §2.2, eq. 21 takes Emb(t_{i+k}), the real future token. At
    inference that token does not exist yet, so the chain is fed its own
    predictions instead -- the one place where drafting is not the training
    forward."""

    def __init__(self, model, sequence, positions, attention_masks):
        self.model = model
        self.positions = positions
        self.attention_masks = attention_masks
        self.embeddings = model.tok_embeddings(sequence)
        self.trunk_hidden = model.trunk_hidden_states(
            self.embeddings, positions, attention_masks
        )
        self.main_logits = model.decode(self.trunk_hidden)
        self.no_draft = sequence.new_empty(0)

    def _shifted_embeddings(
        self, depth: int, position: int, predicted: list[torch.Tensor]
    ) -> torch.Tensor:
        """Row j must hold Emb(t_{j+depth}). The rows up to position-depth read
        it off the verified prefix; the last `depth` rows before `position` have
        no such token yet and take the predictions made so far instead."""
        shifted = torch.cat(
            [
                self.embeddings[depth:],
                self.embeddings.new_zeros(depth, self.embeddings.shape[-1]),
            ]
        )
        shifted[position - depth + 1 : position + 1] = self.model.tok_embeddings(
            torch.stack(predicted)
        )
        return shifted

    def draft(self, position: int) -> torch.Tensor:
        predicted = [self.main_logits[position].argmax()]
        hidden = self.trunk_hidden
        drafted = []
        for depth, mtp_module in enumerate(self.model.mtp_modules.values(), start=1):
            shifted = self._shifted_embeddings(depth, position, predicted)
            hidden = mtp_module(hidden, shifted, self.attention_masks, self.positions)
            token = self.model.decode(hidden)[position].argmax()
            drafted.append(token)
            predicted.append(token)
        return torch.stack(drafted) if drafted else self.no_draft


def make_drafter(model, sequence, positions, attention_masks):
    drafter = DeepSeekDrafter if isinstance(model, DeepSeekModel) else GloeckleDrafter
    return drafter(model, sequence, positions, attention_masks)


@torch.no_grad()
def greedy_decode(
    model, prompt: torch.Tensor, max_new_tokens: int, max_context_length: int
) -> tuple[torch.Tensor, DecodeStats]:
    stats = DecodeStats()
    tokens = prompt
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    for _ in range(max_new_tokens):
        positions, attention_masks = _decode_inputs(model, tokens, max_context_length)
        predictions = model(tokens, positions, attention_masks)
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        tokens = torch.cat([tokens, predictions[-1].argmax().view(1)])
        stats.verification_forwards += 1
        stats.generated_tokens += 1

    end.record()
    torch.cuda.synchronize()
    stats.seconds = start.elapsed_time(end) / 1000.0
    return tokens, stats


@torch.no_grad()
def speculative_decode(
    model, prompt: torch.Tensor, max_new_tokens: int, max_context_length: int
) -> tuple[torch.Tensor, DecodeStats]:
    if isinstance(model, DeepSeekModel):
        assert prompt.numel() > len(model.mtp_modules), (
            "the DeepSeek draft overwrites the last D rows of the shifted "
            "embeddings, so the prompt must be longer than D"
        )
    stats = DecodeStats()
    verified = prompt
    pending = prompt.new_empty(0)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    while stats.generated_tokens < max_new_tokens:
        sequence = torch.cat([verified, pending])
        positions, attention_masks = _decode_inputs(
            model, sequence, max_context_length
        )
        drafter = make_drafter(model, sequence, positions, attention_masks)
        stats.verification_forwards += 1

        last_verified = verified.numel() - 1
        greedy = drafter.main_logits[last_verified:].argmax(dim=-1)
        accepted = accepted_prefix_length(pending, greedy)
        if pending.numel():
            stats.record_draft(int(pending.numel()), accepted)

        verified = torch.cat(
            [verified, pending[:accepted], greedy[accepted].view(1)]
        )
        stats.generated_tokens += accepted + 1

        pending = drafter.draft(last_verified + accepted)

    end.record()
    torch.cuda.synchronize()
    stats.seconds = start.elapsed_time(end) / 1000.0
    # the last iteration can produce more than asked for; the surplus is
    # discarded but its forward is still counted, so alpha is understated
    stats.generated_tokens = min(stats.generated_tokens, max_new_tokens)
    return verified[: prompt.numel() + max_new_tokens], stats
