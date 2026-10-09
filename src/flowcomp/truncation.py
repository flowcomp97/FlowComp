"""Truncation policies shared by training, data preparation and evaluation.

Appendix F.4 of the paper specifies, for long-document summarization, that

    "Documents exceeding model_max_length = 2048 (T5 tokens) use head+tail
    truncation: the first and last portions are retained in equal proportion
    up to the token budget.  An optional word-level cap (max_source_words)
    provides coarser control."

The ICL (F.2) and RAG (F.3) scenarios instead keep the leading context only,
so the head strategy remains the default and head+tail has to be selected
explicitly by the long-document data/eval entry points.
"""

from typing import Dict, List, Optional, Sequence, Tuple


SOURCE_TRUNCATION_CHOICES = ("head", "tail", "head_tail")
HEAD_TAIL_DEFAULT_RATIO = 0.5


def head_tail_budget(
    max_length: int,
    head_tail_ratio: float = HEAD_TAIL_DEFAULT_RATIO,
) -> Tuple[int, int]:
    """Split a token budget into (head, tail) sizes that sum to max_length.

    The paper keeps the first and last portions "in equal proportion", i.e.
    half of the budget each by default; the split is rounded down on the tail
    so that the head never loses a token when the budget is odd.
    """
    max_length = int(max_length)
    if max_length <= 0:
        return 0, 0
    ratio = float(min(max(head_tail_ratio, 0.0), 1.0))
    tail_budget = int(max_length * (1.0 - ratio))
    tail_budget = max(min(tail_budget, max_length), 0)
    return max_length - tail_budget, tail_budget


def truncate_token_ids(
    token_ids: Sequence[int],
    max_length: int,
    mode: str = "head",
    head_tail_ratio: float = HEAD_TAIL_DEFAULT_RATIO,
) -> List[int]:
    """Truncate ``token_ids`` to ``max_length`` under the requested policy.

    ``head`` keeps the leading tokens (the historical behaviour used by the
    ICL and RAG scenarios), ``tail`` keeps the trailing ones, and ``head_tail``
    keeps the first and last portions in equal proportion (Appendix F.4).
    Sequences that already fit are returned unchanged, so this is a no-op for
    the short-context scenarios.
    """
    ids = list(token_ids)
    max_length = int(max_length)
    if max_length <= 0:
        return []
    if len(ids) <= max_length:
        return ids
    if mode == "tail":
        return ids[-max_length:]
    if mode == "head_tail":
        head_budget, tail_budget = head_tail_budget(max_length, head_tail_ratio)
        pieces = []
        if head_budget:
            pieces.append(ids[:head_budget])
        if tail_budget:
            pieces.append(ids[len(ids) - tail_budget :])
        return [token for piece in pieces for token in piece]
    return ids[:max_length]


def truncate_words(
    text: str,
    max_words: int,
    mode: str = "head",
    head_tail_ratio: float = HEAD_TAIL_DEFAULT_RATIO,
) -> str:
    """Word-level counterpart of :func:`truncate_token_ids`.

    Appendix F.4 describes ``max_source_words`` as a coarser, word-level
    control that complements the token-level budget, so both policies are
    available at word granularity as well.
    """
    max_words = int(max_words)
    if max_words <= 0:
        return text
    words = text.split()
    if len(words) <= max_words:
        return text
    if mode == "tail":
        return " ".join(words[-max_words:])
    if mode == "head_tail":
        head_budget, tail_budget = head_tail_budget(max_words, head_tail_ratio)
        pieces = []
        if head_budget:
            pieces.append(words[:head_budget])
        if tail_budget:
            pieces.append(words[len(words) - tail_budget :])
        return " ".join(word for piece in pieces for word in piece)
    return " ".join(words[:max_words])


def source_token_ids(
    tokenizer,
    text: str,
    max_length: int,
    mode: str = "head",
    head_tail_ratio: float = HEAD_TAIL_DEFAULT_RATIO,
    add_special_tokens: bool = True,
    extra: Optional[Dict] = None,
) -> Dict[str, List[int]]:
    """Tokenize ``text`` for the source encoder and apply the truncation policy.

    Returns the dict produced by the tokenizer (``input_ids`` plus
    ``attention_mask`` when requested), truncated to exactly ``max_length``
    tokens.  Tokenizing without truncation first is required for head+tail,
    which cannot be expressed through the tokenizer's own ``truncation_side``.
    """
    encoded = tokenizer(
        text,
        truncation=False,
        padding=False,
        add_special_tokens=False,
        **(extra or {}),
    )
    content_budget = max_length
    if add_special_tokens:
        content_budget = max(max_length - tokenizer.num_special_tokens_to_add(pair=False), 0)
    input_ids = truncate_token_ids(
        encoded["input_ids"],
        content_budget,
        mode,
        head_tail_ratio,
    )
    if add_special_tokens:
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
        input_ids = input_ids[:max_length]
    encoded["input_ids"] = input_ids
    if "attention_mask" in encoded:
        encoded["attention_mask"] = [1] * len(input_ids)
    return encoded
