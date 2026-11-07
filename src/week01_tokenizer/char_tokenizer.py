"""Character tokenizer — WEEK 1, WEDNESDAY EXERCISE.

The simplest possible tokenizer: one token per character. Useless in production,
perfect for understanding what a tokenizer actually is before BPE complicates it.

Your job: make tests/exercises/test_week01_tokenizer.py pass, including every
edge case. Run them with:  make exercises
"""

from __future__ import annotations


class UnknownCharacterError(KeyError):
    """Raised when decoding/encoding a character outside the vocabulary.

    Raise this (not a bare KeyError) so the caller gets a message they can act on.
    """


class CharTokenizer:
    """Maps characters <-> integer token IDs.

    The vocabulary is built from a training corpus, sorted for determinism, so the
    same corpus always produces the same IDs.
    """

    def __init__(self, corpus: str) -> None:
        raise NotImplementedError("Week 1, Wednesday. Build stoi/itos from sorted(set(corpus)).")

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        """text -> token IDs. Raise UnknownCharacterError on an unseen character."""
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        """token IDs -> text. Must satisfy decode(encode(s)) == s."""
        raise NotImplementedError
