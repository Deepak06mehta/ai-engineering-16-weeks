"""WEEK 1, WEDNESDAY — character tokenizer. Run with: make exercises

These are red until you implement src/week01_tokenizer/char_tokenizer.py.
Every test maps to an edge case listed in the vault note "Tokenization".
"""

from __future__ import annotations

import pytest

from src.week01_tokenizer.char_tokenizer import CharTokenizer, UnknownCharacterError

pytestmark = pytest.mark.exercise

CORPUS = "the quick brown fox jumps over the lazy dog\nTHE QUICK BROWN FOX 0123456789"


@pytest.fixture
def tok() -> CharTokenizer:
    return CharTokenizer(CORPUS)


def test_vocab_is_sorted_unique_characters(tok):
    assert tok.vocab_size == len(set(CORPUS))


def test_roundtrip(tok):
    for s in ("the dog", "QUICK 42", "\n", "the quick brown fox"):
        assert tok.decode(tok.encode(s)) == s


def test_empty_string(tok):
    assert tok.encode("") == []
    assert tok.decode([]) == ""


def test_unknown_character_raises_clearly(tok):
    with pytest.raises(UnknownCharacterError):
        tok.encode("—")  # em dash is not in the corpus


def test_whitespace_runs_preserved(tok):
    s = "the   dog"
    assert tok.decode(tok.encode(s)) == s


def test_ids_are_stable_across_instances():
    """Same corpus -> same IDs. Determinism matters for checkpoint compatibility."""
    a, b = CharTokenizer(CORPUS), CharTokenizer(CORPUS)
    assert a.encode("dog") == b.encode("dog")


def test_unicode_roundtrip():
    t = CharTokenizer("नमस्ते world 🙂")
    for s in ("नमस्ते", "🙂", "world 🙂"):
        assert t.decode(t.encode(s)) == s
