"""Tests for the verdict parser used in evaluate.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from finetune.evaluate import _parse_verdict


def test_correct():
    assert _parse_verdict("CORRECT") is True


def test_incorrect():
    # Original bug: "CORRECT" in "INCORRECT" was True — must be False
    assert _parse_verdict("INCORRECT") is False


def test_correct_embedded_in_sentence():
    # Multi-word responses don't match; judge is told to reply with one word only
    assert _parse_verdict("The answer is CORRECT") is False


def test_garbage():
    assert _parse_verdict("xyz123 ???") is False
