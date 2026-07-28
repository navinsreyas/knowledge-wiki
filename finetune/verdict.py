"""Pure-logic verdict parser — no external dependencies."""


def _parse_verdict(raw: str) -> bool:
    """Return True iff the judge replied with exactly CORRECT (case-insensitive).

    Exact match prevents "CORRECT" matching inside "INCORRECT" (the original bug)
    and rejects multi-word responses like "The answer is CORRECT" — the judge
    prompt instructs a single-word reply, so anything else is treated as failure.
    """
    v = raw.strip().upper()
    return v == "CORRECT"
