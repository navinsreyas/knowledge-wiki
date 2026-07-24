"""Scan vault/raw/ for new markdown files and add them to the ingest queue."""

import hashlib
import json
from pathlib import Path

QUEUE_FILE = Path("pipeline/.ingest_queue.json")


def hash_file(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()[:8]


def enqueue_new(raw_dir: Path = Path("vault/raw")) -> list:
    queue: dict = {}
    if QUEUE_FILE.exists():
        queue = json.loads(QUEUE_FILE.read_text(encoding="utf-8"))

    newly_queued = []
    for filepath in sorted(raw_dir.rglob("*.md")):
        if ".keep" in filepath.name:
            continue
        h = hash_file(filepath)
        if h not in queue:
            queue[h] = str(filepath)
            newly_queued.append(filepath)
            print(f"Queued: {filepath.name}")

    QUEUE_FILE.write_text(json.dumps(queue, indent=2), encoding="utf-8")
    print(f"Total new: {len(newly_queued)} files queued")
    return newly_queued


if __name__ == "__main__":
    enqueue_new()
