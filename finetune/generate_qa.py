"""Generate Q&A pairs from wiki articles and split into train/val/test sets."""

import json
import random
import re
import sys
from pathlib import Path

from langchain_ollama import OllamaLLM

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

QA_PROMPT = """
Generate {n} high-quality Q&A pairs from this wiki article.

Requirements:
- Questions must be genuinely useful to an AI Engineer
- Answers come directly from the article only — no additions or hallucinations
- Mix question types: 2 factual, 2 conceptual, 1 application
- Answers must be complete and self-contained (no "as mentioned above")
- Output ONLY valid JSON with no markdown fences or preamble:
[{{"question": "...", "answer": "..."}}]

Article:
{article}
"""

WIKI_DIR = Path("vault/wiki")
FINETUNE_DIR = Path("finetune")


def clean_json_string(raw: str) -> str:
    raw = raw.strip()
    raw = raw.lstrip("```json").lstrip("```").rstrip("```").strip()
    raw = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", raw)
    raw = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", raw)
    return raw


def validate_pair(pair: dict, source: str, llm) -> bool:
    prompt = (
        f"Article: {source[:1500]}\n\n"
        f"Claim: {pair['answer']}\n\n"
        f"Is this claim directly supported by the article? Reply only: YES or NO"
    )
    response = llm.invoke(prompt).strip()
    return response.upper().startswith("YES")


def generate_and_split(pairs_per: int = 5):
    llm = OllamaLLM(model=config.MODEL_NAME, temperature=0.3)

    skip_names = {"_index.md", "log.md"}
    md_files = [
        f for f in WIKI_DIR.rglob("*.md")
        if f.name not in skip_names and f.stem != ".keep"
    ]

    all_pairs = []

    for filepath in sorted(md_files):
        try:
            content = filepath.read_text(encoding="utf-8", errors="replace")
            prompt = QA_PROMPT.format(n=pairs_per, article=content[:4000])
            raw = llm.invoke(prompt)

            raw = clean_json_string(raw)

            pairs = json.loads(raw)
            validated = []
            for pair in pairs:
                if validate_pair(pair, content, llm):
                    validated.append(pair)

            print(f"{filepath.name}: {len(validated)}/{len(pairs)} pairs validated")
            all_pairs.extend(validated)

        except Exception as e:
            print(f"Skipped {filepath.name}: {e}")
            continue

    random.shuffle(all_pairs)

    total = len(all_pairs)
    n_train = int(total * 0.8)
    n_val = int(total * 0.1)

    train = all_pairs[:n_train]
    val = all_pairs[n_train:n_train + n_val]
    test = all_pairs[n_train + n_val:]

    FINETUNE_DIR.mkdir(exist_ok=True)

    (FINETUNE_DIR / "train.json").write_text(json.dumps(train, indent=2), encoding="utf-8")
    (FINETUNE_DIR / "val.json").write_text(json.dumps(val, indent=2), encoding="utf-8")
    (FINETUNE_DIR / "test.json").write_text(json.dumps(test, indent=2), encoding="utf-8")

    print(f"Splits: {len(train)} train | {len(val)} val | {len(test)} test")
    print("test.json is LOCKED. Do not retrain on it.")


if __name__ == "__main__":
    generate_and_split()
