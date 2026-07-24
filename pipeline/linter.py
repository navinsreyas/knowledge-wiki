"""Health check for vault/wiki/: broken backlinks and contradiction detection."""

import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import requests
from langchain_ollama import ChatOllama

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

WIKI_DIR = Path("vault/wiki")
LOG_PATH = Path("vault/log.md")


def find_broken_backlinks() -> list:
    all_titles = {f.stem for f in WIKI_DIR.rglob("*.md")}
    broken = []
    for md_file in WIKI_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        for link in re.findall(r"\[\[([^\]]+)\]\]", content):
            slug = link.lower().strip().replace(" ", "-")
            if slug not in all_titles:
                broken.append({"article": md_file.name, "broken_link": link})
    return broken


def get_related_pairs() -> list:
    link_to_articles: dict = defaultdict(list)
    for md_file in WIKI_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        for link in re.findall(r"\[\[([^\]]+)\]\]", content):
            link_to_articles[link].append(md_file.name)

    pairs = []
    for link, articles in link_to_articles.items():
        if len(articles) >= 2:
            pairs.append((articles[0], articles[1], link))
    return pairs[:8]


def check_contradiction(art1: str, art2: str, llm) -> str:
    def read_article(name: str) -> str:
        matches = list(WIKI_DIR.rglob(name))
        if not matches:
            return f"[file not found: {name}]"
        return matches[0].read_text(encoding="utf-8")[:1500]

    prompt = (
        f"Compare these two wiki articles. Do they contradict each other?\n"
        f"Reply with either:\n"
        f"CONTRADICTION: brief description of the contradiction\n"
        f"CONSISTENT\n\n"
        f"Article 1 ({art1}):\n{read_article(art1)}\n\n"
        f"Article 2 ({art2}):\n{read_article(art2)}"
    )
    return llm.invoke(prompt).content.strip()


def compile_report(broken: list, contradictions: list) -> str:
    today = date.today().isoformat()
    lines = [f"# Wiki Health Report — {today}", ""]

    lines.append(f"## Broken Backlinks ({len(broken)} found)")
    lines.append("")
    if broken:
        for item in broken:
            lines.append(f"- `{item['article']}` links to `[[{item['broken_link']}]]` — no matching file")
    else:
        lines.append("All backlinks resolve correctly.")
    lines.append("")

    lines.append(f"## Contradiction Check ({len(contradictions)} pairs checked)")
    lines.append("")
    if contradictions:
        for art1, art2, link, result in contradictions:
            lines.append(f"- `{art1}` vs `{art2}` (shared concept: `{link}`): {result}")
    else:
        lines.append("No article pairs to check.")
    lines.append("")

    return "\n".join(lines)


def _check_services() -> bool:
    ok = True
    try:
        requests.get("http://127.0.0.1:11434", timeout=3)
    except Exception:
        print("Error: Ollama is not running.\nStart it with:  ollama serve")
        ok = False
    try:
        requests.get(f"{config.VAULT_API}/vault/", headers=config.HEADERS,
                     verify=config.VERIFY_SSL, timeout=3)
    except Exception:
        print(
            f"Error: Obsidian Local REST API is not reachable at {config.VAULT_API}.\n"
            "Make sure Obsidian is open and the Local REST API plugin is enabled."
        )
        ok = False
    return ok


def run_lint():
    if not _check_services():
        return

    llm = ChatOllama(model=config.MODEL_NAME)
    today = date.today().isoformat()

    print("Scanning backlinks...")
    broken = find_broken_backlinks()

    print("Finding related article pairs...")
    pairs = get_related_pairs()

    print(f"Checking {len(pairs)} article pairs for contradictions...")
    contradictions = []
    for art1, art2, link in pairs:
        contradictions.append((art1, art2, link, check_contradiction(art1, art2, llm)))

    report = compile_report(broken, contradictions)

    requests.put(
        f"{config.VAULT_API}/vault/health/lint-report.md",
        headers={**config.HEADERS, "Content-Type": "text/markdown"},
        data=report.encode("utf-8"),
        verify=config.VERIFY_SSL,
    )

    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"\n## [{today}] lint | {len(broken)} broken links, {len(pairs)} pairs checked\n")

    print(f"Lint complete: {len(broken)} broken links, {len(pairs)} pairs checked")
    print("Report saved to vault/health/lint-report.md")


if __name__ == "__main__":
    run_lint()
