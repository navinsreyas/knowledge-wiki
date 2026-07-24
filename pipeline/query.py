"""Answer questions against the wiki using Qwen2.5:7B as a tool-calling agent."""

import sys
from datetime import date
from pathlib import Path

import requests
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


@tool
def read_index() -> str:
    """Read master wiki index — always start here to see what exists"""
    resp = requests.get(
        f"{config.VAULT_API}/vault/wiki/_index.md",
        headers=config.HEADERS,
        verify=config.VERIFY_SSL,
    )
    return resp.text


@tool
def read_article(path: str) -> str:
    """Read a specific wiki article by its vault path"""
    # Strip any leading 'vault/' to prevent a /vault/vault/ double-prefix URL.
    if path.startswith("vault/"):
        path = path[len("vault/"):]
    resp = requests.get(
        f"{config.VAULT_API}/vault/{path}",
        headers=config.HEADERS,
        verify=config.VERIFY_SSL,
    )
    return resp.text


@tool
def search_wiki(query: str) -> str:
    """Search across all wiki articles by reading index and matching content."""
    import re
    from pathlib import Path

    _STOPWORDS = {"what", "this", "that", "with", "from", "have", "been",
                  "does", "your", "their", "which", "when", "where", "would"}
    query_lower = query.lower()
    # Strip punctuation and filter: min 4 chars, not a stopword
    import re as _re
    query_terms = [
        t for t in _re.sub(r"[^\w\s]", "", query_lower).split()
        if len(t) >= 4 and t not in _STOPWORDS
    ]

    # Read the index — try Obsidian API first, fall back to local filesystem.
    index_text: str | None = None
    try:
        index_resp = requests.get(
            f"{config.VAULT_API}/vault/wiki/_index.md",
            headers=config.HEADERS, verify=config.VERIFY_SSL, timeout=5
        )
        if index_resp.status_code == 200:
            index_text = index_resp.text
    except Exception:
        pass

    if index_text is None:
        local_index = Path("vault/wiki/_index.md")
        if local_index.exists():
            index_text = local_index.read_text(encoding="utf-8")

    if not index_text:
        return "Index not found."

    # _index.md is topic-structured: "- [[slug]] — description"
    # There is no (category) field any more — slugs are matched by term.
    slugs: list[str] = []
    for line in index_text.splitlines():
        m = re.search(r'\[\[([^\]]+)\]\]', line)
        if m:
            slugs.append(m.group(1))

    matched_slugs = [s for s in slugs if any(t in s.lower() for t in query_terms)]

    # Fallback: score slugs by description text when term match is sparse
    if len(matched_slugs) < 2:
        for line in index_text.splitlines():
            m = re.search(r'\[\[([^\]]+)\]\]', line)
            if not m:
                continue
            slug = m.group(1)
            if slug in matched_slugs:
                continue
            desc = line[m.end():].lower()
            if any(t in desc for t in query_terms):
                matched_slugs.append(slug)
                if len(matched_slugs) >= 5:
                    break

    if not matched_slugs:
        return f"No articles found matching: {query}"

    # Resolve each slug to its file by globbing all topic subfolders (*/)
    # (Obsidian API: GET /vault/wiki/ returns a JSON listing of all paths)
    vault_root = Path("vault/wiki")
    results = []
    seen: set[str] = set()

    for slug in matched_slugs[:5]:
        if slug in seen:
            continue
        seen.add(slug)

        # Try the Obsidian API directory listing first
        resp_ok = False
        try:
            week_list_resp = requests.get(
                f"{config.VAULT_API}/vault/wiki/",
                headers=config.HEADERS, verify=config.VERIFY_SSL, timeout=5
            )
            if week_list_resp.status_code == 200:
                files = week_list_resp.json().get('files', [])
                for fpath in files:
                    if fpath.endswith(f"/{slug}.md"):
                        art_resp = requests.get(
                            f"{config.VAULT_API}/vault/{fpath}",
                            headers=config.HEADERS, verify=config.VERIFY_SSL, timeout=5
                        )
                        if art_resp.status_code == 200:
                            results.append(f"=== {slug} ===\n{art_resp.text[:2000]}")
                            resp_ok = True
                            break
        except Exception:
            pass

        # Local filesystem fallback (works when running from project root)
        if not resp_ok:
            for candidate in vault_root.glob(f"*/{slug}.md"):
                try:
                    results.append(f"=== {slug} ===\n{candidate.read_text(encoding='utf-8')[:2000]}")
                    resp_ok = True
                    break
                except Exception:
                    pass

        if len(results) >= 3:
            break

    return "\n\n".join(results) if results else f"Articles found in index but could not be read: {query}"


@tool
def save_output(filename: str, content: str) -> str:
    """Save a useful answer back into wiki outputs/ so it compounds"""
    requests.put(
        f"{config.VAULT_API}/vault/outputs/{filename}.md",
        headers={**config.HEADERS, "Content-Type": "text/markdown"},
        data=content.encode("utf-8"),
        verify=config.VERIFY_SSL,
    )
    return f"Saved: outputs/{filename}.md"


_TOOL_MAP = {
    "read_index": read_index,
    "read_article": read_article,
    "search_wiki": search_wiki,
    "save_output": save_output,
}


def append_log(question: str):
    with open("vault/log.md", "a", encoding="utf-8") as f:
        f.write(f"\n## [{date.today()}] query | {question[:80]}\n")


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


def run_query(question: str) -> str:
    if not _check_services():
        return ""

    llm = ChatOllama(model=config.MODEL_NAME)
    agent = llm.bind_tools([read_index, read_article, search_wiki, save_output])

    # Tool-call path — execute any tools Qwen requested, then synthesise
    context = ""
    try:
        response = agent.invoke(question)
        if hasattr(response, "tool_calls") and response.tool_calls:
            parts = []
            for tc in response.tool_calls:
                fn = _TOOL_MAP.get(tc["name"])
                if fn:
                    parts.append(str(fn.invoke(tc["args"])))
            context = "\n".join(parts).strip()
    except Exception:
        pass

    if not context:
        try:
            context = str(search_wiki.invoke(question))[:4000]
        except Exception as exc:
            return f"Search failed: {exc}"

    answer = llm.invoke(
        f"Answer this question using the wiki articles below.\n\n"
        f"Wiki content:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Be specific. Reference the articles by name. "
        f"If the wiki content does not contain the answer, say so clearly."
    ).content
    append_log(question)
    return answer


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is in this wiki?"
    print(run_query(question))
