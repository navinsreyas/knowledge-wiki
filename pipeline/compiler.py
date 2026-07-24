"""Compile raw vault articles into structured wiki pages via Qwen."""

import json
import re
import socket
import sys
from datetime import date
from pathlib import Path

import requests
from langchain_ollama import OllamaLLM

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

GRAPH_REPORT = Path("graphify-out/GRAPH_REPORT.md")
QUEUE_FILE = Path("pipeline/.ingest_queue.json")
LOG_PATH = Path("vault/log.md")

CATEGORY_TO_FOLDER = {
    "linear-algebra":      "maths-statistics-ab-testing",
    "probability":         "maths-statistics-ab-testing",
    "statistics":          "maths-statistics-ab-testing",
    "evaluation":          "maths-statistics-ab-testing",
    "neural":              "deep-learning-transformers",
    "attention":           "deep-learning-transformers",
    "transformer":         "deep-learning-transformers",
    "fine-tuning":         "fine-tuning-alignment-quantisation",
    "finetuning":          "fine-tuning-alignment-quantisation",
    "peft":                "fine-tuning-alignment-quantisation",
    "rlhf":                "fine-tuning-alignment-quantisation",
    "alignment":           "fine-tuning-alignment-quantisation",
    "compression":         "fine-tuning-alignment-quantisation",
    "distributed":         "fine-tuning-alignment-quantisation",
    "rag":                 "rag-vector-retrieval",
    "retrieval":           "rag-vector-retrieval",
    "vector":              "rag-vector-retrieval",
    "graph":               "rag-vector-retrieval",
    "advanced-rag":        "rag-vector-retrieval",
    "agent":               "agents-orchestration",
    "agents":              "agents-orchestration",
    "reasoning":           "agents-orchestration",
    "ragas":               "evaluation-quality",
    "benchmark":           "evaluation-quality",
    "security":            "ai-security-red-teaming",
    "serving":             "llm-serving-latency",
    "inference":           "llm-serving-latency",
    "cost":                "cost-engineering-caching",
    "monitoring":          "monitoring-observability",
    "fastapi":             "apis-fastapi-backend",
    "docker":              "docker-deployment-secrets",
    "langgraph":           "langgraph-hitl-reliability",
    "prompt":              "prompting-structured-outputs",
    "system-design":       "ai-system-design",
    "cloud":               "cloud-iam-infrastructure",
    "data-engineering":    "data-engineering-feature-stores",
    "feature-store":       "data-engineering-feature-stores",
    "experimentation":     "online-experimentation-bandits",
    "kubernetes":          "kubernetes-cicd-gitops",
    "multimodal":          "multimodal-ai",
    "python":              "advanced-python",
    "llm":                 "deep-learning-transformers",
    "paper":               "deep-learning-transformers",
    "arxiv":               "deep-learning-transformers",
}

_VALID_CATEGORIES = list(CATEGORY_TO_FOLDER.keys())
_CATEGORY_PROMPT_LIST = ", ".join(_VALID_CATEGORIES)

llm = OllamaLLM(model=config.MODEL_NAME)

COMPILE_PROMPT = """
You are a wiki compiler for an AI Engineering researcher.

Knowledge graph context — use ONLY these concepts for [[backlinks]]:
{graph_context}

Source material:
{content}

Write a structured wiki article (600+ words) including:
- Core concept explanation with examples
- Key insights from the source
- [[backlinks]] — ONLY link concepts that appear in the graph context above
- Code snippets where relevant
- Sources section at the bottom

Be factual. No hallucination. Write for an AI Engineer who wants depth.
The source may be from a PDF, YouTube transcript, tweet, or web article.
Extract and synthesise the knowledge regardless of original format.
"""


def slugify(text: str) -> str:
    text = text.lower().strip()[:60]
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


def _api_get(vault_path: str) -> requests.Response:
    return requests.get(
        f"{config.VAULT_API}/vault/{vault_path}",
        headers=config.HEADERS,
        verify=config.VERIFY_SSL,
    )


def _api_put(vault_path: str, content: str) -> requests.Response:
    return requests.put(
        f"{config.VAULT_API}/vault/{vault_path}",
        headers={**config.HEADERS, "Content-Type": "text/markdown"},
        data=content.encode("utf-8"),
        verify=config.VERIFY_SSL,
    )


def get_source_type(content: str) -> str:
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("source:"):
            return line.split(":", 1)[1].strip()
        if line == "---" and content.index(line) > 0:
            break
    return "web"


def append_log(action: str, title: str, source_type: str = ""):
    tag = f" [{source_type}]" if source_type else ""
    entry = f"\n## [{date.today()}] {action}{tag} | {title}\n"
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(entry)


def update_index(title: str, folder: str):
    entry = f"- [[{title}]] ({folder})\n"
    resp = _api_get("wiki/_index.md")
    existing = resp.text if resp.status_code == 200 else ""
    if entry.strip() in existing:
        return
    _api_put("wiki/_index.md", existing + entry)


def _resolve_folder(raw_category: str) -> str:
    """Map LLM category key to a vault/wiki/ subfolder path."""
    key = raw_category.strip().lower().replace(" ", "-")
    if key in CATEGORY_TO_FOLDER:
        return CATEGORY_TO_FOLDER[key]
    # Partial match
    for k, v in CATEGORY_TO_FOLDER.items():
        if k in key or key in k:
            return v
    return "models/llms"


def compile_article(raw_path: str):
    path = Path(raw_path)
    if not path.exists():
        print(f"Missing file, skipping: {raw_path}")
        return

    content = path.read_text(encoding="utf-8")
    source_type = get_source_type(content)

    title = path.stem
    for line in content.splitlines():
        if line.startswith("# "):
            title = line.lstrip("# ").strip()
            break

    slug = slugify(title)

    classify_prompt = (
        f"Classify this AI Engineering article into exactly one category key. "
        f"Reply with a single key only — one of: {_CATEGORY_PROMPT_LIST}.\n\n"
        f"{content[:300]}"
    )
    raw_category = llm.invoke(classify_prompt).strip().lower()
    folder = _resolve_folder(raw_category)

    wiki_path = f"wiki/{folder}/{slug}.md"

    if _api_get(wiki_path).status_code == 200:
        print(f"Already compiled: {wiki_path}")
        return

    graph_context = (
        GRAPH_REPORT.read_text(encoding="utf-8")[:2000]
        if GRAPH_REPORT.exists()
        else "No graph available yet."
    )

    article = llm.invoke(COMPILE_PROMPT.format(
        graph_context=graph_context,
        content=content[:4000],
    ))

    _api_put(wiki_path, article)
    update_index(slug, folder)
    append_log("ingest", title, source_type)
    print(f"Compiled [{source_type}]: vault/{wiki_path}")


def _check_obsidian_running() -> bool:
    try:
        sock = socket.create_connection(("127.0.0.1", 27124), timeout=2)
        sock.close()
        return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def _check_ollama() -> bool:
    try:
        resp = requests.get("http://127.0.0.1:11434", timeout=3)
        return resp.status_code < 500
    except Exception:
        print(
            "Error: Ollama is not running.\n"
            "Start it with:  ollama serve\n"
            "Then verify:    ollama list"
        )
        return False


def compile_batch():
    if not _check_obsidian_running():
        print("ERROR: Obsidian is not running or Local REST API plugin is not active.")
        print("Fix: Open Obsidian and wait 5 seconds, then run wiki compile again.")
        return

    if not _check_ollama():
        return

    try:
        from pipeline.graph_builder import run_graph_builder
        incremental = Path("graphify-out/graph.json").exists()
        run_graph_builder(incremental=incremental)
    except Exception as e:
        print(f"Graph builder failed: {e} — compiling without graph context.")

    if not QUEUE_FILE.exists():
        print(f"Queue file not found: {QUEUE_FILE}\nRun pipeline/ingest.py first.")
        return

    queue: dict = json.loads(QUEUE_FILE.read_text(encoding="utf-8"))
    for h, path in queue.items():
        if Path(path).exists():
            compile_article(path)


if __name__ == "__main__":
    compile_batch()
