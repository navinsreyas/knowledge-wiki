"""Build a knowledge graph from vault/raw/ and write graphify-out/GRAPH_REPORT.md."""

import json
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False
    print("Install networkx: pip install networkx")

RAW_DIR     = Path("vault/raw")
OUTPUT_DIR  = Path("graphify-out")
REPORT_PATH = OUTPUT_DIR / "GRAPH_REPORT.md"
GRAPH_PATH  = OUTPUT_DIR / "graph.json"

STOPWORDS = {
    # Common English
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "and", "or", "but",
    "if", "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "up", "about", "into", "through", "during", "this", "that", "these",
    "those", "it", "its", "we", "they", "he", "she", "you", "i", "me",
    "my", "our", "their", "what", "which", "who", "how", "when", "where",
    "why", "not", "no", "nor", "so", "yet", "both", "either", "neither",
    "each", "few", "more", "most", "other", "some", "such", "than", "too",
    "very", "just", "also", "as", "than", "then", "thus", "however",
    "therefore",
    # Frontmatter / web scraping artifacts
    "title", "source", "ingested", "http", "https", "html", "document",
    "documents", "page", "pages", "website", "content", "section",
    "author", "authors", "complete", "available", "provide", "given",
    "learn", "used", "using", "uses", "make", "made", "well", "good",
    "need", "know", "work", "works", "working", "like", "look", "first",
    "many", "even", "here", "there", "where", "their", "them", "over",
    "only", "after", "before", "between", "without", "number", "time",
    "different", "same", "example", "examples", "include", "including",
    # Domain-generic (too common to be meaningful concepts)
    "learning", "machine", "linear", "deep", "training", "science",
    "images", "engineering", "computer", "dive", "model", "models",
    "data", "system", "systems", "based", "figure", "table", "chapter",
    "link", "links", "note", "notes", "click", "download", "install",
    "github", "arxiv", "paper", "papers", "course", "courses", "lecture",
    "lectures", "book", "books", "guide", "guides", "introduction",
    "overview", "tutorial", "tutorials", "documentation", "version",
    "updated", "created", "written", "sources", "original", "date",
    "year", "month", "three", "four", "five", "next", "last",
    # HTML / web scraping noise
    "searchtype", "submitted", "reading", "read", "reads", "test", "tests",
    "submit", "button", "bottom", "arrow", "always", "called", "comes",
    "create", "called", "classes", "class", "style", "color", "image",
    "search", "form", "input", "type", "value", "label", "field",
    # Table of contents / page structure scraped text
    "contents", "revised", "estimated", "study", "lilian", "weng",
    "table", "chapter", "appendix", "abstract", "references", "related",
    "suggested", "further", "back", "return", "previous", "section",
    "revised", "published", "posted", "view", "views", "visit", "copy",
    "share", "save", "print", "close", "open", "show", "hide", "toggle",
    # Too generic to be meaningful nodes
    "intuitive", "zero", "building", "getting", "take", "help", "helps",
    "want", "think", "understand", "understanding", "simple", "simply",
    "high", "long", "short", "large", "small", "real", "true", "false",
    "right", "left", "mean", "means", "meaning", "point", "points",
    "case", "cases", "part", "parts", "kind", "place", "thing", "things",
    "idea", "ideas", "level", "levels", "term", "terms", "word", "words",
    "name", "names", "list", "lines", "line", "item", "items", "step",
    "steps", "reason", "reasons", "result", "results", "output", "outputs",
    "start", "end", "add", "adds", "adding", "change", "changes",
    "language", "languages", "python", "code", "query", "queries",
    # Meta / date / blog navigation noise
    "concept", "concepts", "hero", "post", "posts", "blog", "article",
    "articles", "january", "february", "march", "april", "june", "july",
    "august", "september", "october", "november", "december", "monday",
    "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "week", "weeks", "hour", "hours", "minute", "minutes", "today",
}

# Technical AI/ML terms that get 3× weight when scoring concepts
PRIORITY_TERMS = {
    "transformer", "attention", "embedding", "gradient", "backpropagation",
    "regularization", "optimization", "inference", "finetuning", "retrieval",
    "augmented", "generation", "reinforcement", "classification", "regression",
    "clustering", "dimensionality", "tokenization", "quantization", "distillation",
    "pruning", "multimodal", "alignment", "hallucination", "evaluation", "benchmark",
    "deployment", "monitoring", "observability", "qlora", "lora", "peft", "rag",
    "llm", "gpt", "bert", "langchain", "langgraph", "langsmith", "autogen",
    "vectordb", "softmax", "relu", "dropout", "batchnorm", "layernorm", "encoder",
    "decoder", "crossentropy", "perplexity", "bleu", "rouge", "ndcg", "precision",
    "recall", "accuracy", "f1score", "overfitting", "underfitting", "bias",
    "variance", "eigenvalue", "eigenvector", "svd", "pca", "bayesian",
    "frequentist", "causality", "probability", "distribution", "entropy",
    "divergence", "loss", "reward", "policy", "agent", "multiagent",
    "orchestration", "tool", "memory", "context", "prompt", "chain",
}


def _strip_frontmatter(text: str) -> str:
    """Remove YAML frontmatter (--- ... ---) from the start of a markdown file."""
    if not text.startswith("---"):
        return text
    end = text.find("---", 3)
    return text[end + 3:].lstrip() if end != -1 else text


def _is_artifact(word: str) -> bool:
    """Return True for URL fragments, scraping noise, and other non-concept strings."""
    if len(word) > 25:
        return True
    if word.isupper():                        # navigation/acronym noise
        return True
    if re.search(r"\d", word):                # digits mixed into letters
        return True
    return False


def extract_concepts(text: str, title: str) -> list[str]:
    body = _strip_frontmatter(text)

    # Sample title + first 200 chars of each of the first 5 non-empty paragraphs
    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
    sampled = title + " " + " ".join(p[:200] for p in paragraphs[:5])

    words = re.findall(r"[a-z]+", sampled.lower())
    words = [w for w in words if len(w) >= 4 and w not in STOPWORDS and not _is_artifact(w)]

    counts = Counter(words)

    # Apply 3× boost to priority technical terms
    boosted: Counter = Counter()
    for word, count in counts.items():
        boosted[word] = count * 3 if word in PRIORITY_TERMS else count

    return [word for word, _ in boosted.most_common(15)]


def build_graph(articles: list[dict]):
    if not HAS_NX:
        return {"nodes": [], "edges": []}

    G = nx.Graph()

    concept_articles = defaultdict(list)
    concept_frequency = Counter()
    for article in articles:
        for concept in article["concepts"]:
            concept_articles[concept].append(article["title"])
            concept_frequency[concept] += 1

    for concept, freq in concept_frequency.items():
        G.add_node(concept, frequency=freq, articles=concept_articles[concept])

    edge_weights = Counter()
    for article in articles:
        concepts = article["concepts"]
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                pair = tuple(sorted([concepts[i], concepts[j]]))
                edge_weights[pair] += 1

    for (u, v), weight in edge_weights.items():
        G.add_edge(u, v, weight=weight)

    return G


def find_god_nodes(G, top_n: int = 10) -> list[tuple]:
    if not HAS_NX or len(G.nodes()) == 0:
        return []

    results = [
        (node, G.degree(node), G.nodes[node].get("frequency", 0))
        for node in G.nodes()
    ]
    return sorted(results, key=lambda x: x[1], reverse=True)[:top_n]


def find_communities(G) -> list[list]:
    if not HAS_NX or len(G.nodes()) < 3:
        return []

    try:
        raw = list(nx.community.greedy_modularity_communities(G))
        communities = [sorted(list(c)) for c in raw]

        # Merge smallest communities until we are at or below 15
        while len(communities) > 15:
            communities.sort(key=len)
            communities = [communities[0] + communities[1]] + communities[2:]

        communities.sort(key=len, reverse=True)
        return communities
    except Exception:
        return []


def find_surprising_connections(G, articles: list[dict]) -> list[str]:
    if not HAS_NX or len(G.edges()) == 0:
        return []

    connections = []
    for u, v, data in sorted(G.edges(data=True), key=lambda e: e[2].get("weight", 1), reverse=True):
        weight = data.get("weight", 1)
        u_freq = G.nodes[u].get("frequency", 1)
        v_freq = G.nodes[v].get("frequency", 1)
        # Surprising: co-occur often but each also appears independently
        if u_freq > weight and v_freq > weight:
            connections.append(f"{u} ↔ {v}: appear together in {weight} articles")
        if len(connections) >= 5:
            break
    return connections


def build_report(articles, god_nodes, communities, surprising, G=None) -> str:
    today = date.today().isoformat()
    node_count = len(G.nodes()) if HAS_NX and G is not None else 0
    edge_count = len(G.edges()) if HAS_NX and G is not None else 0

    lines = [
        "# Knowledge Graph Report",
        f"Generated: {today}",
        f"Articles indexed: {len(articles)}",
        f"Total concepts: {node_count}",
        f"Total connections: {edge_count}",
        "",
        "## God Nodes (Most Connected Concepts)",
        "These concepts connect the most other concepts in your wiki.",
        "Use these as [[backlinks]] — they are the core vocabulary of your knowledge base.",
        "",
    ]

    for i, (concept, degree, article_count) in enumerate(god_nodes, 1):
        lines.append(f"{i}. [[{concept}]] — appears in {article_count} articles, connects to {degree} concepts")

    lines += [
        "",
        "## Concept Communities",
        "Concepts that frequently appear together form communities.",
        "Articles in the same community should link to each other.",
        "",
    ]

    if communities:
        for i, community in enumerate(communities, 1):
            name = community[0] if community else f"cluster-{i}"
            lines.append(f"### Community {i}: {name}")
            lines.append(f"Concepts: {', '.join(community[:20])}")
            lines.append("")
    else:
        lines += ["No communities detected.", ""]

    lines += [
        "## Surprising Connections",
        "Concepts from different topic areas that appear together frequently.",
        "These unexpected links are worth exploring as cross-domain insights.",
        "",
    ]

    if surprising:
        for s in surprising:
            lines.append(f"- {s}")
    else:
        lines.append("Not enough data for connection analysis.")
    lines.append("")

    lines += [
        "## All Concepts (for backlink reference)",
        "Full list of concepts extracted from your vault.",
        "When compiling wiki articles, prefer linking to concepts on this list.",
        "",
    ]

    all_concepts = sorted(G.nodes()) if HAS_NX and G is not None else []
    lines.append(", ".join(all_concepts))
    lines.append("")

    return "\n".join(lines)


def run_graph_builder(incremental: bool = False):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    md_files = [f for f in RAW_DIR.rglob("*.md") if ".keep" not in f.name]

    existing_graph_data = None
    if incremental and GRAPH_PATH.exists():
        cutoff = GRAPH_PATH.stat().st_mtime
        new_files = [f for f in md_files if f.stat().st_mtime > cutoff]
        if not new_files:
            print("Graph is up to date — no new articles to process.")
            return
        with open(GRAPH_PATH, encoding="utf-8") as f:
            existing_graph_data = json.load(f)
        md_files = new_files
        print(f"Incremental mode: processing {len(md_files)} new articles...")

    articles = []
    for filepath in sorted(md_files):
        content = filepath.read_text(encoding="utf-8", errors="replace")
        title = filepath.stem.replace("-", " ").replace("_", " ")
        for line in content.splitlines():
            if line.startswith("# "):
                title = line.lstrip("# ").strip()
                break
        concepts = extract_concepts(content, title)
        if concepts:
            articles.append({"title": title, "path": str(filepath), "concepts": concepts})

    print(f"Building graph from {len(articles)} articles...")

    G = build_graph(articles)

    # In incremental mode, merge new nodes/edges into the existing graph
    if incremental and existing_graph_data and HAS_NX:
        for node in existing_graph_data.get("nodes", []):
            nid = node["id"]
            if nid not in G.nodes():
                G.add_node(nid, frequency=node["frequency"], articles=node["articles"])
            else:
                G.nodes[nid]["frequency"] += node["frequency"]
                G.nodes[nid]["articles"] = list(set(G.nodes[nid]["articles"] + node["articles"]))
        for edge in existing_graph_data.get("edges", []):
            u, v, w = edge["source"], edge["target"], edge["weight"]
            if G.has_edge(u, v):
                G[u][v]["weight"] += w
            else:
                G.add_edge(u, v, weight=w)

    god_nodes   = find_god_nodes(G)
    communities = find_communities(G)
    surprising  = find_surprising_connections(G, articles)

    report = build_report(articles, god_nodes, communities, surprising, G)
    REPORT_PATH.write_text(report, encoding="utf-8")

    if HAS_NX:
        graph_data = {
            "nodes": [
                {"id": n, "frequency": G.nodes[n].get("frequency", 0),
                 "articles": G.nodes[n].get("articles", [])}
                for n in G.nodes()
            ],
            "edges": [
                {"source": u, "target": v, "weight": G[u][v].get("weight", 1)}
                for u, v in G.edges()
            ],
        }
    else:
        graph_data = G

    GRAPH_PATH.write_text(json.dumps(graph_data, indent=2), encoding="utf-8")

    print(f"Graph built: {len(god_nodes)} god nodes, {len(communities)} communities")
    print(f"Report saved to {REPORT_PATH}")


if __name__ == "__main__":
    incremental = "--update" in sys.argv
    run_graph_builder(incremental=incremental)
