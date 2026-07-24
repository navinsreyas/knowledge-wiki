"""wiki CLI — ingest, compile, query, lint."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def show_status():
    wiki_dir = Path("vault/wiki")
    raw_dir = Path("vault/raw")
    queue_file = Path("pipeline/.ingest_queue.json")

    wiki_count = sum(
        1 for f in wiki_dir.rglob("*.md")
        if f.name not in ("_index.md", "log.md")
    ) if wiki_dir.exists() else 0

    raw_count = sum(
        1 for f in raw_dir.rglob("*.md")
        if ".keep" not in f.name
    ) if raw_dir.exists() else 0

    queue_count = 0
    if queue_file.exists():
        queue_count = len(json.loads(queue_file.read_text(encoding="utf-8")))

    print(f"Wiki articles : {wiki_count}")
    print(f"Raw sources   : {raw_count}")
    print(f"Queue entries : {queue_count}")
    print(f"Phase 2 gate  : {wiki_count}/50 articles")


def cli():
    parser = argparse.ArgumentParser(prog="wiki", description="AI Engineering wiki pipeline")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("status", help="Show wiki and queue counts")

    p_ingest = sub.add_parser("ingest", help="Add a source to vault/raw/articles/")
    p_ingest.add_argument("source", help="File path or URL to ingest")

    p_compile = sub.add_parser("compile", help="Compile raw articles into wiki pages")
    p_compile.add_argument("path", nargs="?", default=None,
                           help="Specific raw file to compile (omit to compile all queued)")

    p_query = sub.add_parser("query", help="Ask a question against the wiki")
    p_query.add_argument("question", nargs="+", help="Question words")

    sub.add_parser("lint", help="Check backlinks and contradictions")

    args = parser.parse_args()

    if args.command == "status":
        show_status()

    elif args.command == "ingest":
        from pipeline.ingest_sources import ingest
        ok = ingest(args.source)
        if ok:
            print("Run 'wiki compile' to compile into the wiki.")

    elif args.command == "compile":
        if args.path:
            from pipeline.compiler import compile_article
            compile_article(args.path)
        else:
            from pipeline.ingest import enqueue_new
            from pipeline.compiler import compile_batch
            enqueue_new()
            compile_batch()

    elif args.command == "query":
        from pipeline.query import run_query
        print(run_query(" ".join(args.question)))

    elif args.command == "lint":
        from pipeline.linter import run_lint
        run_lint()

    else:
        parser.print_help()


if __name__ == "__main__":
    cli()
