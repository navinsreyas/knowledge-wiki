"""Convert any source (PDF, YouTube, Twitter, web URL) into a vault/raw/articles/ markdown file."""

import re
import sys
from datetime import date
from pathlib import Path

import pymupdf4llm
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

VAULT_ROOT = Path(__file__).parent.parent / "vault"
OUT_DIR = VAULT_ROOT / "raw" / "articles"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TODAY = date.today().isoformat()

BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


def slugify(text: str) -> str:
    text = text.lower().strip()[:60]
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


def _frontmatter(source_type: str, title: str, url: str = "") -> str:
    url_line = f"url: {url}" if url else "url:"
    return f"---\nsource: {source_type}\ntitle: {title}\ningested: {TODAY}\n{url_line}\n---\n\n"


def _write(slug: str, content: str) -> Path:
    dest = OUT_DIR / f"{slug}.md"
    dest.write_text(content, encoding="utf-8")
    return dest


def ingest_pdf(file_path: str) -> bool:
    try:
        path = Path(file_path)
        raw_title = path.stem.replace("-", " ").replace("_", " ").title()
        slug = slugify(raw_title)
        md = pymupdf4llm.to_markdown(str(path))
        _write(slug, _frontmatter("pdf", raw_title) + md)
        print(f"Ingested PDF: {path.name}")
        return True
    except Exception as exc:
        print(f"Error ingesting PDF '{file_path}': {exc}")
        return False


def _extract_video_id(url: str) -> str | None:
    m = re.search(r"youtu\.be/([A-Za-z0-9_-]{11})", url)
    if m:
        return m.group(1)
    m = re.search(r"[?&]v=([A-Za-z0-9_-]{11})", url)
    if m:
        return m.group(1)
    return None


def ingest_youtube(url: str) -> bool:
    try:
        video_id = _extract_video_id(url)
        if not video_id:
            print(f"Error: could not extract video ID from '{url}'")
            return False

        # Fetch title via oEmbed (no API key needed)
        resp = requests.get(f"https://www.youtube.com/oembed?url={url}&format=json", timeout=10)
        resp.raise_for_status()
        title = resp.json().get("title", f"youtube-{video_id}")

        try:
            ytt_api = YouTubeTranscriptApi()
            transcript = ytt_api.fetch(video_id)
            segments = [{"text": t.text, "start": t.start} for t in transcript]
        except TranscriptsDisabled:
            print(
                f"Warning: transcripts are disabled for '{url}'.\n"
                f"To ingest manually:\n"
                f"  1. Open the video and copy the transcript from the '...' menu\n"
                f"  2. Save it as: vault/raw/articles/{slugify(title)}.md"
            )
            return False

        # Group every 10 segments into a paragraph
        paragraphs = []
        for i in range(0, len(segments), 10):
            chunk = segments[i : i + 10]
            paragraphs.append(" ".join(s["text"] for s in chunk))

        slug = slugify(title)
        _write(slug, _frontmatter("youtube", title, url) + "\n\n".join(paragraphs))
        print(f"Ingested YouTube: {title}")
        return True
    except Exception as exc:
        print(f"Error ingesting YouTube '{url}': {exc}")
        return False


def ingest_twitter(url: str) -> bool:
    m = re.search(r"/status/(\d+)", url)
    if not m:
        print(f"Error: could not extract tweet ID from '{url}'")
        return False

    tweet_id = m.group(1)
    slug = f"tweet-{tweet_id}"
    headers = {"User-Agent": BROWSER_UA}
    body = None

    # Attempt 1 — Thread Reader App
    try:
        thr_url = f"https://threadreaderapp.com/thread/{tweet_id}.html"
        resp = requests.get(thr_url, headers=headers, timeout=15)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            container = soup.find("div", class_="content") or soup.find("article")
            if container:
                body = markdownify(str(container), heading_style="ATX").strip()
    except Exception:
        pass

    # Attempt 2 — fetch X directly
    if not body:
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                article = soup.find("article")
                if article:
                    body = article.get_text(separator="\n").strip()
        except Exception:
            pass

    if not body:
        print(
            f"Could not fetch tweet {tweet_id} automatically.\n"
            f"To ingest manually:\n"
            f"  1. Open the thread and copy the full text\n"
            f"  2. Save it as: vault/raw/articles/tweet-{tweet_id}.md\n"
            f"  3. Add frontmatter: source: twitter, title: <title>, "
            f"ingested: {TODAY}, url: {url}"
        )
        return False

    _write(slug, _frontmatter("twitter", slug, url) + body)
    print(f"Ingested Twitter: {tweet_id}")
    return True


def ingest_url(url: str) -> bool:
    try:
        resp = requests.get(url, headers={"User-Agent": BROWSER_UA}, timeout=15)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove noise tags
        for tag in soup.find_all(["nav", "footer", "script", "style", "aside"]):
            tag.decompose()

        title_tag = soup.find("title")
        h1_tag = soup.find("h1")
        if title_tag and title_tag.string:
            title = title_tag.string.strip()
        elif h1_tag:
            title = h1_tag.get_text(strip=True)
        else:
            title = url

        # Find main content — priority order
        content_tag = (
            soup.find("article")
            or soup.find("main")
            or soup.find("div", class_=re.compile(r"\b(content|post|article|body)\b", re.I))
            or soup.find("body")
        )

        raw_md = markdownify(str(content_tag), heading_style="ATX").strip()
        clean_md = re.sub(r"\n{3,}", "\n\n", raw_md)

        slug = slugify(title)
        _write(slug, _frontmatter("web", title, url) + clean_md)
        print(f"Ingested URL: {title}")
        return True
    except Exception as exc:
        print(f"Error ingesting URL '{url}': {exc}")
        return False


def ingest(source: str) -> bool:
    if source.lower().endswith(".pdf"):
        return ingest_pdf(source)
    if "youtube.com" in source or "youtu.be" in source:
        return ingest_youtube(source)
    if "twitter.com" in source or "x.com" in source:
        return ingest_twitter(source)
    if source.startswith("http"):
        return ingest_url(source)

    print(
        f"Unknown source type: '{source}'\n\n"
        "Usage examples:\n"
        "  PDF    : wiki ingest path/to/paper.pdf\n"
        "  YouTube: wiki ingest https://youtu.be/<id>\n"
        "  Twitter: wiki ingest https://twitter.com/user/status/<id>\n"
        "  Web    : wiki ingest https://example.com/article"
    )
    return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline/ingest_sources.py <source> [<source> ...]")
        sys.exit(0)
    results = [ingest(src) for src in sys.argv[1:]]
    sys.exit(0 if all(results) else 1)
