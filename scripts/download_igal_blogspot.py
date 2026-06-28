#!/usr/bin/env python3
"""Download all posts from a Blogger blog into a raw data directory."""

from __future__ import annotations

import argparse
import html
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "https://httpigal-igal.blogspot.com/"
DEFAULT_OUT_DIR = Path("data/raw/igal-blogspot")
DEFAULT_MAX_RESULTS = 500
USER_AGENT = "GofmanGPT blogspot downloader/1.0"

CYRILLIC_TRANSLIT = str.maketrans(
    {
        "а": "a",
        "б": "b",
        "в": "v",
        "г": "g",
        "д": "d",
        "е": "e",
        "ё": "e",
        "ж": "zh",
        "з": "z",
        "и": "i",
        "й": "i",
        "к": "k",
        "л": "l",
        "м": "m",
        "н": "n",
        "о": "o",
        "п": "p",
        "р": "r",
        "с": "s",
        "т": "t",
        "у": "u",
        "ф": "f",
        "х": "h",
        "ц": "ts",
        "ч": "ch",
        "ш": "sh",
        "щ": "sch",
        "ъ": "",
        "ы": "y",
        "ь": "",
        "э": "e",
        "ю": "yu",
        "я": "ya",
    }
)


def build_feed_url(base_url: str, start_index: int, max_results: int) -> str:
    query = urlencode(
        {
            "alt": "json",
            "max-results": max_results,
            "start-index": start_index,
        }
    )
    return f"{urljoin(base_url.rstrip('/') + '/', 'feeds/posts/default')}?{query}"


def _text(value: dict[str, Any] | None) -> str:
    if not value:
        return ""
    return str(value.get("$t", ""))


def parse_posts(feed: dict[str, Any]) -> list[dict[str, Any]]:
    posts: list[dict[str, Any]] = []
    for entry in feed.get("feed", {}).get("entry", []) or []:
        alternate_url = ""
        for link in entry.get("link", []) or []:
            if link.get("rel") == "alternate":
                alternate_url = str(link.get("href", ""))
                break

        posts.append(
            {
                "id": _text(entry.get("id")),
                "published": _text(entry.get("published")),
                "updated": _text(entry.get("updated")),
                "title": html.unescape(_text(entry.get("title"))),
                "url": alternate_url,
                "labels": [
                    str(category.get("term", ""))
                    for category in entry.get("category", []) or []
                    if category.get("term")
                ],
                "content_html": _text(entry.get("content")),
            }
        )
    return posts


def slugify(value: str) -> str:
    value = value.lower().translate(CYRILLIC_TRANSLIT)
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return value or "post"


def post_stem(post: dict[str, Any]) -> str:
    date = str(post.get("published", ""))[:10] or "undated"
    slug = slugify(str(post.get("title", "")))
    raw_id = str(post.get("id", ""))
    match = re.search(r"(?:post-|\.post-?)(\d+)", raw_id)
    post_id = match.group(1) if match else re.sub(r"\D+", "", raw_id)[-12:]
    return "-".join(part for part in [date, slug, post_id] if part)


def index_json(posts: list[dict[str, Any]]) -> str:
    sorted_posts = sorted(posts, key=lambda item: (item.get("published", ""), item.get("id", "")))
    payload = {
        "source": DEFAULT_BASE_URL,
        "count": len(sorted_posts),
        "posts": [
            {
                key: post.get(key, "")
                for key in ("id", "published", "updated", "title", "url", "labels")
            }
            for post in sorted_posts
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def fetch_json(url: str, retries: int = 3, timeout: int = 30) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        request = Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(attempt)
    raise RuntimeError(f"Could not fetch {url}: {last_error}") from last_error


def write_posts(out_dir: Path, posts: list[dict[str, Any]]) -> None:
    posts_dir = out_dir / "posts"
    posts_dir.mkdir(parents=True, exist_ok=True)
    used_names: set[str] = set()

    for post in posts:
        stem = post_stem(post)
        unique_stem = stem
        suffix = 2
        while unique_stem in used_names:
            unique_stem = f"{stem}-{suffix}"
            suffix += 1
        used_names.add(unique_stem)

        (posts_dir / f"{unique_stem}.html").write_text(
            str(post.get("content_html", "")),
            encoding="utf-8",
        )
        (posts_dir / f"{unique_stem}.json").write_text(
            json.dumps(post, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def download_blog(base_url: str, out_dir: Path, max_results: int, delay: float) -> list[dict[str, Any]]:
    feed_dir = out_dir / "feed"
    feed_dir.mkdir(parents=True, exist_ok=True)

    all_posts: list[dict[str, Any]] = []
    start_index = 1
    page = 1

    while True:
        url = build_feed_url(base_url, start_index=start_index, max_results=max_results)
        print(f"Fetching page {page}: {url}")
        feed = fetch_json(url)
        (feed_dir / f"page-{page:04d}.json").write_text(
            json.dumps(feed, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        posts = parse_posts(feed)
        if not posts:
            break

        all_posts.extend(posts)
        if len(posts) < max_results:
            break

        start_index += max_results
        page += 1
        if delay:
            time.sleep(delay)

    write_posts(out_dir, all_posts)
    (out_dir / "index.json").write_text(index_json(all_posts), encoding="utf-8")
    return all_posts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-results", type=int, default=DEFAULT_MAX_RESULTS)
    parser.add_argument("--delay", type=float, default=0.2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    posts = download_blog(args.base_url, args.out_dir, args.max_results, args.delay)
    print(f"Downloaded {len(posts)} posts into {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
