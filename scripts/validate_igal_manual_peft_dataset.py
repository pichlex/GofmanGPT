"""Validate and merge manually curated Igal blog PEFT JSONL shards."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


LENGTH_BUCKETS = {
    "short": (5, 20),
    "normal": (21, 60),
    "expanded": (61, 140),
    "long": (141, 250),
}

LOW_QUALITY_USER_PHRASES = (
    "почему именно это выделил",
    "почему это выделил",
    "вот этот фрагмент",
    "этот фрагмент",
    "эта цитата",
    "прокомментируй цитату",
    "объясни цитату",
)

BROKEN_ASSISTANT_ENDINGS = (
    "и",
    "что",
    "т.",
    "т.е.",
    "см.",
    "греч.",
    "например",
)

ANNOUNCEMENT_ONLY_PATTERNS = (
    "есть еще одна очень интересная интерпритация",
    "есть еще одна очень интересная интерпретация",
    "существует такое понятие",
)

BLOCK_TAGS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "div",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "p",
    "section",
    "tr",
}


class BlogHTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "br":
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def text(self) -> str:
        value = "".join(self.parts)
        value = html.unescape(value).replace("\r\n", "\n").replace("\r", "\n")
        lines = [line.strip() for line in value.split("\n")]
        value = "\n".join(line for line in lines if line)
        return re.sub(r"\n{3,}", "\n\n", value).strip()


def clean_blog_text(content_html: str) -> str:
    parser = BlogHTMLTextExtractor()
    parser.feed(content_html)
    parser.close()
    return parser.text()


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def normalized_words(text: str) -> list[str]:
    return re.findall(r"[0-9A-Za-zА-Яа-яЁё]+", text.lower())


def has_shared_ngram(left: list[str], right: list[str], size: int = 4) -> bool:
    if len(left) < size or len(right) < size:
        return False
    right_ngrams = {tuple(right[index : index + size]) for index in range(len(right) - size + 1)}
    return any(tuple(left[index : index + size]) in right_ngrams for index in range(len(left) - size + 1))


def user_quality_error(user: str, assistant: str) -> str | None:
    user_lower = user.lower()
    if any(phrase in user_lower for phrase in LOW_QUALITY_USER_PHRASES):
        return "meta prompt"
    if has_shared_ngram(normalized_words(user), normalized_words(assistant)):
        return "copies assistant text"
    return None


def assistant_quality_error(assistant: str) -> str | None:
    stripped = assistant.strip()
    lower = stripped.lower()
    if re.match(r"^[а-яё]", stripped):
        return "starts mid-sentence"
    if any(lower.endswith(ending) for ending in BROKEN_ASSISTANT_ENDINGS):
        return "broken trailing phrase"
    if stripped.count("(") != stripped.count(")") or stripped.count("[") != stripped.count("]"):
        return "unbalanced brackets"
    if any(pattern in lower for pattern in ANNOUNCEMENT_ONLY_PATTERNS) and word_count(stripped) <= 10:
        return "announces content without content"
    numbered_lines = [line for line in stripped.splitlines() if re.match(r"^\s*\d+\s+\S+", line)]
    if len(numbered_lines) >= 2:
        return "looks like table of contents"
    return None


def length_bucket(text: str) -> str | None:
    count = word_count(text)
    for bucket, (lower, upper) in LENGTH_BUCKETS.items():
        if lower <= count <= upper:
            return bucket
    return None


def load_sources(raw_dir: Path) -> dict[str, dict[str, str]]:
    sources: dict[str, dict[str, str]] = {}
    for path in sorted(raw_dir.glob("*.json")):
        post = json.loads(path.read_text(encoding="utf-8"))
        sources[path.name] = {
            "text": clean_blog_text(post.get("content_html", "")),
            "title": str(post.get("title", "")),
            "url": str(post.get("url", "")),
            "published": str(post.get("published", "")),
        }
    return sources


def parse_sample(line: str, shard: Path, line_number: int) -> dict[str, Any]:
    try:
        sample = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{shard}:{line_number}: invalid JSON: {exc}") from exc
    if not isinstance(sample, dict):
        raise ValueError(f"{shard}:{line_number}: sample must be a JSON object")
    return sample


def validate_sample(
    sample: dict[str, Any],
    sources: dict[str, dict[str, str]],
    shard: Path,
    line_number: int,
) -> dict[str, Any]:
    location = f"{shard}:{line_number}"
    source_post = sample.get("source_post")
    if not isinstance(source_post, str) or not source_post:
        raise ValueError(f"{location}: missing string source_post")
    if source_post not in sources:
        raise ValueError(f"{location}: unknown source_post {source_post!r}")

    messages = sample.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError(f"{location}: messages must contain exactly user and assistant")
    if messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
        raise ValueError(f"{location}: messages roles must be user, assistant")
    user = messages[0].get("content")
    assistant = messages[1].get("content")
    if not isinstance(user, str) or not user.strip():
        raise ValueError(f"{location}: empty user content")
    if not isinstance(assistant, str) or not assistant.strip():
        raise ValueError(f"{location}: empty assistant content")
    if assistant_error := assistant_quality_error(assistant):
        raise ValueError(f"{location}: low-quality assistant: {assistant_error}")
    if quality_error := user_quality_error(user, assistant):
        raise ValueError(f"{location}: low-quality user prompt: {quality_error}")

    source_text = sources[source_post]["text"]
    if assistant not in source_text:
        raise ValueError(f"{location}: assistant is not an exact substring of cleaned source")

    bucket = length_bucket(assistant)
    if bucket is None:
        count = word_count(assistant)
        raise ValueError(f"{location}: assistant word count {count} is outside 5..250")

    normalized = {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "source_post": source_post,
        "source_url": sources[source_post]["url"],
        "source_title": sources[source_post]["title"],
        "published": sources[source_post]["published"],
        "assistant_word_count": word_count(assistant),
        "assistant_length_bucket": bucket,
        "source_hash": hashlib.sha256(source_text.encode("utf-8")).hexdigest()[:16],
    }
    return normalized


def validate_shards(raw_dir: Path, shard_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sources = load_sources(raw_dir)
    samples: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    seen_assistants: set[tuple[str, str]] = set()

    shard_paths = sorted(shard_dir.glob("*.jsonl"))
    if not shard_paths:
        raise ValueError(f"no JSONL shards found in {shard_dir}")

    for shard in shard_paths:
        for line_number, line in enumerate(shard.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            sample = validate_sample(parse_sample(line, shard, line_number), sources, shard, line_number)
            pair_key = (
                sample["messages"][0]["content"],
                sample["messages"][1]["content"],
            )
            if pair_key in seen_pairs:
                raise ValueError(f"{shard}:{line_number}: duplicate user/assistant pair")
            seen_pairs.add(pair_key)

            assistant_key = (sample["source_post"], sample["messages"][1]["content"])
            if assistant_key in seen_assistants:
                raise ValueError(f"{shard}:{line_number}: duplicate assistant span for source_post")
            seen_assistants.add(assistant_key)
            samples.append(sample)

    if len(samples) >= 10:
        unique_user_count = len({sample["messages"][0]["content"] for sample in samples})
        if unique_user_count / len(samples) < 0.8:
            raise ValueError(
                f"low unique user prompt share: {unique_user_count}/{len(samples)}"
            )

    by_source: dict[str, list[str]] = {}
    for sample in samples:
        by_source.setdefault(sample["source_post"], []).append(sample["messages"][1]["content"])
    for source_post, assistants in by_source.items():
        for outer_index, outer in enumerate(assistants):
            for inner_index, inner in enumerate(assistants):
                if outer_index == inner_index:
                    continue
                if inner in outer:
                    raise ValueError(f"nested assistant span in {source_post}")

    bucket_counts = Counter(sample["assistant_length_bucket"] for sample in samples)
    total = len(samples)
    manifest = {
        "actual_count": total,
        "shard_count": len(shard_paths),
        "source_post_count": len({sample["source_post"] for sample in samples}),
        "unique_user_prompts": len({sample["messages"][0]["content"] for sample in samples}),
        "length_buckets": {
            bucket: {
                "count": bucket_counts[bucket],
                "share": round(bucket_counts[bucket] / total, 4) if total else 0.0,
            }
            for bucket in LENGTH_BUCKETS
        },
    }
    return samples, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/igal-blogspot/posts"))
    parser.add_argument("--shard-dir", type=Path, default=Path("data/processed/igal-peft/manual_shards"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/igal-peft/live_speech_2400.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/igal-peft/manifest.json"))
    parser.add_argument("--target-count", type=int, default=2400)
    args = parser.parse_args()

    samples, manifest = validate_shards(args.raw_dir, args.shard_dir)
    manifest["target_count"] = args.target_count
    manifest["count_matches_target"] = len(samples) == args.target_count

    if len(samples) != args.target_count:
        raise SystemExit(f"expected {args.target_count} samples, got {len(samples)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(sample, ensure_ascii=False) + "\n" for sample in samples),
        encoding="utf-8",
    )
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
