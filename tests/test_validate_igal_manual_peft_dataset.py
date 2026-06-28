import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.validate_igal_manual_peft_dataset import validate_shards


def write_post(path: Path, content: str) -> None:
    path.write_text(
        json.dumps(
            {
                "title": "Post",
                "url": "https://example.test/post",
                "published": "2010-01-01T00:00:00.000+00:00",
                "content_html": f"<p>{content}</p>",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


class ValidateIgalManualPeftDatasetTests(unittest.TestCase):
    def test_rejects_user_that_copies_assistant_prefix_with_meta_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            shard_dir = root / "shards"
            raw_dir.mkdir()
            shard_dir.mkdir()
            assistant = "Родился 19 марта 1906 в Золингене."
            write_post(raw_dir / "post.json", assistant)
            (shard_dir / "shard.jsonl").write_text(
                json.dumps(
                    {
                        "source_post": "post.json",
                        "messages": [
                            {
                                "role": "user",
                                "content": "Почему именно это выделил? Родился 19 марта 1906 в",
                            },
                            {"role": "assistant", "content": assistant},
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "low-quality user prompt"):
                validate_shards(raw_dir, shard_dir)

    def test_rejects_assistant_that_looks_like_table_of_contents(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            shard_dir = root / "shards"
            raw_dir.mkdir()
            shard_dir.mkdir()
            assistant = "4 Законотворчество\n5 См. также"
            write_post(raw_dir / "post.json", assistant)
            (shard_dir / "shard.jsonl").write_text(
                json.dumps(
                    {
                        "source_post": "post.json",
                        "messages": [
                            {"role": "user", "content": "что там было в оглавлении?"},
                            {"role": "assistant", "content": assistant},
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "low-quality assistant"):
                validate_shards(raw_dir, shard_dir)

    def test_rejects_assistant_with_broken_trailing_abbreviation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            shard_dir = root / "shards"
            raw_dir.mkdir()
            shard_dir.mkdir()
            assistant = "А. принимал именно в психиатрической больнице, т."
            write_post(raw_dir / "post.json", assistant)
            (shard_dir / "shard.jsonl").write_text(
                json.dumps(
                    {
                        "source_post": "post.json",
                        "messages": [
                            {"role": "user", "content": "где он принимал?"},
                            {"role": "assistant", "content": assistant},
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "low-quality assistant"):
                validate_shards(raw_dir, shard_dir)

    def test_rejects_assistant_with_unbalanced_opening_parenthesis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            shard_dir = root / "shards"
            raw_dir.mkdir()
            shard_dir.mkdir()
            assistant = "который после рождения близнецов стал называться Делосом (греч."
            write_post(raw_dir / "post.json", assistant)
            (shard_dir / "shard.jsonl").write_text(
                json.dumps(
                    {
                        "source_post": "post.json",
                        "messages": [
                            {"role": "user", "content": "как стал называться остров?"},
                            {"role": "assistant", "content": assistant},
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "low-quality assistant"):
                validate_shards(raw_dir, shard_dir)

    def test_rejects_assistant_ending_with_te_without_explanation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            shard_dir = root / "shards"
            raw_dir.mkdir()
            shard_dir.mkdir()
            assistant = "В вычислительной технике существует такое понятие, как виртуальность-т.е."
            write_post(raw_dir / "post.json", assistant)
            (shard_dir / "shard.jsonl").write_text(
                json.dumps(
                    {
                        "source_post": "post.json",
                        "messages": [
                            {"role": "user", "content": "что такое виртуальность в этой аналогии?"},
                            {"role": "assistant", "content": assistant},
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "low-quality assistant"):
                validate_shards(raw_dir, shard_dir)

    def test_rejects_assistant_that_only_announces_interpretation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            shard_dir = root / "shards"
            raw_dir.mkdir()
            shard_dir.mkdir()
            assistant = "Есть еще одна очень интересная интерпритация того же."
            write_post(raw_dir / "post.json", assistant)
            (shard_dir / "shard.jsonl").write_text(
                json.dumps(
                    {
                        "source_post": "post.json",
                        "messages": [
                            {"role": "user", "content": "что за новая интерпретация там дана?"},
                            {"role": "assistant", "content": assistant},
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "low-quality assistant"):
                validate_shards(raw_dir, shard_dir)

    def test_rejects_low_unique_user_prompt_share(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            shard_dir = root / "shards"
            raw_dir.mkdir()
            shard_dir.mkdir()
            texts = [f"Это самостоятельное предложение номер {idx} с нормальным окончанием." for idx in range(1, 11)]
            write_post(raw_dir / "post.json", " ".join(texts))
            with (shard_dir / "shard.jsonl").open("w", encoding="utf-8") as fh:
                for idx, assistant in enumerate(texts):
                    user = "что здесь было в самом деле?" if idx < 8 else f"живой вопрос {idx}?"
                    fh.write(
                        json.dumps(
                            {
                                "source_post": "post.json",
                                "messages": [
                                    {"role": "user", "content": user},
                                    {"role": "assistant", "content": assistant},
                                ],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            with self.assertRaisesRegex(ValueError, "low unique user prompt share"):
                validate_shards(raw_dir, shard_dir)

    def test_rejects_nested_assistant_spans_from_same_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            shard_dir = root / "shards"
            raw_dir.mkdir()
            shard_dir.mkdir()
            first = "Первое самостоятельное предложение достаточно длинное для проверки."
            second = "Второе самостоятельное предложение тоже достаточно длинное для проверки."
            write_post(raw_dir / "post.json", f"{first} {second}")
            rows = [
                ("что первое?", first),
                ("что вся мысль?", f"{first} {second}"),
            ]
            with (shard_dir / "shard.jsonl").open("w", encoding="utf-8") as fh:
                for user, assistant in rows:
                    fh.write(
                        json.dumps(
                            {
                                "source_post": "post.json",
                                "messages": [
                                    {"role": "user", "content": user},
                                    {"role": "assistant", "content": assistant},
                                ],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            with self.assertRaisesRegex(ValueError, "nested assistant span"):
                validate_shards(raw_dir, shard_dir)

    def test_rejects_assistant_starting_lowercase_mid_sentence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            shard_dir = root / "shards"
            raw_dir.mkdir()
            shard_dir.mkdir()
            assistant = "упоминается их родина, которую они считали родиной всех людей."
            write_post(raw_dir / "post.json", assistant)
            (shard_dir / "shard.jsonl").write_text(
                json.dumps(
                    {
                        "source_post": "post.json",
                        "messages": [
                            {"role": "user", "content": "что они считали родиной?"},
                            {"role": "assistant", "content": assistant},
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "low-quality assistant"):
                validate_shards(raw_dir, shard_dir)


if __name__ == "__main__":
    unittest.main()
