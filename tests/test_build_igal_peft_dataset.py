import json
import re
import tempfile
import unittest
from pathlib import Path

from scripts import build_igal_peft_dataset as builder


def write_post(path: Path, title: str, content: str) -> None:
    path.write_text(
        json.dumps(
            {
                "id": f"post-{path.stem}",
                "published": "2010-01-01T00:00:00.000+00:00",
                "updated": "2010-01-01T00:00:00.000+00:00",
                "title": title,
                "url": f"https://example.test/{path.stem}",
                "labels": [],
                "content_html": content,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


class BuildIgalPeftDatasetTests(unittest.TestCase):
    def test_builds_requested_number_of_chat_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw"
            out_dir = Path(tmp) / "processed"
            raw_dir.mkdir()
            words = " ".join(f"слово{i}" for i in range(1, 2500))
            write_post(raw_dir / "post-a.json", "Большой текст", f"<p>{words}</p>")

            samples, manifest = builder.build_dataset(raw_dir, out_dir, target_count=120)

            self.assertEqual(len(samples), 120)
            self.assertEqual(manifest["target_count"], 120)
            self.assertEqual(manifest["actual_count"], 120)
            self.assertTrue((out_dir / "live_speech_120.jsonl").exists())
            self.assertTrue((out_dir / "manifest.json").exists())

    def test_samples_use_chat_schema_and_source_metadata(self):
        post_text = " ".join(f"текст{i}" for i in range(1, 500))
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw"
            out_dir = Path(tmp) / "processed"
            raw_dir.mkdir()
            write_post(raw_dir / "post-a.json", "Тема", f"<p>{post_text}</p>")

            samples, _ = builder.build_dataset(raw_dir, out_dir, target_count=20)

            sample = samples[0]
            self.assertEqual([m["role"] for m in sample["messages"]], ["user", "assistant"])
            self.assertIn(sample["assistant_length_bucket"], builder.LENGTH_BUCKETS)
            self.assertEqual(sample["source_post"], "post-a.json")
            self.assertGreaterEqual(sample["assistant_word_count"], 5)
            self.assertIn(sample["messages"][1]["content"], post_text)

    def test_length_mix_matches_requested_live_speech_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw"
            out_dir = Path(tmp) / "processed"
            raw_dir.mkdir()
            for idx in range(3):
                words = " ".join(f"w{idx}_{i}" for i in range(1, 5000))
                write_post(raw_dir / f"post-{idx}.json", f"Title {idx}", f"<p>{words}</p>")

            samples, manifest = builder.build_dataset(raw_dir, out_dir, target_count=400)
            buckets = manifest["length_buckets"]

            self.assertEqual(sum(item["count"] for item in buckets.values()), len(samples))
            self.assertGreaterEqual(buckets["short"]["share"], 0.35)
            self.assertLessEqual(buckets["short"]["share"], 0.45)
            self.assertGreaterEqual(buckets["normal"]["share"], 0.40)
            self.assertLessEqual(buckets["normal"]["share"], 0.50)
            self.assertGreaterEqual(buckets["expanded"]["share"], 0.10)
            self.assertLessEqual(buckets["expanded"]["share"], 0.20)
            self.assertLessEqual(buckets["long"]["share"], 0.05)

    def test_user_prompts_are_varied(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw"
            out_dir = Path(tmp) / "processed"
            raw_dir.mkdir()
            words = " ".join(f"слово{i}" for i in range(1, 4000))
            write_post(raw_dir / "post-a.json", "Большой текст", f"<p>{words}</p>")

            samples, manifest = builder.build_dataset(raw_dir, out_dir, target_count=200)
            users = [sample["messages"][0]["content"] for sample in samples]

            self.assertGreaterEqual(len(set(users)), 120)
            self.assertGreaterEqual(manifest["unique_user_prompts"], 120)

    def test_user_prompt_is_grounded_in_assistant_span_not_only_title(self):
        candidate = builder.Candidate(
            source_post="post-a.json",
            source_url="https://example.test/a",
            title="Общий заголовок",
            published="2010-01-01T00:00:00.000+00:00",
            bucket="normal",
            text="Сначала человек вспоминает старый двор, и странный разговор у ворот.",
            start_word=0,
            end_word=10,
            start_char=0,
            end_char=68,
            overlap_ratio=0.0,
            source_hash="abc",
        )

        sample = builder.candidates_to_samples([candidate])[0]
        user = sample["messages"][0]["content"].lower()

        self.assertNotIn("общий заголовок", user)
        self.assertTrue(builder.user_is_grounded(user, candidate.text))
        self.assertIn(sample["prompt_family"], builder.GROUNDED_PROMPT_FAMILIES)
        self.assertEqual(sample["messages"][1]["content"], candidate.text)

    def test_assistant_preserves_source_punctuation(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw"
            out_dir = Path(tmp) / "processed"
            raw_dir.mkdir()
            text = " ".join(
                [
                    "Начало, где есть запятая.",
                    "Потом вопрос: почему так?",
                    "И дальше — авторская интонация!",
                ]
                * 80
            )
            write_post(raw_dir / "post-a.json", "Пунктуация", f"<p>{text}</p>")

            samples, _ = builder.build_dataset(raw_dir, out_dir, target_count=40)

            self.assertTrue(any(re.search(r"[,.!?;:—-]", sample["messages"][1]["content"]) for sample in samples))

    def test_dataset_manifest_reports_grounded_user_prompts(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw"
            out_dir = Path(tmp) / "processed"
            raw_dir.mkdir()
            words = " ".join(f"история{i}" for i in range(1, 3000))
            write_post(raw_dir / "post-a.json", "Заголовок", f"<p>{words}</p>")

            samples, manifest = builder.build_dataset(raw_dir, out_dir, target_count=100)

            self.assertEqual(manifest["grounded_user_prompts"], 100)
            self.assertTrue(all(builder.user_is_grounded(s["messages"][0]["content"], s["messages"][1]["content"]) for s in samples))


if __name__ == "__main__":
    unittest.main()
