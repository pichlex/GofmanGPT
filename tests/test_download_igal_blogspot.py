import json
import unittest

from scripts import download_igal_blogspot as downloader


class BloggerDownloaderTests(unittest.TestCase):
    def test_builds_blogger_feed_url_with_pagination(self):
        url = downloader.build_feed_url(
            "https://httpigal-igal.blogspot.com/",
            start_index=501,
            max_results=500,
        )

        self.assertEqual(
            url,
            "https://httpigal-igal.blogspot.com/feeds/posts/default"
            "?alt=json&max-results=500&start-index=501",
        )

    def test_extracts_posts_from_blogger_json_feed(self):
        feed = {
            "feed": {
                "entry": [
                    {
                        "id": {"$t": "tag:blogger.com,1999:blog-1.post-42"},
                        "published": {"$t": "2012-03-04T10:20:30.000+03:00"},
                        "updated": {"$t": "2012-03-05T10:20:30.000+03:00"},
                        "title": {"$t": "Hello, Blogger!"},
                        "content": {"$t": "<p>Body</p>"},
                        "category": [{"term": "AI"}, {"term": "Notes"}],
                        "link": [
                            {"rel": "self", "href": "feed-url"},
                            {
                                "rel": "alternate",
                                "href": "https://example.com/2012/03/hello.html",
                            },
                        ],
                    }
                ]
            }
        }

        posts = downloader.parse_posts(feed)

        self.assertEqual(
            posts,
            [
                {
                    "id": "tag:blogger.com,1999:blog-1.post-42",
                    "published": "2012-03-04T10:20:30.000+03:00",
                    "updated": "2012-03-05T10:20:30.000+03:00",
                    "title": "Hello, Blogger!",
                    "url": "https://example.com/2012/03/hello.html",
                    "labels": ["AI", "Notes"],
                    "content_html": "<p>Body</p>",
                }
            ],
        )

    def test_makes_safe_post_filename_from_date_and_title(self):
        post = {
            "published": "2012-03-04T10:20:30.000+03:00",
            "title": "Hello, Blogger! Привет",
            "id": "tag:blogger.com,1999:blog-1.post-42",
        }

        self.assertEqual(
            downloader.post_stem(post),
            "2012-03-04-hello-blogger-privet-42",
        )

    def test_index_json_is_sorted_by_publish_date(self):
        posts = [
            {"published": "2020-01-02T00:00:00.000+03:00", "title": "B"},
            {"published": "2019-01-02T00:00:00.000+03:00", "title": "A"},
        ]

        payload = json.loads(downloader.index_json(posts))

        self.assertEqual([post["title"] for post in payload["posts"]], ["A", "B"])
        self.assertEqual(payload["count"], 2)


if __name__ == "__main__":
    unittest.main()
