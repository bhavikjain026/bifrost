"""Tests with a fake Tavily: no network, no credits.  Run: python3 -m pytest -q  (or python3 test_freshness.py)"""
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from claims import flag, search_query, values_in  # noqa: E402
from server import check_answer  # noqa: E402
from verify import Budget, Config, value_in  # noqa: E402

NOW = datetime(2026, 11, 3, tzinfo=timezone.utc)


def fake_search(results):
    calls = []

    def search(query, api_key, max_results=5, live=False):
        calls.append(query)
        return {"request_id": "req-1", "usage": {"credits": 1}, "results": results}
    search.calls = calls
    return search


class TestClaims(unittest.TestCase):
    def test_flags_time_sensitive_claims(self):
        cases = {
            "ChatGPT Plus costs $20 per month.": "pricing",
            "Satya Nadella is the CEO of Microsoft.": "roles",
            "The latest stable version of Python is 3.12.": "software",
            "India's repo rate is 6.5%.": "statistics",
            "Bitcoin is trading around $68,000.": "markets",
            "A visa is required for Indian citizens visiting Japan.": "policy",
            "The new model was released recently.": "recency",
        }
        for text, cat in cases.items():
            c = flag(text)[0]
            self.assertTrue(c.flagged, text)
            self.assertEqual(c.category, cat, text)

    def test_leaves_stable_facts_alone(self):
        for text in ["Water boils at 100 degrees Celsius at sea level.", "Paris is the capital of France.",
                     "Mix flour, water and yeast, then knead for ten minutes.", "Ohm's law relates voltage and current."]:
            self.assertFalse(flag(text)[0].flagged, text)

    def test_split_and_values(self):
        claims = flag("Python 3.12 is the latest version. It was first released in 1991.\n- Pricing: $20 per month")
        self.assertEqual(len(claims), 3)
        self.assertIn("3.12", claims[0].values)
        self.assertIn("$20", values_in("It costs $20 per month"))

    def test_query_drops_old_value(self):
        c = flag("ChatGPT Plus costs $20 per month.")[0]
        q = search_query(c)
        self.assertNotIn("20", q)
        self.assertIn("ChatGPT", q)

    def test_value_matching(self):
        self.assertTrue(value_in("₹1,299", "now priced at Rs 1299 a month"))
        self.assertFalse(value_in("$20", "now $22 per month"))
        self.assertTrue(value_in("Satya Nadella", "CEO Satya Nadella said"))


class TestCheck(unittest.TestCase):
    cfg = Config(api_key="test", max_claims=5)

    def test_supported(self):
        s = fake_search([{"url": "https://openai.com/plus", "title": "Plus", "published_date": "2026-10-20",
                          "content": "ChatGPT Plus costs $20 per month and includes GPT access."}])
        r = check_answer("ChatGPT Plus costs $20 per month.", cfg=self.cfg, budget=Budget(100), search=s)
        c = r["claims"][0]
        self.assertEqual(c["verdict"], "supported")
        self.assertGreaterEqual(c["score"], 70)
        self.assertEqual(c["label"], "fresh")

    def test_contradicted_needs_two_sources(self):
        res = [{"url": f"https://news{i}.example/plus", "title": "Price rise", "published_date": "2026-10-25",
                "content": "ChatGPT Plus price per month is now $25 after the change."} for i in range(2)]
        r = check_answer("ChatGPT Plus costs $20 per month.", cfg=self.cfg, budget=Budget(100), search=fake_search(res))
        c = r["claims"][0]
        self.assertEqual(c["verdict"], "contradicted")
        self.assertEqual(c["label"], "stale")
        self.assertLess(c["score"], 40)
        one = check_answer("ChatGPT Plus costs $20 per month.", cfg=self.cfg, budget=Budget(100), search=fake_search(res[:1]))
        self.assertEqual(one["claims"][0]["verdict"], "not-enough-info")

    def test_irrelevant_numbers_do_not_contradict(self):
        res = [{"url": f"https://x{i}.example", "title": "Other", "content": "Netflix basic plan is $7 a month."} for i in range(3)]
        r = check_answer("ChatGPT Plus costs $20 per month.", cfg=self.cfg, budget=Budget(100), search=fake_search(res))
        self.assertEqual(r["claims"][0]["verdict"], "not-enough-info")

    def test_budget_cap_and_no_key(self):
        s = fake_search([])
        r = check_answer("ChatGPT Plus costs $20 per month. Python 3.12 is the latest version.",
                         cfg=self.cfg, budget=Budget(1), search=s)
        verdicts = sorted(c["verdict"] for c in r["claims"])
        self.assertIn("not-checked", verdicts)
        self.assertEqual(len(s.calls), 1)
        nokey = check_answer("ChatGPT Plus costs $20 per month.", cfg=Config(api_key=""), budget=Budget(100), search=s)
        self.assertEqual(nokey["claims"][0]["reason"], "no TAVILY_API_KEY")

    def test_unflagged_sentences_cost_nothing(self):
        s = fake_search([])
        r = check_answer("Paris is the capital of France. Water boils at 100 degrees.", cfg=self.cfg, budget=Budget(100), search=s)
        self.assertEqual(r["summary"]["flagged"], 0)
        self.assertEqual(s.calls, [])



class TestUnconfirmed(unittest.TestCase):
    """Could not confirm is not the same as out of date (FreshQA run, 4 Nov 2026: a
    fast-changing claim with no deciding line used to be labelled stale, and the
    regeneration was told the answer was out of date with no source)."""
    cfg = Config(api_key="test")
    answer = "Bitcoin is trading around $68,000."
    unrelated = [{"url": "https://x.example", "title": "Crypto", "content": "Crypto markets were busy this week."}]

    def test_unconfirmed_fast_claim_is_not_stale(self):
        r = check_answer(self.answer, cfg=self.cfg, budget=Budget(100), search=fake_search(self.unrelated))
        c = r["claims"][0]
        self.assertEqual(c["verdict"], "not-enough-info")
        self.assertEqual(c["label"], "unverified")
        self.assertTrue(c["urgent"])
        self.assertEqual((r["summary"]["stale"], r["summary"]["unconfirmed"]), (0, 1))
        self.assertEqual(r["question_evidence"], [])  # no question given: no extra search

    def test_question_search_returns_relevant_lines_only(self):
        q_results = [{"url": "https://prices.example/btc", "title": "BTC", "published_date": "2026-11-04",
                      "content": "Bitcoin price today is $91,200. Weather in London is mild."}]
        calls = []

        def search(query, api_key, max_results=5, live=False):
            calls.append(query)
            res = self.unrelated if len(calls) == 1 else q_results
            return {"usage": {"credits": 1}, "results": res}
        r = check_answer(self.answer, "What is the price of Bitcoin today?", cfg=self.cfg, budget=Budget(100), search=search)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1], "What is the price of Bitcoin today?")
        self.assertEqual([e["line"] for e in r["question_evidence"]], ["Bitcoin price today is $91,200."])
        self.assertEqual(r["summary"]["credits"], 2)

    def test_no_question_search_when_contradicted_or_disabled(self):
        res = [{"url": f"https://n{i}.example", "title": "BTC", "content": "Bitcoin is trading around $91,200 today."} for i in range(2)]
        s = fake_search(res)
        r = check_answer(self.answer, "What is the price of Bitcoin?", cfg=self.cfg, budget=Budget(100), search=s)
        self.assertEqual(r["summary"]["stale"], 1)
        self.assertEqual(len(s.calls), 1)
        s2 = fake_search(self.unrelated)
        off = Config(api_key="test", question_search=False)
        r2 = check_answer(self.answer, "What is the price of Bitcoin?", cfg=off, budget=Budget(100), search=s2)
        self.assertEqual((r2["summary"]["unconfirmed"], len(s2.calls), r2["question_evidence"]), (1, 1, []))

    def test_slow_unconfirmed_claim_triggers_nothing(self):
        r = check_answer("ChatGPT Plus costs $20 per month.", "How much is ChatGPT Plus?", cfg=self.cfg,
                         budget=Budget(100), search=fake_search(self.unrelated))
        self.assertEqual((r["summary"]["stale"], r["summary"]["unconfirmed"]), (0, 0))


class TestRealCases(unittest.TestCase):
    """Regression cases from the first live run (3 Nov 2026): the rules-only mode wrongly
    called these supported because the old value merely appeared in a source line."""
    cfg = Config(api_key="test")

    def one(self, answer, results):
        return check_answer(answer, cfg=self.cfg, budget=Budget(100), search=fake_search(results))["claims"][0]

    def test_replaced_ceo_is_not_support(self):
        res = [{"url": "https://en.wikipedia.org/wiki/Twitter,_Inc.", "title": "Twitter, Inc. - Wikipedia",
                "published_date": "2026-08-08", "content": '"IITian Parag Agrawal to replace Jack Dorsey as Twitter CEO".'}]
        c = self.one("The CEO of Twitter is Jack Dorsey.", res)
        self.assertEqual(c["verdict"], "contradicted")
        self.assertEqual(c["label"], "stale")

    def test_older_version_mention_is_not_support(self):
        res = [{"url": "https://a.example", "title": "The Art of Choosing Stable Python Version", "published_date": "2022-05-10",
                "content": "As by this time python 3.9 is significantly more stable than the latest releases."},
               {"url": "https://b.example", "title": "Which Python Version Should I Install?", "published_date": "2026-09-21",
                "content": "Python 3.14 is the latest stable release. Projects might require an older Python 3 minor version, eg 3.9 or 3.10."}]
        c = self.one("The latest stable version of Python is 3.9.", res)
        self.assertEqual((c["verdict"], c["newer_value"]), ("contradicted", "3.14"))

    def test_plain_support_still_works(self):
        res = [{"url": "https://c.example", "title": "ChatGPT pricing", "published_date": "2026-04-30",
                "content": "The ChatGPT Plus cost has held at $20 per month since launch."}]
        self.assertEqual(self.one("ChatGPT Plus costs $20 per month.", res)["verdict"], "supported")


class TestJudge(unittest.TestCase):
    """The local model only points at lines; code must reject citations that do not hold the value."""

    def run_with_judge(self, reply, results):
        import json
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                body = json.dumps({"message": {"content": json.dumps(reply)}}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        srv = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            cfg = Config(api_key="test", judge_model="fake", ollama_url=f"http://127.0.0.1:{srv.server_port}")
            return check_answer("ChatGPT Plus costs $20 per month.", cfg=cfg, budget=Budget(100), search=fake_search(results))["claims"][0]
        finally:
            srv.shutdown()
            srv.server_close()

    res = [{"url": "https://a.example", "title": "A", "published_date": "2026-10-30",
            "content": "ChatGPT Plus now costs $25 per month. Other plans are unchanged."}]

    def test_contradiction_verified_by_code(self):
        c = self.run_with_judge({"verdict": "contradicted", "lines": [1], "newer_value": "$25"}, self.res)
        self.assertEqual((c["verdict"], c["newer_value"], c["label"]), ("contradicted", "$25", "stale"))

    def test_hallucinated_support_is_downgraded(self):
        c = self.run_with_judge({"verdict": "supported", "lines": [1]}, self.res)
        self.assertEqual(c["verdict"], "not-enough-info")

    def test_made_up_newer_value_is_downgraded(self):
        c = self.run_with_judge({"verdict": "contradicted", "lines": [2], "newer_value": "$30"}, self.res)
        self.assertEqual(c["verdict"], "not-enough-info")


if __name__ == "__main__":
    unittest.main()
