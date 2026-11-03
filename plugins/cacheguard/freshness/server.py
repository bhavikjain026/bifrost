"""Best-Before freshness checker: takes any LLM answer, flags claims likely to have
gone stale, re-verifies them against live sources (Tavily) and scores each claim.

    TAVILY_API_KEY=tvly-... python3 server.py            (rules only)
    TAVILY_API_KEY=tvly-... FRESHNESS_JUDGE_MODEL=qwen2.5:7b python3 server.py

Open http://127.0.0.1:8098 to paste an answer, or POST /check:
    {"answer": "...", "question": "optional"}
Standard library only.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from claims import flag
from verify import Budget, Config, check_claim

CFG = Config.from_env()
BUDGET = Budget(CFG.credits_per_hour)
PAGE = (Path(__file__).parent / "index.html").read_bytes()
MAX_BODY = 64 << 10


def check_answer(answer: str, question: str = "", cfg: Config = CFG, budget: Budget = BUDGET, search=None) -> dict:
    t0 = time.time()
    claims = flag(answer)
    flagged = [c for c in claims if c.flagged][: cfg.max_claims]
    kwargs = {"search": search} if search else {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        checks = list(pool.map(lambda c: check_claim(c, question, cfg, budget, **kwargs), flagged))
    by_index = {c.index: k for c, k in zip(flagged, checks)}
    out = []
    for c in claims:
        item = {"index": c.index, "text": c.text, "flagged": c.flagged, "category": c.category,
                "rules": c.rules, "values": c.values}
        if c.index in by_index:
            item.update(asdict(by_index[c.index]))
        elif c.flagged:
            item.update({"verdict": "not-checked", "reason": f"over max_claims={cfg.max_claims}", "score": 0, "label": "unverified"})
        out.append(item)
    scored = [x for x in out if x.get("verdict") in ("supported", "contradicted", "not-enough-info")]
    return {
        "claims": out,
        "summary": {
            "claims": len(claims),
            "flagged": sum(1 for c in claims if c.flagged),
            "checked": len(scored),
            "stale": sum(1 for x in scored if x["label"] == "stale"),
            "min_score": min((x["score"] for x in scored), default=None),
            "credits": sum(x.get("credits", 0) for x in out),
            "mode": f"judge:{cfg.judge_model}" if cfg.judge_model else "rules",
            "ms": round((time.time() - t0) * 1000),
        },
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str = "application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif self.path == "/health":
            self._send(200, json.dumps({"ok": True, "tavily_key": bool(CFG.api_key), "judge": CFG.judge_model or None}).encode())
        else:
            self._send(404, b'{"error":"not found"}')

    def do_POST(self):
        if self.path != "/check":
            return self._send(404, b'{"error":"not found"}')
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > MAX_BODY:
            return self._send(413, b'{"error":"body must be 1 byte to 64 KiB"}')
        try:
            req = json.loads(self.rfile.read(n))
            answer = str(req["answer"])
        except (ValueError, KeyError, TypeError):
            return self._send(400, b'{"error":"send JSON {\\"answer\\": \\"...\\", \\"question\\": \\"optional\\"}"}')
        result = check_answer(answer, str(req.get("question") or ""))
        self._send(200, json.dumps(result, ensure_ascii=False).encode())

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    port = int(os.environ.get("FRESHNESS_PORT", "8098"))
    print(f"freshness checker on http://127.0.0.1:{port}  (tavily key: {'yes' if CFG.api_key else 'MISSING'}, "
          f"mode: {'judge ' + CFG.judge_model if CFG.judge_model else 'rules'}, cap {CFG.credits_per_hour} credits/h)")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
