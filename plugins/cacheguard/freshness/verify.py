"""Re-verify flagged claims against live web sources (Tavily Search) and score them.

The decision is made by code. An optional local model (Ollama) only points at a
numbered source line; code then checks that the claimed (or the newer) value is
really in that line. Without a model, a rules-only matcher is used.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from claims import Claim, search_query

TAVILY_URL = "https://api.tavily.com/search"


# ------------------------------------------------------------------ Tavily

class Budget:
    """Hard cap on Tavily credits per rolling hour (basic search = 1 credit)."""

    def __init__(self, per_hour: int):
        self.per_hour = per_hour
        self.spent: list[tuple[float, float]] = []
        self.lock = threading.Lock()

    def reserve(self, credits: float = 1.0) -> bool:
        with self.lock:
            now = time.time()
            self.spent = [(t, c) for t, c in self.spent if now - t < 3600]
            if sum(c for _, c in self.spent) + credits > self.per_hour:
                return False
            self.spent.append((now, credits))
            return True


def tavily_search(query: str, api_key: str, max_results: int = 5, timeout: float = 15.0, live: bool = False) -> dict:
    body = {
        "query": query,
        "search_depth": "basic",
        "topic": "general",
        "max_results": max_results,
        "chunks_per_source": 3,
        "include_published_date": True,
        "include_answer": False,
        "include_raw_content": False,
        "auto_parameters": False,
        "include_usage": True,
    }
    if live:
        body["max_age_hours"] = 0
    req = urllib.request.Request(TAVILY_URL, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


# ------------------------------------------------------------------ evidence lines

@dataclass
class Line:
    n: int
    text: str
    source: int  # index into sources
    url: str
    title: str
    published: datetime | None


def parse_date(s: str | None) -> datetime | None:
    if not s:
        return None
    for f in (lambda x: datetime.fromisoformat(x.replace("Z", "+00:00")), parsedate_to_datetime):
        try:
            d = f(s)
            return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            pass
    return None


def evidence_lines(results: list[dict], max_lines: int = 60) -> list[Line]:
    lines: list[Line] = []
    for si, r in enumerate(results):
        content = r.get("content") or ""
        pub = parse_date(r.get("published_date"))
        for chunk in content.split("[...]"):
            for piece in re.split(r"(?<=[.!?])\s+|\n+", chunk):
                piece = piece.strip()
                if len(piece) >= 12:
                    lines.append(Line(len(lines) + 1, piece[:400], si, r.get("url", ""), r.get("title", ""), pub))
                if len(lines) >= max_lines:
                    return lines
    return lines


# ------------------------------------------------------------------ value matching

def norm_value(v: str) -> str:
    v = v.lower().replace("₹", "inr ").replace("rs.", "inr ").replace("rs ", "inr ").replace("per cent", "%").replace("percent", "%")
    v = re.sub(r"(?<=\d),(?=\d)", "", v)
    v = re.sub(r"\s+", "", v)
    return v


def numbers(v: str) -> list[str]:
    return [n.replace(",", "") for n in re.findall(r"\d[\d,]*(?:\.\d+)?", v)]


def value_in(value: str, text: str) -> bool:
    """Is this claimed value present in the text? Numbers are compared as numbers
    (1,299 == 1299); names case-insensitively as whole words."""
    nums = numbers(value)
    if nums:
        text_nums = set(numbers(text))
        return all(n in text_nums for n in nums)
    return re.search(r"\b" + re.escape(value.lower()) + r"\b", text.lower()) is not None


def subject_words(claim: Claim) -> list[str]:
    q = search_query(claim)
    return [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9+#.-]{2,}", q) if w.lower() != "latest"]


def relevant(line: Line, subj: list[str]) -> bool:
    if not subj:
        return True
    low = line.text.lower()
    hits = sum(1 for w in subj if w in low)
    return hits >= min(2, len(subj))


def same_kind_values(claim_value: str, text: str) -> list[str]:
    """Values in text of the same kind as claim_value (money->money, version->version, percent->percent)."""
    from claims import _MONEY, _PERCENT, _VERSION
    for rx in (_MONEY, _PERCENT, _VERSION):
        if rx.fullmatch(claim_value.strip()):
            return [m.group(0) for m in rx.finditer(text)]
    return []


# ------------------------------------------------------------------ deciding

@dataclass
class Check:
    verdict: str = "not-enough-info"  # supported | contradicted | not-enough-info | not-checked
    score: int = 50
    label: str = "unverified"         # fresh | unverified | stale
    reason: str = ""
    newer_value: str = ""
    evidence: list[dict] = field(default_factory=list)
    query: str = ""
    credits: float = 0.0
    request_id: str = ""
    urgent: bool = False              # fast-changing fact (review deadline <= 1 h)


# A line that mentions the claimed value but says it is old, replaced or compared
# against something newer does not support the claim.
STALE_CUES = re.compile(
    r"\b(?:former(?:ly)?|ex-|replac(?:e|ed|es|ing)|succeed(?:ed|s)?|successor|stepped down|steps down|resign(?:ed|s)?|"
    r"ousted|left the company|no longer|previous(?:ly)?|used to|until|older|outdated|deprecated|end[- ]of[- ]life|eol|"
    r"than the latest|instead of|was the|were the|had been|predecessor|retired)\b", re.I)
LATEST_CUES = re.compile(r"\b(?:latest|newest|current|stable|released|release|now available|most recent)\b", re.I)


def version_tuple(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", v))


def rules_decide(claim: Claim, lines: list[Line]) -> tuple[str, list[Line], str]:
    """No model. Conservative on purpose:
    - supported: a relevant line holds every claimed value and does not call it old/replaced
      (and, for versions, no relevant line names a newer version);
    - contradicted: (a) 2+ sources agree on a different value of the same kind, (b) a newer
      version appears next to latest/release wording, or (c) a line says the claimed person
      or value is former/replaced;
    - otherwise not-enough-info."""
    subj = subject_words(claim)
    rel = [l for l in lines if relevant(l, subj)]
    key_vals = [v for v in claim.values if numbers(v) or claim.category == "roles"]
    if not key_vals:
        return "not-enough-info", [], "no checkable value in claim"
    mentions = [l for l in rel if all(value_in(v, l.text) for v in key_vals)]

    # (b) newer version
    if claim.category == "software":
        from claims import _VERSION
        claimed = [version_tuple(v) for v in key_vals if _VERSION.fullmatch(v)]
        if claimed:
            c = max(claimed)
            newer = [(version_tuple(m.group(0)), m.group(0), l) for l in rel if LATEST_CUES.search(l.text)
                     for m in _VERSION.finditer(l.text)
                     if version_tuple(m.group(0))[:1] == c[:1] and version_tuple(m.group(0)) > c]
            if newer:
                best = max(newer, key=lambda x: x[0])
                return "contradicted", [x[2] for x in newer if x[1] == best[1]][:3], f"sources mention newer version {best[1]}"

    # (c) the claimed value is described as old / replaced
    stale = [l for l in mentions if STALE_CUES.search(l.text)]
    if stale and claim.category in ("roles", "software", "policy"):
        return "contradicted", stale[:3], "a source line describes the claimed value as former or replaced"

    support = [l for l in mentions if not STALE_CUES.search(l.text)]
    if claim.category == "roles":
        role = re.compile(r"\b(?:ceo|chief executive|president|prime minister|chair\w*|director|head|leader|minister|governor|captain|coach|founder|owner)\b", re.I)
        support = [l for l in support if role.search(l.text)]
    if support:
        return "supported", support[:3], "claimed value found in a relevant source line"

    # (a) agreement on a different value
    others: dict[str, set[int]] = {}
    lines_for: dict[str, list[Line]] = {}
    for v in key_vals:
        for l in rel:
            for o in same_kind_values(v, l.text):
                if norm_value(o) != norm_value(v):
                    others.setdefault(norm_value(o), set()).add(l.source)
                    lines_for.setdefault(norm_value(o), []).append(l)
    best = max(others.items(), key=lambda kv: len(kv[1]), default=None)
    if best and len(best[1]) >= 2:
        return "contradicted", lines_for[best[0]][:3], f"sources agree on {best[0]}"
    return "not-enough-info", [], "no relevant line that clearly supports or contradicts"


def judge_decide(claim: Claim, lines: list[Line], question: str, model: str, ollama_url: str, timeout: float = 60.0):
    """Local model points at line numbers; code verifies the cited lines."""
    numbered = "\n".join(f"[{l.n}] (source {l.source + 1}, {l.published.date() if l.published else 'undated'}) {l.text}" for l in lines)
    prompt = (
        "You check whether a claim from an AI answer is still true, using ONLY the numbered source lines.\n"
        f"Question the answer was for: {question or '(not given)'}\n"
        f"Claim: {claim.text}\n\nSource lines:\n{numbered}\n\n"
        "Reply with JSON: verdict = supported (a line states the same fact), contradicted (a line states a "
        "different, newer value for the same thing), or not_enough_info. Give the line numbers you rely on, and "
        "for contradicted, newer_value copied exactly from that line."
    )
    schema = {"type": "object", "properties": {
        "verdict": {"type": "string", "enum": ["supported", "contradicted", "not_enough_info"]},
        "lines": {"type": "array", "items": {"type": "integer"}},
        "newer_value": {"type": "string"}}, "required": ["verdict", "lines"]}
    body = {"model": model, "stream": False, "format": schema, "options": {"temperature": 0, "num_ctx": int(os.environ.get("FRESHNESS_NUM_CTX", "16384"))},
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(ollama_url.rstrip("/") + "/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(json.load(r)["message"]["content"])
    by_n = {l.n: l for l in lines}
    cited = [by_n[n] for n in out.get("lines", []) if n in by_n]
    verdict = out.get("verdict", "not_enough_info").replace("_", "-")
    key_vals = [v for v in claim.values if numbers(v) or claim.category == "roles"]
    # Code checks: the model never decides on its own.
    if verdict == "supported":
        ok = [l for l in cited if key_vals and all(value_in(v, l.text) for v in key_vals)]
        if not ok:
            return "not-enough-info", [], "model said supported but cited lines lack the claimed value", ""
        return "supported", ok, "cited line contains the claimed value", ""
    if verdict == "contradicted":
        newer = (out.get("newer_value") or "").strip()
        ok = [l for l in cited if newer and value_in(newer, l.text) and not all(value_in(v, l.text) for v in key_vals or [""])]
        if not ok or (key_vals and any(norm_value(newer) == norm_value(v) for v in key_vals)):
            return "not-enough-info", [], "model said contradicted but the newer value is not in the cited line", ""
        return "contradicted", ok, "cited line states a different value", newer
    return "not-enough-info", [], "model found no deciding line", ""


def score(verdict: str, claim: Claim, evidence: list[Line], now: datetime) -> tuple[int, str]:
    """Freshness score 0-100 (documented formula):
    supported      75 + 25 x recency of the newest supporting source (<=30 days: 1, 365+ days: 0, undated: 0.5)
    not-enough-info  35 for fast-changing facts (deadline <= 1 h), else 50
    contradicted   10, or 5 when contradicting lines come from 2+ sources
    The label follows the verdict, not the score: supported = fresh, contradicted =
    stale, anything else = unverified. "Could not confirm" is never "stale": a failed
    search is not evidence that a claim changed. The score is a heuristic for ordering
    claims, not a probability."""
    if verdict == "supported":
        dates = [l.published for l in evidence if l.published]
        if dates:
            age = (now - max(dates)).days
            r = 1.0 if age <= 30 else max(0.0, 1 - (age - 30) / 335)
        else:
            r = 0.5
        s = round(75 + 25 * r)
    elif verdict == "contradicted":
        s = 5 if len({l.source for l in evidence}) >= 2 else 10
    else:
        s = 35 if claim.ttl_seconds and claim.ttl_seconds <= 3600 else 50
    label = {"supported": "fresh", "contradicted": "stale"}.get(verdict, "unverified")
    return s, label


@dataclass
class Config:
    api_key: str = ""
    judge_model: str = ""          # e.g. "qwen2.5:7b"; empty = rules only
    ollama_url: str = "http://127.0.0.1:11434"
    max_claims: int = 5
    max_results: int = 5
    live: bool = False             # ask Tavily for live fetches (max_age_hours=0)
    credits_per_hour: int = 100
    question_search: bool = True   # one extra search on the question when a fast-changing claim is unconfirmed

    @classmethod
    def from_env(cls) -> "Config":
        return cls(api_key=os.environ.get("TAVILY_API_KEY", ""),
                   judge_model=os.environ.get("FRESHNESS_JUDGE_MODEL", ""),
                   ollama_url=os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"),
                   max_claims=int(os.environ.get("FRESHNESS_MAX_CLAIMS", "5")),
                   live=os.environ.get("FRESHNESS_LIVE", "") == "1",
                   credits_per_hour=int(os.environ.get("FRESHNESS_CREDITS_PER_HOUR", "100")),
                   question_search=os.environ.get("FRESHNESS_QUESTION_SEARCH", "1") != "0")


def check_claim(claim: Claim, question: str, cfg: Config, budget: Budget, search=tavily_search, now: datetime | None = None) -> Check:
    now = now or datetime.now(timezone.utc)
    c = Check(query=search_query(claim, question), urgent=bool(claim.ttl_seconds and claim.ttl_seconds <= 3600))
    if not cfg.api_key:
        c.verdict, c.reason, c.score, c.label = "not-checked", "no TAVILY_API_KEY", 0, "unverified"
        return c
    if not budget.reserve(1.0):
        c.verdict, c.reason, c.score, c.label = "not-checked", "hourly credit cap reached", 0, "unverified"
        return c
    try:
        res = search(c.query, cfg.api_key, max_results=cfg.max_results, live=cfg.live)
    except (urllib.error.URLError, TimeoutError, ValueError) as e:
        c.verdict, c.reason, c.score, c.label = "not-checked", f"search failed: {e}", 0, "unverified"
        return c
    c.request_id = res.get("request_id", "")
    c.credits = float((res.get("usage") or {}).get("credits", 1))
    lines = evidence_lines(res.get("results", []))
    if not lines:
        verdict, ev, why, newer = "not-enough-info", [], "search returned no usable text", ""
    elif cfg.judge_model:
        try:
            verdict, ev, why, newer = judge_decide(claim, lines, question, cfg.judge_model, cfg.ollama_url)
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as e:
            verdict, ev, why = rules_decide(claim, lines)
            why, newer = f"{why} (model unavailable: {e})", ""
    else:
        verdict, ev, why = rules_decide(claim, lines)
        newer = ""
        if verdict == "contradicted":
            for prefix in ("sources agree on ", "sources mention newer version "):
                if why.startswith(prefix):
                    newer = why[len(prefix):]
    c.verdict, c.reason, c.newer_value = verdict, why, newer
    c.score, c.label = score(verdict, claim, ev, now)
    c.evidence = [{"line": l.text, "url": l.url, "title": l.title,
                   "published": l.published.isoformat() if l.published else None} for l in ev]
    return c


_QWORDS_STOP = {"what", "who", "whom", "which", "when", "where", "how", "many", "much", "the", "is", "are", "was",
                "were", "does", "did", "has", "have", "of", "in", "on", "for", "and", "to", "a", "an", "this", "that",
                "most", "recent", "latest", "current", "currently", "now", "today"}


def question_evidence(question: str, cfg: Config, budget: Budget, search=tavily_search, max_lines: int = 6) -> tuple[list[dict], float, str]:
    """One search on the question itself, used when a fast-changing claim could be
    neither confirmed nor contradicted. Returns source lines only (no verdict): the
    regeneration may use them, and must say so when they don't state the answer."""
    if not question.strip():
        return [], 0.0, "no question"
    if not cfg.api_key:
        return [], 0.0, "no TAVILY_API_KEY"
    if not budget.reserve(1.0):
        return [], 0.0, "hourly credit cap reached"
    try:
        res = search(question.strip()[:380], cfg.api_key, max_results=cfg.max_results, live=cfg.live)
    except (urllib.error.URLError, TimeoutError, ValueError) as e:
        return [], 0.0, f"search failed: {e}"
    credits = float((res.get("usage") or {}).get("credits", 1))
    words = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9+#.'-]{2,}", question) if w.lower() not in _QWORDS_STOP]
    per_source: dict[int, int] = {}
    out = []
    for l in evidence_lines(res.get("results", [])):
        low = l.text.lower()
        if words and sum(1 for w in words if w in low) < min(2, len(words)):
            continue
        if per_source.get(l.source, 0) >= 2:
            continue
        per_source[l.source] = per_source.get(l.source, 0) + 1
        out.append({"line": l.text, "url": l.url, "title": l.title,
                    "published": l.published.isoformat() if l.published else None})
        if len(out) >= max_lines:
            break
    return out, credits, "" if out else "no relevant line"
