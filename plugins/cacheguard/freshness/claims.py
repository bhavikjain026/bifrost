"""Split an LLM answer into claims and flag the ones likely to go stale.

Plain rules only: every flag says which rule fired, so an operator can see why.
English only (other scripts are reported as unsupported, not guessed).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# How fast each kind of fact changes (seconds) -> used for the review deadline and the score.
CATEGORY_TTL = {
    "markets": 300,
    "live_results": 300,
    "weather_ops": 900,
    "recency": 3600,
    "pricing": 86400,
    "software": 86400,
    "roles": 86400,
    "policy": 86400,
    "statistics": 86400,
}

_ROLE = (r"ceo|cto|cfo|coo|chief executive|president|prime minister|chief minister|governor|chair(?:man|woman|person)?|"
         r"founder|director|captain|coach|owner|head|leader|minister|mayor|speaker|vice[- ]president|vice[- ]chancellor|dean|ambassador")

_MONEY = re.compile(
    r"(?:[$€£₹¥]\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:k|m|bn|million|billion|crore|lakh))?"
    r"|\b(?:usd|inr|eur|gbp|rs\.?)\s?\d[\d,]*(?:\.\d+)?"
    r"|\b\d[\d,]*(?:\.\d+)?\s?(?:dollars|rupees|euros|pounds|usd|inr|eur)\b)", re.I)
_PERCENT = re.compile(r"\b\d+(?:\.\d+)?\s?(?:%|percent\b|per cent\b)", re.I)
_VERSION = re.compile(r"\b[vV]?\d+\.\d+(?:\.\d+)*\b")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_NAME = re.compile(r"\b[A-Z][a-z]+(?:[ -][A-Z][a-z]+)+\b")

RULES = [
    # (rule id, category, pattern, extra condition or None)
    ("market", "markets", re.compile(r"\b(?:stock|share) price|exchange rate|market cap|bitcoin|ethereum|\bbtc\b|sensex|nifty|nasdaq|gold (?:rate|price)", re.I), None),
    ("live-results", "live_results", re.compile(r"\b(?:is|are) (?:currently )?(?:leading|ahead|top of the table)|\bstandings\b|\bleaderboard\b|\bvote count\b", re.I), None),
    ("weather", "weather_ops", re.compile(r"\b(?:weather|forecast|temperature|aqi|flight (?:is|was) delayed|outage)\b", re.I), None),
    ("price", "pricing", _MONEY, re.compile(r"\b(?:cost|costs|price|priced|pricing|fee|fees|plan|subscription|per month|/month|a month|per year|/year|annually|charges?|fare|tariff|pay)\b", re.I)),
    ("version", "software", re.compile(r"\b(?:latest|newest|current|stable|most recent) (?:\w+ ){0,3}(?:version|release|model|update)\b|\b(?:version|release|v)\s?\d+(?:\.\d+)+", re.I), None),
    ("version-number", "software", _VERSION, re.compile(r"\b(?:version|release|released|latest|api|sdk|python|node|go|java|ios|android|macos|windows|kernel|model)\b", re.I)),
    ("role", "roles", re.compile(rf"\b(?:is|became|serves as|has been|was appointed|was named|took over as)(?: the)?(?: current| new)? (?:{_ROLE})\b|\b(?:{_ROLE}) (?:is|of \w+(?: \w+)? is)\b|\bled by\b|\bheaded by\b", re.I), None),
    ("policy", "policy", re.compile(r"\b(?:law|act|regulation|regulations|policy|rule|rules|tax rate|gst|visa|ban|banned|illegal|legal|mandatory|required|eligib\w*|deadline|compliance)\b", re.I),
     re.compile(r"\b(?:is|are|requires?|must|allowed|permitted|prohibited|currently|now|since|effective|from)\b", re.I)),
    ("statistic", "statistics", re.compile(r"\b(?:inflation|cpi|gdp|repo rate|interest rate|unemployment|population|growth rate)\b", re.I), re.compile(r"\d")),
    ("recency", "recency", re.compile(r"\b(?:currently|as of|at present|right now|latest|newest|most recent|recently|this year|this month|today|nowadays|is now|are now)\b", re.I), None),
]

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|\n+")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


@dataclass
class Claim:
    text: str
    index: int
    flagged: bool = False
    category: str = ""
    ttl_seconds: int = 0
    rules: list[str] = field(default_factory=list)
    values: list[str] = field(default_factory=list)  # the parts that would go stale (amounts, versions, names...)


def split_sentences(answer: str) -> list[str]:
    out = []
    for part in _SENT.split(answer.strip()):
        part = _BULLET.sub("", part).strip().strip("*").strip()
        if len(part) >= 3 and re.search(r"[A-Za-z]", part):
            out.append(part)
    return out


def latin_share(s: str) -> float:
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 1.0
    return sum(1 for c in letters if c.isascii() or "À" <= c <= "ɏ") / len(letters)


def values_in(text: str) -> list[str]:
    vals: list[str] = []
    for rx in (_MONEY, _PERCENT, _VERSION):
        vals += [m.group(0).strip() for m in rx.finditer(text)]
    taken = " ".join(vals)
    vals += [m.group(0) for m in _NAME.finditer(text) if m.group(0) not in taken]
    vals += [m.group(0) for m in _YEAR.finditer(text) if m.group(0) not in taken]
    seen, out = set(), []
    for v in vals:
        if v.lower() not in seen:
            seen.add(v.lower())
            out.append(v)
    return out


def flag(answer: str) -> list[Claim]:
    claims = []
    for i, s in enumerate(split_sentences(answer)):
        c = Claim(text=s, index=i)
        if latin_share(s) < 0.7:
            c.rules = ["unsupported-language"]
            claims.append(c)
            continue
        for rule_id, cat, rx, extra in RULES:
            if rx.search(s) and (extra is None or extra.search(s)):
                c.rules.append(rule_id)
                ttl = CATEGORY_TTL[cat]
                if not c.flagged or ttl < c.ttl_seconds or (c.category == "recency" and cat != "recency"):
                    if not (c.flagged and cat == "recency" and c.category != "recency"):
                        c.category, c.ttl_seconds = cat, ttl
                c.flagged = True
        if c.flagged:
            c.values = values_in(s)
        claims.append(c)
    return claims


_STOP = set("""a an the is are was were be been being of in on at to for from by with and or as that this these those it its
it's which who whom whose what when where how than then there their they them you your we our i me my he she his her
currently now latest newest recent recently today current present also about around approximately roughly just only per""".split())


def search_query(claim: Claim, question: str = "") -> str:
    """Query for re-verification: the claim with its stale-prone values removed, so the
    search is not steered towards old pages that repeat the old value."""
    text = claim.text
    for v in sorted(claim.values, key=len, reverse=True):
        if _NAME.fullmatch(v) and claim.category != "roles":
            continue  # keep subject names (product, company) except for "who holds the role" claims
        text = text.replace(v, " ")
    words = [w for w in re.findall(r"[A-Za-z0-9+#.'-]+", text) if w.lower() not in _STOP and not _YEAR.fullmatch(w)]
    q = " ".join(words)
    if question:
        q = question.strip() + " " + q
    q = (q + " latest").strip()
    return q[:380]
