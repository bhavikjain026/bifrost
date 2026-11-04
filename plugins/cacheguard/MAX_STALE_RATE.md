# Stale-rate target (`best_before.max_stale_rate`)

Set one number instead of per-category review deadlines: the share of cache hits you
allow to serve an out-of-date answer.

```json
"best_before": {
  "enabled": true,
  "max_stale_rate": 0.02,
  "checker_recall": 1.0
}
```

| key | meaning | default |
|---|---|---|
| `max_stale_rate` | target share of hits served stale (0.02 = 2%). `0` = off: fixed deadlines per category | 0 |
| `checker_recall` | share of real changes the freshness checker catches; a weaker checker gets shorter intervals | 1.0 |
| `change_rates_file` | rates learned from your own traffic (same format as `simulate.py --export-rates`) | built-in |

## What it does
- Each category gets a review interval from how often its answers really change:
  `interval = k / √rate`, with `k` chosen so the expected stale share equals the target.
  (In theory this spreads checks where they buy the most freshness; on FreshQA it did not
  beat one well-chosen interval for everything, see below.)
- **Answers the detector does not flag are reviewed too** (group `not-flagged`). Without
  this, they are never re-checked.
- Real-time categories (`markets`, `weather_ops`, `live_results`) keep their short fixed
  deadlines: the learning data is weekly and can't measure faster change.
- An explicit `ttl_seconds` entry for a category always wins.
- A flagged category with no learned rate keeps its fixed deadline.
- Each hit reports the interval used, e.g.
  `x-bifrost-cacheguard-review-every: 333h47m0s (max-stale-rate)`.

## Where the default rates come from
They come from two years of FreshQA weekly releases, with each question grouped by this
plugin's own detector. Rates were learned on Feb 2024 to Jan 2025. The policy was then
replayed on Feb 2025 to Apr 2026 (444 days, never seen when learning). See
`cacheguard-replay/freshqa/timetravel/`.

| policy (one cached answer per question, asked daily) | stale answers served | checks per 1,000 requests |
|---|---|---|
| never re-check | 24.3% | 0 |
| fixed deadlines (the plugin without this setting) | 15.6% | 252 |
| `max_stale_rate: 0.02` | 1.5% | 88 |
| one interval for everything, best value picked afterwards (about every 14 days) | 1.5% | 70 |
| `max_stale_rate: 0.02`, `checker_recall: 0.6` (checker catches 60% of changes) | 1.2% | 178 |
| one interval for everything, best value picked afterwards, same 60% checker | 1.2% | 179 |

What this shows, and what it doesn't:
- The target held on the 444 unseen days at every level tried (0.5%, 1%, 2%, 5%), with a
  perfect checker and with one that catches only 60% of changes.
- It does **not** need fewer checks than one well-chosen interval for everything. With a
  perfect checker it used about 26% more than the best single interval (picked with
  hindsight); with the 60% checker, about the same. The gain is that you state the risk
  you accept and the intervals follow from learned change rates and checker recall: you
  don't have to know the right interval in advance, and a weaker checker automatically
  gets shorter intervals.
- One likely reason the per-category split doesn't help here: three quarters of the
  questions are not flagged by the detector, so they all share one group and one interval.
- Denominator: every simulated request (432 questions x 444 days = 191,808), all served
  from the cache. A review due that day happens before the answer is served.
- Full tables: `cacheguard-replay/freshqa/timetravel/sim_report_detector.md` and
  `sim_report_detector_recall60.md` (rows `plugin-slo@...`).

## Limits
- Rates are learned from FreshQA's English trivia-style questions. For real traffic,
  learn your own and use `change_rates_file`.
- Time resolution is about a week. The model assumes changes arrive randomly at a steady
  rate, so scheduled events such as yearly awards are only roughly modelled.
- The target is an expectation over the traffic mix, not a per-answer guarantee.
- Not-flagged answers are reviewed by the claim checker. If it finds no time-sensitive
  claim, the review costs no search credit.
