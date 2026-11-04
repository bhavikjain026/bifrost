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
  This puts checks where they buy the most freshness.
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
| `max_stale_rate: 0.02`, `checker_recall: 0.6` | 1.2% | 178 |

The target held on the test period at every level tried (0.5%, 1%, 2%, 5%), including
with a checker that catches only 60% of changes.

## Limits
- Rates are learned from FreshQA's English trivia-style questions. For real traffic,
  learn your own and use `change_rates_file`.
- Time resolution is about a week. The model assumes changes arrive randomly at a steady
  rate, so scheduled events such as yearly awards are only roughly modelled.
- The target is an expectation over the traffic mix, not a per-answer guarantee.
- Not-flagged answers are reviewed by the claim checker. If it finds no time-sensitive
  claim, the review costs no search credit.
