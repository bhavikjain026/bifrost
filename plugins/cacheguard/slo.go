package main

// Stale-rate target (best_before.max_stale_rate).
//
// Instead of a fixed review deadline per category, the operator sets one number: the
// share of cache hits allowed to serve a stale answer (for example 0.02 = 2%). The
// plugin turns it into a review interval per category from how often answers in that
// category actually change, and spreads the check budget where it buys the most:
//
//   interval_c = k / sqrt(rate_c)      (k solved so the expected stale share = target)
//
// With changes as a Poisson process (rate λ per day) and a check every Δ days that
// catches a change with probability r, the expected stale share of an entry is about
// f(λ·Δ·(2/r − 1)), f(x) = 1 − (1 − e^−x)/x. The expected overall stale share is the
// mix-weighted sum over categories.
//
// Answers the detector does not flag are reviewed too (group "not-flagged"): on real
// data most fast-changing questions match no rule, and a fixed-TTL design never
// re-checks them.
//
// Default rates were learned from two years of FreshQA weekly releases (Feb 2024 to
// Jan 2025 for learning; tested on Feb 2025 to Apr 2026 in cacheguard-replay/freshqa/
// timetravel). Weekly data cannot measure sub-week change, so the real-time categories
// (markets, weather_ops, live_results) keep their short fixed deadlines.
// Operators can supply their own rates (change_rates_file) learned from their traffic.

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"time"

	"github.com/maximhq/bifrost/plugins/cacheguard/bestbefore"
)

const notFlaggedGroup = "not-flagged"

// sloRates is the change_rates_file format (what simulate.py --export-rates writes).
type sloRates struct {
	Source         string             `json:"source"`
	ChangesPerYear map[string]float64 `json:"changes_per_year"`
	Share          map[string]float64 `json:"share"`
}

// defaultSLORates: learned from FreshQA releases 2024-02-26..2025-02-01, grouped by
// this plugin's own detector (simulate.py --groups detect.json --export-rates).
var defaultSLORates = sloRates{
	Source: "FreshQA weekly releases, learned 2024-02-26..2025-02-01",
	ChangesPerYear: map[string]float64{
		"live_results": 5.7087, "markets": 5.6195, notFlaggedGroup: 1.0372, "open_year": 0.0,
		"pricing": 0.6116, "recency": 2.1257, "roles": 0.2817, "software": 1.4272,
	},
	Share: map[string]float64{
		"live_results": 0.0069, "markets": 0.0093, notFlaggedGroup: 0.7477, "open_year": 0.0046,
		"pricing": 0.0162, "recency": 0.1644, "roles": 0.044, "software": 0.0069,
	},
}

// Real-time categories: change faster than the data can show; keep fixed deadlines.
var sloFixed = map[string]bool{
	bestbefore.CatMarkets: true, bestbefore.CatWeatherOps: true, bestbefore.CatLiveResults: true,
}

const (
	sloMinInterval = 24 * time.Hour       // the learning data has about weekly resolution
	sloMaxInterval = 365 * 24 * time.Hour // never-changing groups still get a yearly look
)

// sloIntervals is computed once at Init; nil when max_stale_rate is off.
var sloIntervals map[string]time.Duration

func staleShare(x float64) float64 {
	if x <= 0 {
		return 0
	}
	return 1 - (1-math.Exp(-x))/x
}

// computeSLOIntervals solves for k (bisection on a log scale) and returns the review
// interval per managed category. Categories with no learned change get the maximum.
func computeSLOIntervals(rates sloRates, target, recall float64) map[string]time.Duration {
	m := 2/recall - 1
	minD, maxD := sloMinInterval.Hours()/24, sloMaxInterval.Hours()/24
	clamp := func(v float64) float64 { return math.Min(maxD, math.Max(minD, v)) }
	var cats []string
	for c := range rates.ChangesPerYear {
		if !sloFixed[c] {
			cats = append(cats, c)
		}
	}
	sort.Strings(cats)
	expStale := func(k float64) float64 {
		s := 0.0
		for _, c := range cats {
			lam := rates.ChangesPerYear[c] / 365
			if lam > 0 {
				s += rates.Share[c] * staleShare(lam*m*clamp(k/math.Sqrt(lam)))
			}
		}
		return s
	}
	lo, hi := 1e-4, 1e4
	for i := 0; i < 100; i++ {
		mid := math.Sqrt(lo * hi)
		if expStale(mid) <= target {
			lo = mid
		} else {
			hi = mid
		}
	}
	out := make(map[string]time.Duration, len(cats))
	for _, c := range cats {
		days := maxD
		if lam := rates.ChangesPerYear[c] / 365; lam > 0 {
			days = clamp(lo / math.Sqrt(lam))
		}
		out[c] = time.Duration(days * float64(24*time.Hour))
	}
	return out
}

func initSLO() error {
	bb := &cfg.BestBefore
	sloIntervals = nil
	if bb.MaxStaleRate == 0 {
		return nil
	}
	if bb.MaxStaleRate < 0 || bb.MaxStaleRate >= 0.5 {
		return fmt.Errorf("cacheguard: best_before.max_stale_rate must be between 0 and 0.5 (e.g. 0.02 = 2%%), got %v", bb.MaxStaleRate)
	}
	if bb.CheckerRecall == 0 {
		bb.CheckerRecall = 1
	}
	if bb.CheckerRecall < 0.05 || bb.CheckerRecall > 1 {
		return fmt.Errorf("cacheguard: best_before.checker_recall must be between 0.05 and 1, got %v", bb.CheckerRecall)
	}
	rates := defaultSLORates
	if bb.ChangeRatesFile != "" {
		b, err := os.ReadFile(bb.ChangeRatesFile)
		if err != nil {
			return fmt.Errorf("cacheguard: best_before.change_rates_file: %w", err)
		}
		var r sloRates
		if err := json.Unmarshal(b, &r); err != nil || len(r.ChangesPerYear) == 0 {
			return fmt.Errorf("cacheguard: best_before.change_rates_file %s: need changes_per_year and share (%v)", bb.ChangeRatesFile, err)
		}
		for c, v := range r.ChangesPerYear {
			if v < 0 || math.IsNaN(v) || math.IsInf(v, 0) {
				return fmt.Errorf("cacheguard: best_before.change_rates_file: bad rate for %q", c)
			}
		}
		if _, ok := r.ChangesPerYear[notFlaggedGroup]; !ok {
			return fmt.Errorf("cacheguard: best_before.change_rates_file: needs a %q rate", notFlaggedGroup)
		}
		rates = r
	}
	sloIntervals = computeSLOIntervals(rates, bb.MaxStaleRate, bb.CheckerRecall)
	dbg("max_stale_rate %.3g (recall %.2g, rates: %s): review intervals %v", bb.MaxStaleRate, bb.CheckerRecall, rates.Source, sloIntervals)
	return nil
}

// sloInterval returns the review interval for a category (or the not-flagged group)
// when max_stale_rate is on. An explicit ttl_seconds override always wins.
func sloInterval(category string) (time.Duration, bool) {
	if sloIntervals == nil {
		return 0, false
	}
	if _, overridden := cfg.BestBefore.TTLSeconds[category]; overridden {
		return 0, false
	}
	iv, ok := sloIntervals[category]
	return iv, ok
}
