package main

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/maximhq/bifrost/plugins/cacheguard/bestbefore"
)

func days(d time.Duration) float64 { return d.Hours() / 24 }

// Same numbers as cacheguard-replay/freshqa/timetravel/simulate.py (plugin-slo policy).
func TestSLOIntervalsMatchSimulation(t *testing.T) {
	for _, tc := range []struct {
		target, recall float64
		want           map[string]float64
	}{
		{0.02, 1, map[string]float64{"not-flagged": 13.91, "open_year": 365, "pricing": 18.11, "recency": 9.72, "roles": 26.69, "software": 11.86}},
		{0.02, 0.6, map[string]float64{"not-flagged": 5.96, "pricing": 7.76, "recency": 4.16, "roles": 11.44, "software": 5.08}},
		{0.01, 1, map[string]float64{"not-flagged": 6.91, "pricing": 8.99, "recency": 4.82, "roles": 13.25, "software": 5.89}},
	} {
		got := computeSLOIntervals(defaultSLORates, tc.target, tc.recall)
		for c, w := range tc.want {
			if math.Abs(days(got[c])-w) > 0.02 {
				t.Errorf("target %v recall %v %s: got %.2f days, want %.2f", tc.target, tc.recall, c, days(got[c]), w)
			}
		}
		for c := range sloFixed {
			if _, ok := got[c]; ok {
				t.Errorf("real-time category %s must keep its fixed deadline", c)
			}
		}
	}
}

func TestSLOMeetsTargetAndOrdersByChangeRate(t *testing.T) {
	iv := computeSLOIntervals(defaultSLORates, 0.02, 1)
	exp := 0.0
	for c, d := range iv {
		lam := defaultSLORates.ChangesPerYear[c] / 365
		exp += defaultSLORates.Share[c] * staleShare(lam*days(d))
	}
	if math.Abs(exp-0.02) > 0.0005 {
		t.Errorf("expected stale share %.4f, want 0.02", exp)
	}
	if !(iv["recency"] < iv["not-flagged"] && iv["not-flagged"] < iv["pricing"] && iv["pricing"] < iv["roles"]) {
		t.Errorf("faster-changing groups must be reviewed more often: %v", iv)
	}
	tight, weakChecker := computeSLOIntervals(defaultSLORates, 0.005, 1), computeSLOIntervals(defaultSLORates, 0.02, 0.5)
	for _, c := range []string{"recency", "not-flagged", "roles"} {
		if !(tight[c] < iv[c] && weakChecker[c] < iv[c]) {
			t.Errorf("%s: a tighter target or a weaker checker must shorten the interval (%v %v %v)", c, tight[c], weakChecker[c], iv[c])
		}
		if iv[c] < sloMinInterval || iv[c] > sloMaxInterval {
			t.Errorf("%s: interval %v outside [%v, %v]", c, iv[c], sloMinInterval, sloMaxInterval)
		}
	}
}

func TestSLOReviewsUnflaggedAnswers(t *testing.T) {
	var calls atomic.Int32
	srv := fakeChecker(t, false, &calls)
	defer srv.Close()
	q := "How do I bake bread with yeast?"
	if r := bestbefore.Detect(q, time.Now()); r.Status != bestbefore.NotFlagged {
		t.Fatalf("test question must not be flagged: %+v", r)
	}
	initBB(t, srv.URL, map[string]any{"max_stale_rate": 0.02})
	iv := sloIntervals[notFlaggedGroup]

	store.put("s1", storedPrompt{prompt: q, cachedAt: time.Now().Add(-iv / 2)})
	d := hit(t, q, "s1", "semantic")
	if d.Freshness != "within-deadline" || calls.Load() != 0 {
		t.Fatalf("young unflagged entry must be served without a check: %+v", d)
	}
	if h := d.headers()[hdrPrefix+"review-every"]; !strings.Contains(h, "max-stale-rate") {
		t.Errorf("review-every header: %q", h)
	}
	store.put("s2", storedPrompt{prompt: q, cachedAt: time.Now().Add(-iv - time.Hour)})
	if d := hit(t, q, "s2", "semantic"); d.Freshness != "rechecked-fresh" || calls.Load() != 1 {
		t.Fatalf("unflagged entry past its interval must be re-checked: %+v (calls %d)", d, calls.Load())
	}
	// Off by default: the same old entry is never reviewed.
	initBB(t, srv.URL, nil)
	store.put("s3", storedPrompt{prompt: q, cachedAt: time.Now().Add(-1000 * time.Hour)})
	if d := hit(t, q, "s3", "semantic"); d.Freshness != "not-time-sensitive" || calls.Load() != 1 {
		t.Fatalf("without max_stale_rate unflagged answers are not reviewed: %+v", d)
	}
}

func TestSLOIntervalForFlaggedCategoriesAndOverrides(t *testing.T) {
	q := "Who won the most recent Super Bowl?"
	r := bestbefore.Detect(q, time.Now())
	if _, learned := defaultSLORates.ChangesPerYear[r.Category]; r.Status != bestbefore.Flagged || sloFixed[r.Category] || !learned {
		t.Fatalf("need a flagged question in a learned, non-real-time category: %+v", r)
	}
	initBB(t, "http://127.0.0.1:1/check", map[string]any{"max_stale_rate": 0.02})
	now := time.Now()
	_, ttl, ok := reviewDeadline(q, storedPrompt{cachedAt: now}, true)
	if !ok || ttl != sloIntervals[r.Category] || ttl == bestbefore.DefaultTTL[r.Category] {
		t.Fatalf("%s: got ttl %v ok %v, want the SLO interval %v", r.Category, ttl, ok, sloIntervals[r.Category])
	}
	// An explicit ttl_seconds override for the category wins.
	initBB(t, "http://127.0.0.1:1/check", map[string]any{"max_stale_rate": 0.02, "ttl_seconds": map[string]int{r.Category: 600}})
	if _, ttl, _ := reviewDeadline(q, storedPrompt{cachedAt: now}, true); ttl != 10*time.Minute {
		t.Errorf("override must win, got %v", ttl)
	}
	// A flagged category with no learned rate keeps its fixed deadline.
	if rr := bestbefore.Detect(repoQ, now); rr.Status == bestbefore.Flagged {
		if _, learned := defaultSLORates.ChangesPerYear[rr.Category]; !learned {
			initBB(t, "http://127.0.0.1:1/check", map[string]any{"max_stale_rate": 0.02})
			if _, ttl, _ := reviewDeadline(repoQ, storedPrompt{cachedAt: now}, true); ttl != bestbefore.TTLFor(rr, nil) {
				t.Errorf("%s has no learned rate and must keep %v, got %v", rr.Category, bestbefore.TTLFor(rr, nil), ttl)
			}
		}
	}
	// Real-time categories keep their short fixed deadline.
	mq := "What is the current price of Bitcoin?"
	if mr := bestbefore.Detect(mq, now); mr.Status == bestbefore.Flagged && sloFixed[mr.Category] {
		if _, ttl, _ := reviewDeadline(mq, storedPrompt{cachedAt: now}, true); ttl != bestbefore.DefaultTTL[mr.Category] {
			t.Errorf("%s must keep %v, got %v", mr.Category, bestbefore.DefaultTTL[mr.Category], ttl)
		}
	}
}

func TestSLOConfigValidation(t *testing.T) {
	dir := t.TempDir()
	noGroup := filepath.Join(dir, "bad.json")
	_ = os.WriteFile(noGroup, []byte(`{"changes_per_year":{"recency":2},"share":{"recency":1}}`), 0o600)
	negative := filepath.Join(dir, "neg.json")
	_ = os.WriteFile(negative, []byte(`{"changes_per_year":{"not-flagged":-1},"share":{"not-flagged":1}}`), 0o600)
	for _, extra := range []map[string]any{
		{"max_stale_rate": 0.7},
		{"max_stale_rate": -0.1},
		{"max_stale_rate": 0.02, "checker_recall": 2},
		{"max_stale_rate": 0.02, "change_rates_file": filepath.Join(dir, "missing.json")},
		{"max_stale_rate": 0.02, "change_rates_file": noGroup},
		{"max_stale_rate": 0.02, "change_rates_file": negative},
	} {
		bb := map[string]any{"enabled": true}
		for k, v := range extra {
			bb[k] = v
		}
		if err := Init(map[string]any{"best_before": bb}); err == nil {
			t.Errorf("config %v must be rejected", extra)
		}
	}
	good := filepath.Join(dir, "rates.json")
	_ = os.WriteFile(good, []byte(`{"source":"own traffic","changes_per_year":{"not-flagged":1,"recency":6},"share":{"not-flagged":0.5,"recency":0.5}}`), 0o600)
	if err := Init(map[string]any{"best_before": map[string]any{"enabled": true, "max_stale_rate": 0.02, "change_rates_file": good}}); err != nil {
		t.Fatalf("valid rates file rejected: %v", err)
	}
	if !(sloIntervals["recency"] < sloIntervals[notFlaggedGroup]) {
		t.Errorf("rates file not used: %v", sloIntervals)
	}
	if err := Init(map[string]any{"best_before": map[string]any{"enabled": true}}); err != nil || sloIntervals != nil {
		t.Errorf("max_stale_rate 0 must turn the feature off (err %v, intervals %v)", err, sloIntervals)
	}
}

// A request holding an older copy of the entry must not start a second refresh after
// another one already finished.
func TestRefreshSkipsWhenAlreadyRefreshed(t *testing.T) {
	var calls atomic.Int32
	srv := fakeChecker(t, false, &calls)
	defer srv.Close()
	initBB(t, srv.URL, nil)
	old := time.Now().Add(-23 * time.Hour)
	store.put("rr", storedPrompt{prompt: repoQ, cachedAt: old, checkedAt: time.Now()})
	if startRefresh("rr", repoQ, "answer", nil, old) {
		t.Fatal("entry was re-checked after the caller's snapshot; no new refresh expected")
	}
	if !startRefresh("rr", repoQ, "answer", nil, time.Now().Add(time.Minute)) {
		t.Fatal("an up-to-date caller must be able to refresh")
	}
	waitBackground(t)
	if calls.Load() != 1 {
		t.Errorf("calls %d, want 1", calls.Load())
	}
}
