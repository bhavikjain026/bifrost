package bestbefore

import (
	"testing"
	"time"
)

var clock = time.Date(2026, 11, 3, 12, 0, 0, 0, time.UTC)

func TestDetect(t *testing.T) {
	cases := []struct {
		q    string
		want Status
		cat  string
	}{
		// time-sensitive
		{"What is the bitcoin price right now?", Flagged, CatMarkets},
		{"What's the USD to INR exchange rate today?", Flagged, CatMarkets},
		{"Is it going to rain in Delhi tomorrow?", Flagged, CatWeatherOps},
		{"Who is leading the medal table right now?", Flagged, CatMarkets + "|" + CatLiveResults},
		{"How much does ChatGPT Plus cost?", Flagged, CatPricing},
		{"What is the latest Python release?", Flagged, CatSoftware},
		{"Who is the CEO of Microsoft?", Flagged, CatRoles},
		{"Who is the current prime minister of the UK?", Flagged, CatRoles},
		{"Is a visa required for Indians to visit Thailand now?", Flagged, CatPolicy},
		{"What is the latest CPI inflation reading for India?", Flagged, CatStatistics},
		{"What happened in the news today?", Flagged, CatRecency},
		{"Which phones have launched in 2026 so far?", Flagged, CatOpenYear},
		{"Who was CEO in 2010, and who is CEO now?", Flagged, CatRoles},
		{"Explain price elasticity using today's oil price", Flagged, ""},
		// not time-sensitive
		{"What is Ohm's law?", NotFlagged, ""},
		{"How do I find the current in this circuit?", NotFlagged, ""},
		{"Explain stock price volatility", NotFlagged, ""},
		{"What does a CEO do?", NotFlagged, ""},
		{"Who was the CEO of Apple in 2010?", NotFlagged, ""},
		{"What changed in Python 3.10?", NotFlagged, ""},
		{"How do I bake bread without yeast?", NotFlagged, ""},
		{"Who won the 2019 Cricket World Cup final?", NotFlagged, ""},
		{"What is the capital of New Zealand?", NotFlagged, ""},
		{"What is the time complexity of quicksort?", NotFlagged, ""},
		{"Define opportunity cost", NotFlagged, ""},
		// out of scope
		{"आज सोने का भाव क्या है?", Unknown, ""},
	}
	for _, c := range cases {
		r := Detect(c.q, clock)
		if r.Status != c.want {
			t.Errorf("%q: status %s (%v), want %s", c.q, r.Status, r.Hits, c.want)
			continue
		}
		if c.cat != "" && c.cat != r.Category && !(len(c.cat) > len(r.Category) && contains(c.cat, r.Category)) {
			t.Errorf("%q: category %s, want %s", c.q, r.Category, c.cat)
		}
	}
}

func contains(alts, s string) bool {
	for _, a := range splitBar(alts) {
		if a == s {
			return true
		}
	}
	return false
}

func splitBar(s string) []string {
	var out []string
	start := 0
	for i := 0; i <= len(s); i++ {
		if i == len(s) || s[i] == '|' {
			out = append(out, s[start:i])
			start = i + 1
		}
	}
	return out
}

func TestShortestDeadlineWins(t *testing.T) {
	r := Detect("What is the bitcoin price today and who is the CEO of Coinbase?", clock)
	if r.TTL != 5*time.Minute {
		t.Errorf("ttl %s, want 5m", r.TTL)
	}
	if got := TTLFor(r, map[string]int{CatMarkets: 60}); got != time.Minute {
		t.Errorf("override ttl %s, want 1m", got)
	}
}

// The ten worked examples from the design review.
func TestCompareWorkedExamples(t *testing.T) {
	cases := []struct {
		before, after []string
		want          Verdict
		reason        string
	}{
		{[]string{"Plan costs ₹1,23,456 per year."}, []string{"Plan costs INR 123456 per year."}, Unchanged, ""},
		{[]string{"Plan costs INR 999/month."}, []string{"Plan costs INR 1099/month."}, Changed, "numbers-changed"},
		{[]string{"Limit: 10 MB per upload."}, []string{"Limit: 10 Mb per upload."}, Changed, "text-changed"},
		{[]string{"Exports are allowed for free accounts."}, []string{"Exports are not allowed for free accounts."}, Changed, "negation-changed"},
		{[]string{"Asha is CEO; Bala is CFO."}, []string{"Asha is CFO; Bala is CEO."}, Changed, "text-changed"},
		{[]string{"Offer ends 3 November 2026."}, []string{"Offer ends November 3, 2026."}, Unchanged, ""},
		{[]string{"Offer ends 03/04/2026."}, []string{"Offer ends April 3, 2026."}, Changed, "dates-changed"},
		{[]string{"Inflation was 5.5 per cent in September."}, []string{"Inflation was 5.5% in September."}, Unchanged, ""},
		{[]string{"A", "B"}, []string{"B", "A"}, Unchanged, ""},
		{[]string{"The repo rate is 5.5%."}, nil, Undecided, "empty"},
	}
	for i, c := range cases {
		got := Compare(c.before, c.after)
		if got.Verdict != c.want || (c.reason != "" && got.Reason != c.reason) {
			t.Errorf("example %d: got %s/%s (-%v +%v), want %s/%s", i+1, got.Verdict, got.Reason, got.Removed, got.Added, c.want, c.reason)
		}
	}
}

func TestNormalizeKeepsAmbiguity(t *testing.T) {
	for in, want := range map[string]string{
		"1,5 kg":           "1,5 kg",           // ambiguous decimal comma: kept
		"version 3.11":     "version 3.11",     // not a date, not regrouped
		"31 February 2026": "31 February 2026", // invalid date: kept
		"Sept 9, 2026":     "2026-09-09",
		"10,000 users":     "10000 users",
		"&nbsp;₹99&amp;up": "INR 99&up",
		"1,2,3 items":      "1,2,3 items",
		"Price:  ₹ 1,099 ": "Price: INR 1099",
	} {
		if got := Normalize(in); got != want {
			t.Errorf("Normalize(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestOverall(t *testing.T) {
	base := map[string][]string{
		"https://a": {"Plan INR 99/month."},
		"https://b": {"B unchanged."},
		"https://c": {"C unchanged."},
	}
	same := []SourceObservation{{URL: "https://a", Chunks: []string{"Plan INR 99/month."}}, {URL: "https://b", Chunks: []string{"B unchanged."}}, {URL: "https://c", Chunks: []string{"C unchanged."}}}
	if v, _, n := Overall(base, same); v != Unchanged || n != 3 {
		t.Errorf("all same: %s %d", v, n)
	}
	oneChanged := append([]SourceObservation(nil), same...)
	oneChanged[0] = SourceObservation{URL: "https://a", Chunks: []string{"Plan INR 199/month."}}
	if v, r, _ := Overall(base, oneChanged); v != Changed || r != "numbers-changed" {
		t.Errorf("one of three changed must invalidate: %s %s", v, r)
	}
	partial := same[:2]
	if v, r, n := Overall(base, partial); v != Undecided || r != "partial-failure" || n != 2 {
		t.Errorf("partial failure: %s %s %d", v, r, n)
	}
	if v, _, _ := Overall(nil, same); v != Undecided {
		t.Errorf("no baseline must be unknown")
	}
}

func TestSplitChunks(t *testing.T) {
	got := SplitChunks("first part [...] second part [...]  ")
	if len(got) != 2 || got[0] != "first part" || got[1] != "second part" {
		t.Errorf("SplitChunks = %q", got)
	}
}

func BenchmarkDetect(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		Detect("Who is the current CEO of Microsoft and what is its share price today?", clock)
	}
}

func BenchmarkCompare(b *testing.B) {
	before := []string{"Plan costs ₹1,23,456 per year, billed on 3 November 2026.", "Exports are allowed for free accounts."}
	after := []string{"Plan costs ₹1,29,999 per year, billed on 3 November 2026.", "Exports are allowed for free accounts."}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		Compare(before, after)
	}
}
