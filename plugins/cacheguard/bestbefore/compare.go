package bestbefore

import (
	"html"
	"regexp"
	"sort"
	"strings"
	"time"
)

// Verdict of comparing a source's saved passages with a new observation.
type Verdict string

const (
	Unchanged Verdict = "unchanged"
	Changed   Verdict = "changed"
	Undecided Verdict = "unknown" // could not compare reliably: treated like a change (no reuse)
)

// Comparison explains one verdict.
type Comparison struct {
	Verdict Verdict
	Reason  string   // numbers-changed, dates-changed, negation-changed, text-changed, chunk-drift, empty, ...
	Removed []string // diagnostic: numbers/dates that disappeared
	Added   []string // diagnostic: numbers/dates that appeared
}

const NormalizerVersion = "v1"

// SplitChunks splits Tavily Extract query-focused content into its chunks
// (Tavily joins the selected chunks with "[...]").
func SplitChunks(raw string) []string {
	var out []string
	for _, c := range strings.Split(raw, "[...]") {
		if c = strings.TrimSpace(c); c != "" {
			out = append(out, c)
		}
	}
	return out
}

var (
	reSpaces  = regexp.MustCompile(`[ \t\r\n\f\v]+`)
	reNumber  = regexp.MustCompile(`\d[\d,]*(?:\.\d+)?`)
	reWestern = regexp.MustCompile(`^\d{1,3}(?:,\d{3})+(?:\.\d+)?$`)
	reIndian  = regexp.MustCompile(`^\d{1,2}(?:,\d{2})*,\d{3}(?:\.\d+)?$`)
	rePercent = regexp.MustCompile(`(\d(?:\.\d+)?)\s*(?:%|percent\b|per cent\b)`)
	reDMY     = regexp.MustCompile(`\b(\d{1,2}) (January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.? (\d{4})\b`)
	reMDY     = regexp.MustCompile(`\b(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.? (\d{1,2}), (\d{4})\b`)
	reISO     = regexp.MustCompile(`\b\d{4}-\d{2}-\d{2}\b`)
	reNeg     = regexp.MustCompile(`(?i)\b(?:not|no|never|none|without|cannot|can't|won't|isn't|aren't|wasn't|weren't|doesn't|don't|didn't|only|may|might)\b`)
	reWord    = regexp.MustCompile(`[\p{L}\p{N}]+`)
)

var months = map[string]time.Month{
	"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

// Normalize applies the small, conservative, versioned normalisation (v1):
// HTML entities, non-breaking spaces and whitespace; validated digit grouping
// (1,23,456 and 123,456 -> plain digits); the rupee sign; percent spellings; and
// unambiguous English dates -> YYYY-MM-DD. Case, punctuation, negation, units and
// ambiguous dates (03/04/2026) are preserved.
func Normalize(s string) string {
	s = html.UnescapeString(s)
	s = strings.NewReplacer(" ", " ", " ", " ", " ", " ", "₹", "INR ").Replace(s)
	s = reDMY.ReplaceAllStringFunc(s, func(m string) string {
		p := reDMY.FindStringSubmatch(m)
		return isoDate(p[3], p[2], p[1], m)
	})
	s = reMDY.ReplaceAllStringFunc(s, func(m string) string {
		p := reMDY.FindStringSubmatch(m)
		return isoDate(p[3], p[1], p[2], m)
	})
	s = reNumber.ReplaceAllStringFunc(s, func(m string) string {
		trimmed := strings.TrimRight(m, ",")
		if reWestern.MatchString(trimmed) || reIndian.MatchString(trimmed) {
			return strings.ReplaceAll(trimmed, ",", "") + m[len(trimmed):]
		}
		return m
	})
	s = rePercent.ReplaceAllString(s, "$1%")
	s = reSpaces.ReplaceAllString(s, " ")
	s = strings.ReplaceAll(s, "INR  ", "INR ")
	return strings.TrimSpace(s)
}

func isoDate(year, month, day, orig string) string {
	mo, ok := months[strings.ToLower(month)[:3]]
	if strings.EqualFold(month, "sept") {
		mo, ok = 9, true
	}
	if !ok {
		return orig
	}
	t, err := time.Parse("2006-1-2", year+"-"+itoa(int(mo))+"-"+day)
	if err != nil || t.Day() != atoi(day) { // reject 31 February etc.
		return orig
	}
	return t.Format("2006-01-02")
}

// Compare decides whether a source changed. Reuse is allowed only when the
// multiset of normalised chunks is identical; the rest only explains why not.
func Compare(before, after []string) Comparison {
	if len(after) == 0 {
		return Comparison{Verdict: Undecided, Reason: "empty"}
	}
	nb, na := normAll(before), normAll(after)
	if equalMultiset(nb, na) {
		return Comparison{Verdict: Unchanged, Reason: "same-passages"}
	}
	jb, ja := strings.Join(nb, "\n"), strings.Join(na, "\n")
	removed, added := diff(facts(jb), facts(ja))
	c := Comparison{Verdict: Changed, Removed: removed, Added: added}
	switch {
	case len(removed)+len(added) > 0 && hasDate(removed, added):
		c.Reason = "dates-changed"
	case len(removed)+len(added) > 0:
		c.Reason = "numbers-changed"
	case !equalStrings(sorted(reNeg.FindAllString(strings.ToLower(jb), -1)), sorted(reNeg.FindAllString(strings.ToLower(ja), -1))):
		c.Reason = "negation-changed"
	case wordOverlap(jb, ja) < 0.5:
		// Mostly different text: the query-focused window probably moved. Still no reuse.
		c.Verdict, c.Reason = Undecided, "chunk-drift"
	default:
		c.Reason = "text-changed"
	}
	return c
}

func normAll(chunks []string) []string {
	out := make([]string, 0, len(chunks))
	for _, c := range chunks {
		if n := Normalize(c); n != "" {
			out = append(out, n)
		}
	}
	return out
}

// facts returns the numbers and ISO dates in normalised text, as a multiset.
func facts(s string) []string {
	var out []string
	out = append(out, reISO.FindAllString(s, -1)...)
	s = reISO.ReplaceAllString(s, " ")
	for _, n := range reNumber.FindAllString(s, -1) {
		out = append(out, strings.TrimRight(n, ","))
	}
	return out
}

func diff(a, b []string) (removed, added []string) {
	count := map[string]int{}
	for _, x := range a {
		count[x]++
	}
	for _, x := range b {
		if count[x] > 0 {
			count[x]--
		} else {
			added = append(added, x)
		}
	}
	for x, n := range count {
		for ; n > 0; n-- {
			removed = append(removed, x)
		}
	}
	sort.Strings(removed)
	return removed, added
}

func hasDate(lists ...[]string) bool {
	for _, l := range lists {
		for _, x := range l {
			if reISO.MatchString(x) {
				return true
			}
		}
	}
	return false
}

func wordOverlap(a, b string) float64 {
	wa, wb := map[string]bool{}, map[string]bool{}
	for _, x := range reWord.FindAllString(strings.ToLower(a), -1) {
		wa[x] = true
	}
	for _, x := range reWord.FindAllString(strings.ToLower(b), -1) {
		wb[x] = true
	}
	if len(wa) == 0 || len(wb) == 0 {
		return 0
	}
	inter := 0
	for x := range wa {
		if wb[x] {
			inter++
		}
	}
	return float64(inter) / float64(len(wa)+len(wb)-inter)
}

func equalMultiset(a, b []string) bool {
	return equalStrings(sorted(a), sorted(b))
}

func sorted(a []string) []string {
	c := append([]string(nil), a...)
	sort.Strings(c)
	return c
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// SourceObservation is what a recheck returned for one saved URL.
type SourceObservation struct {
	URL    string
	Chunks []string
	Failed bool // in failed_results, missing from the response, or empty
}

// Overall combines per-source results: every source must be unchanged; one changed
// source is enough to invalidate (no majority vote); any failure means unknown.
func Overall(baseline map[string][]string, observed []SourceObservation) (Verdict, string, int) {
	byURL := map[string]SourceObservation{}
	for _, o := range observed {
		byURL[o.URL] = o
	}
	ok, verdict, reason := 0, Unchanged, "same-passages"
	urls := make([]string, 0, len(baseline))
	for u := range baseline {
		urls = append(urls, u)
	}
	sort.Strings(urls)
	for _, u := range urls {
		o, found := byURL[u]
		if !found || o.Failed || len(o.Chunks) == 0 {
			if verdict == Unchanged {
				verdict, reason = Undecided, "partial-failure"
			}
			continue
		}
		ok++
		c := Compare(baseline[u], o.Chunks)
		switch c.Verdict {
		case Changed:
			verdict, reason = Changed, c.Reason
		case Undecided:
			if verdict == Unchanged {
				verdict, reason = Undecided, c.Reason
			}
		}
	}
	if len(baseline) == 0 {
		return Undecided, "no-baseline", 0
	}
	return verdict, reason, ok
}

func itoa(i int) string {
	if i < 10 {
		return string(rune('0' + i))
	}
	return string(rune('0'+i/10)) + string(rune('0'+i%10))
}

func atoi(s string) int {
	n := 0
	for _, r := range s {
		n = n*10 + int(r-'0')
	}
	return n
}
