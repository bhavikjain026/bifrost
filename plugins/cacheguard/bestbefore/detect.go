// Package bestbefore decides which cached answers need a review deadline
// ("is this question time-sensitive?") and whether the web sources saved for an
// answer have changed since it was cached. Both decisions are plain, explainable
// rules: no model is called.
package bestbefore

import (
	"regexp"
	"strconv"
	"strings"
	"time"
	"unicode"
)

// Status of the time-sensitivity check.
type Status string

const (
	Flagged    Status = "flagged"     // time-sensitive: give the cached answer a review deadline
	NotFlagged Status = "not-flagged" // no supported trigger found (NOT proof the answer never changes)
	Unknown    Status = "unknown"     // outside the supported scope (too long, non-English, ...)
)

// Categories and their default review deadlines (operator-configurable).
const (
	CatMarkets     = "markets"
	CatWeatherOps  = "weather_ops"
	CatLiveResults = "live_results"
	CatPricing     = "pricing"
	CatSoftware    = "software"
	CatRoles       = "roles"
	CatPolicy      = "policy"
	CatStatistics  = "statistics"
	CatRecency     = "recency"
	CatOpenYear    = "open_year"
)

// DefaultTTL is the review deadline per category.
var DefaultTTL = map[string]time.Duration{
	CatMarkets:     5 * time.Minute,
	CatWeatherOps:  15 * time.Minute,
	CatLiveResults: 5 * time.Minute,
	CatPricing:     24 * time.Hour,
	CatSoftware:    24 * time.Hour,
	CatRoles:       24 * time.Hour,
	CatPolicy:      24 * time.Hour,
	CatStatistics:  24 * time.Hour,
	CatRecency:     time.Hour,
	CatOpenYear:    24 * time.Hour,
}

// Hit is one rule that fired.
type Hit struct {
	Rule     string
	Category string
	Matched  string
}

// Result of Detect.
type Result struct {
	Status   Status
	Category string        // category with the shortest deadline among the hits
	TTL      time.Duration // shortest deadline among the hits
	Hits     []Hit
	Reason   string // for NotFlagged / Unknown
}

const RulesVersion = "v1"

const maxQuestionBytes = 8 << 10

// Phrases that use trigger words in a fixed, conceptual sense. They are blanked
// out before matching, so only *their* span is suppressed.
var conceptPhrases = []string{
	"electric current", "alternating current", "direct current", "current density", "current flow",
	"current in a circuit", "current in the circuit", "current in this circuit", "eddy current",
	"price elasticity", "pricing theory", "price discrimination", "cost function", "opportunity cost",
	"cost-benefit", "cost benefit", "marginal cost", "fixed cost", "variable cost",
	"time complexity", "space complexity", "newton's law", "ohm's law", "law of", "laws of",
	"current affairs quiz",
}

type rule struct {
	id, cat string
	re      *regexp.Regexp
	needs   *regexp.Regexp // optional extra condition anywhere in the clause
}

func w(alts string) string { return `\b(?:` + alts + `)\b` }

var (
	frameAsk  = regexp.MustCompile(w(`what|what's|how much|how many|which|who|when|is|are|does|current|currently|latest|today|now|right now`))
	frameLive = regexp.MustCompile(w(`now|right now|today|tonight|tomorrow|current|currently|latest|live|so far|this week|this weekend|at the moment|at present`))
)

var rules = []rule{
	// Live markets
	{"market-phrase", CatMarkets, regexp.MustCompile(w(`stock price|share price|exchange rate|forex rate|crypto price|market cap|gold rate|gold price|silver rate|silver price|petrol price|diesel price`)), frameAsk},
	{"market-asset", CatMarkets, regexp.MustCompile(w(`bitcoin|btc|ethereum|eth|sensex|nifty|dow jones|nasdaq|s&p 500|usd to inr|dollar to rupee|rupee to dollar`)), regexp.MustCompile(w(`price|rate|value|trading|worth|level|at`))},

	// Weather / operations / availability
	{"weather-ops", CatWeatherOps, regexp.MustCompile(w(`weather|forecast|temperature|rain|raining|aqi|air quality|traffic|flight status|delayed|outage|down for everyone|in stock|availability|available seats`)), frameLive},

	// Live sport / elections
	{"live-results", CatLiveResults, regexp.MustCompile(w(`score|standings|leaderboard|points table|medal table|leading|results|vote count|who is winning|who's winning`)), frameLive},

	// Retail / service pricing
	{"pricing", CatPricing, regexp.MustCompile(w(`price|prices|pricing|cost|costs|fee|fees|subscription|fare|fares|tariff|discount|offer|plan cost`)), regexp.MustCompile(w(`how much|what is the|what's the|what are the|current|currently|latest|today|now|cheapest|does it cost`))},

	// Software versions / support
	{"software-latest", CatSoftware, regexp.MustCompile(`\b(?:latest|newest|current|most recent)(?: [a-z0-9.+#-]+){0,3} (?:version|release|update|build|model|firmware)\b`), nil},
	{"software", CatSoftware, regexp.MustCompile(w(`latest version|latest release|newest version|current version|stable version|latest stable|release date|supported versions|end of life|eol|deprecated|still supported|security patch|latest update|newest release`)), nil},

	// Office holders / membership / records
	{"role-who-is", CatRoles, regexp.MustCompile(`\bwho(?: is|'s| are)(?: the)?(?: current| present)? (?:[a-z.&'-]+ ){0,3}?(?:ceo|cto|cfo|coo|president|prime minister|pm|chief minister|cm|governor|chair|chairman|chairperson|director|captain|coach|owner|founder|head|leader|minister|mayor|speaker|vice president|vice-chancellor|vc|dean|manager)\b`), nil},
	{"role-who-runs", CatRoles, regexp.MustCompile(w(`who leads|who owns|who runs|who heads|who manages`)), nil},
	{"role-current", CatRoles, regexp.MustCompile(`\bcurrent (?:ceo|cto|cfo|president|prime minister|chief minister|governor|chair|chairman|director|captain|coach|owner|head|leader|minister|champion|world champion|record holder|holder)\b`), nil},
	{"record", CatRoles, regexp.MustCompile(w(`holds the record|world record|current record|current champion|reigning champion`)), nil},

	// Rules / laws / policies
	{"policy", CatPolicy, regexp.MustCompile(w(`visa|tax rate|gst rate|income tax|tax slab|tax slabs|policy|law|laws|regulation|regulations|eligibility|eligible|requirement|requirements|deadline|ban|banned|legal`)), regexp.MustCompile(w(`current|currently|latest|still|now|effective|valid|allowed|required|need|needed|this year|today|new`))},

	// Scheduled statistical releases
	{"statistics", CatStatistics, regexp.MustCompile(w(`inflation|inflation rate|cpi|gdp|repo rate|interest rate|unemployment rate|population|fuel price index|wpi`)), regexp.MustCompile(w(`what|current|currently|latest|most recent|now|today|this year|this month`))},

	// Explicit recency with no domain match
	{"recency", CatRecency, regexp.MustCompile(w(`latest|newest|most recent|up to date|up-to-date|currently|today|tonight|this week|this month|right now|at present|at the moment|as of now|breaking`)), nil},
}

var (
	reWS        = regexp.MustCompile(`\s+`)
	reYear      = regexp.MustCompile(`\b(1[5-9]\d{2}|20\d{2})\b`)
	reClause    = regexp.MustCompile(`[?;!|]+|\.(?:\s|$)`)
	reAndWh     = regexp.MustCompile(`,?\s+and\s+(who|what|which|how|is|are|where|when)\b`)
	rePastFrame = regexp.MustCompile(w(`was|were|did|had|won|finished|ended|held|as of|in|during|before`))
	reLiveWords = regexp.MustCompile(w(`now|today|still|currently|so far|upcoming|will|next|current|present|right now|at the moment`))
	reOpenYear  = regexp.MustCompile(w(`latest|so far|year to date|ytd|upcoming|forecast|expected|scheduled|will|until now`))
	reDefine    = regexp.MustCompile(`^(?:define |what does .* mean|what is the meaning of |meaning of |explain the (?:concept|theory|idea) of )`)
)

// Detect classifies a single English user question. now is injected for testability.
func Detect(question string, now time.Time) Result {
	if len(question) > maxQuestionBytes {
		return Result{Status: Unknown, Reason: "too-long"}
	}
	if !mostlyLatin(question) {
		return Result{Status: Unknown, Reason: "unsupported-language"}
	}
	q := strings.ToLower(question)
	q = strings.NewReplacer("’", "'", "‘", "'", "“", `"`, "”", `"`).Replace(q)
	q = strings.TrimSpace(reWS.ReplaceAllString(q, " "))
	for _, p := range conceptPhrases {
		q = strings.ReplaceAll(q, p, strings.Repeat("_", len(p)))
	}
	if q == "" {
		return Result{Status: NotFlagged, Reason: "empty"}
	}
	definition := reDefine.MatchString(q) && !reLiveWords.MatchString(q)

	var hits []Hit
	for _, clause := range splitClauses(q) {
		if fixedPast(clause, now) || definition {
			continue
		}
		for _, r := range rules {
			if m := r.re.FindString(clause); m != "" && (r.needs == nil || r.needs.MatchString(clause)) {
				hits = append(hits, Hit{Rule: r.id, Category: r.cat, Matched: m})
			}
		}
		if y := openYear(clause, now); y != "" {
			hits = append(hits, Hit{Rule: "open-year", Category: CatOpenYear, Matched: y})
		}
	}
	if len(hits) == 0 {
		return Result{Status: NotFlagged, Reason: "no-supported-trigger"}
	}
	// Plain recency words are a fallback: drop them when a domain rule also fired.
	if hasDomain(hits) {
		kept := hits[:0]
		for _, h := range hits {
			if h.Category != CatRecency {
				kept = append(kept, h)
			}
		}
		hits = kept
	}
	res := Result{Status: Flagged, Hits: hits}
	for _, h := range hits {
		ttl := DefaultTTL[h.Category]
		if res.TTL == 0 || ttl < res.TTL {
			res.TTL, res.Category = ttl, h.Category
		}
	}
	return res
}

// TTLFor resolves the deadline for a result using operator overrides (category -> seconds).
func TTLFor(r Result, overrides map[string]int) time.Duration {
	var best time.Duration
	for _, h := range r.Hits {
		ttl := DefaultTTL[h.Category]
		if s, ok := overrides[h.Category]; ok && s > 0 {
			ttl = time.Duration(s) * time.Second
		}
		if best == 0 || ttl < best {
			best = ttl
		}
	}
	return best
}

func hasDomain(hits []Hit) bool {
	for _, h := range hits {
		if h.Category != CatRecency {
			return true
		}
	}
	return false
}

func splitClauses(q string) []string {
	parts := reClause.Split(reAndWh.ReplaceAllString(q, "|$1"), -1)
	out := parts[:0]
	for _, p := range parts {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}

// fixedPast: the clause is explicitly about a completed past period
// ("who was CEO in 2010", "the latest king as of 1700") and asks nothing about now.
func fixedPast(clause string, now time.Time) bool {
	if reLiveWords.MatchString(clause) && !strings.Contains(clause, "as of") {
		return false
	}
	for _, m := range reYear.FindAllString(clause, -1) {
		y, _ := strconv.Atoi(m)
		if y < now.Year() && rePastFrame.MatchString(clause) {
			return true
		}
	}
	return false
}

// openYear: the current or previous year together with an open-ended frame.
func openYear(clause string, now time.Time) string {
	if !reOpenYear.MatchString(clause) {
		return ""
	}
	for _, m := range reYear.FindAllString(clause, -1) {
		y, _ := strconv.Atoi(m)
		if y == now.Year() || y == now.Year()-1 || y == now.Year()+1 {
			return m
		}
	}
	return ""
}

func mostlyLatin(s string) bool {
	var letters, other int
	for _, r := range s {
		if !unicode.IsLetter(r) {
			continue
		}
		letters++
		if !unicode.Is(unicode.Latin, r) {
			other++
		}
	}
	return letters == 0 || float64(other)/float64(letters) <= 0.3
}
