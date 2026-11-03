// Package guard decides whether a cached answer for one prompt may be served
// for another prompt that the embedding search judged "similar".
//
// Stage-1 checks are deterministic, allocation-light and run in microseconds.
// They target the differences that cosine similarity is known to miss:
// negation/polarity, numbers & versions & dates, named entities, script/language,
// and minimal-pair substitutions (one or two content words swapped).
package guard

import (
	"regexp"
	"strings"
	"unicode"
)

// Verdict is the result of comparing a new prompt with a cached prompt.
type Verdict struct {
	Reuse  bool   // true = safe to serve the cached answer
	Reason string // why it was rejected ("" when Reuse is true)
	Detail string // the tokens that triggered the rejection
}

var (
	tokenRe  = regexp.MustCompile(`[\p{L}\p{N}][\p{L}\p{N}'’.\-]*`)
	digitRe  = regexp.MustCompile(`\p{N}`)
	negators = set("not", "no", "never", "without", "cannot", "none", "nor", "neither", "nothing", "nobody")
	// General-English antonym pairs (not dataset-specific).
	antonyms = pairs(
		"with:without", "include:exclude", "includes:excludes", "included:excluded",
		"enable:disable", "enabled:disabled", "allow:deny", "allow:block", "allowed:blocked",
		"public:private", "accept:reject", "add:remove", "increase:decrease", "before:after",
		"min:max", "minimum:maximum", "more:less", "upload:download", "import:export",
		"input:output", "on:off", "start:stop", "open:close", "encrypt:decrypt", "read:write",
		"inbound:outbound", "internal:external", "global:local", "first:last", "old:new",
		"above:below", "higher:lower", "cheapest:most", "free:paid", "required:optional",
		"sync:async", "synchronous:asynchronous", "online:offline", "login:logout",
	)
	stopwords = set(
		// function words
		"a", "an", "the", "to", "for", "of", "on", "in", "at", "by", "from", "into", "about", "and", "or",
		"is", "are", "was", "were", "be", "been", "am", "do", "does", "did", "can", "could", "would",
		"should", "will", "shall", "may", "might", "must", "i", "me", "my", "we", "our", "you", "your",
		"it", "its", "this", "that", "these", "those", "there", "here", "if", "so", "any", "some",
		"how", "what", "where", "why", "when", "which", "who", "whom", "whose",
		// politeness / meta-question words that paraphrases add or drop
		"please", "kindly", "thanks", "thank", "hi", "hello", "hey", "tell", "know", "want", "need",
		"step", "steps", "process", "way", "ways", "method", "procedure", "guide", "explain",
		"get", "find", "see", "use", "using", "help", "possible",
	)
	stemSuffixes = []string{"ations", "ation", "ments", "ment", "ures", "ure", "ings", "ing", "ies", "ed", "es", "s", "ly"}
	negPrefixes  = []string{"un", "in", "ir", "il", "im", "dis", "non"}
	// Relative-time words: "last week" vs "this week" asks about a different period.
	deixis = set("yesterday", "today", "tomorrow", "tonight", "last", "this", "next", "previous",
		"current", "upcoming", "past", "ago", "now", "latest", "recent", "recently", "earlier", "later")
	sentenceEnd = regexp.MustCompile(`[.?!]\s*$`)
)

// Rule names, usable in Options.Disabled.
const (
	RuleLanguage     = "language"
	RuleNegation     = "negation"
	RuleNumber       = "number"
	RuleTime         = "time"
	RuleEntity       = "entity"
	RuleAntonym      = "antonym"
	RuleSubstitution = "substitution"
)

// AllRules lists every stage-1 rule in evaluation order.
var AllRules = []string{RuleLanguage, RuleNegation, RuleNumber, RuleTime, RuleEntity, RuleAntonym, RuleSubstitution}

// Options tunes the checker. The zero value enables every rule.
type Options struct {
	Disabled map[string]bool
}

func (o Options) on(rule string) bool { return !o.Disabled[rule] }

// Verify returns whether the cached answer for cachedPrompt may be served for newPrompt,
// with every rule enabled.
func Verify(newPrompt, cachedPrompt string) Verdict {
	return VerifyWith(newPrompt, cachedPrompt, Options{})
}

// VerifyWith is Verify with configurable rules.
func VerifyWith(newPrompt, cachedPrompt string, o Options) Verdict {
	if norm(newPrompt) == norm(cachedPrompt) {
		return Verdict{Reuse: true}
	}
	a, b := tokens(newPrompt), tokens(cachedPrompt)

	if la, lb := scriptShare(newPrompt), scriptShare(cachedPrompt); o.on(RuleLanguage) && (la > 0.3) != (lb > 0.3) {
		return Verdict{Reason: "language", Detail: "different script"}
	}
	if o.on(RuleNegation) && countNeg(a) != countNeg(b) {
		return Verdict{Reason: "negation", Detail: "negation word count differs"}
	}
	if d := diffList(numbers(a), numbers(b)); o.on(RuleNumber) && d != "" {
		return Verdict{Reason: "number", Detail: d}
	}
	if d := diffList(filter(a, deixis), filter(b, deixis)); o.on(RuleTime) && d != "" {
		return Verdict{Reason: "time", Detail: d}
	}
	if d := missingEntity(newPrompt, cachedPrompt); o.on(RuleEntity) && d != "" {
		return Verdict{Reason: "entity", Detail: d}
	}

	ca, cb := content(a), content(b)
	onlyA, onlyB, common := multisetDiff(ca, cb)
	if d := antonymHit(onlyA, onlyB); o.on(RuleAntonym) && d != "" {
		return Verdict{Reason: "antonym", Detail: d}
	}
	// Minimal pair: same sentence, one or two content words swapped for others.
	// The swapped words are then the point of the question (staging vs production,
	// trial vs enterprise accounts), so the answers differ.
	if o.on(RuleSubstitution) && len(onlyA) >= 1 && len(onlyA) <= 2 && len(onlyB) >= 1 && len(onlyB) <= 2 && len(common) >= 2 {
		return Verdict{Reason: "substitution", Detail: strings.Join(onlyA, ",") + " vs " + strings.Join(onlyB, ",")}
	}
	return Verdict{Reuse: true}
}

func norm(s string) string { return strings.Join(tokens(s), " ") }

func tokens(s string) []string {
	raw := tokenRe.FindAllString(strings.ToLower(s), -1)
	out := make([]string, 0, len(raw))
	for _, t := range raw {
		t = strings.Trim(strings.ReplaceAll(t, "’", "'"), ".-'")
		if t != "" {
			out = append(out, t)
		}
	}
	return out
}

func countNeg(ts []string) int {
	n := 0
	for _, t := range ts {
		if negators[t] || strings.HasSuffix(t, "n't") {
			n++
		}
	}
	return n
}

// numbers returns normalised numeric tokens: "3.1", "100k", "30-day" -> "30-day", "v2", "q1", "2026".
func numbers(ts []string) []string {
	var out []string
	for _, t := range ts {
		if digitRe.MatchString(t) {
			out = append(out, t)
		}
	}
	return out
}

// missingEntity reports a capitalised / acronym token from either prompt that the
// other prompt does not contain (case-insensitive), e.g. EU vs US, Redis vs Postgres.
func missingEntity(x, y string) string {
	ex, ey := entities(x), entities(y)
	tx, ty := set(tokens(x)...), set(tokens(y)...)
	for _, e := range ex {
		if !ty[e] {
			return e
		}
	}
	for _, e := range ey {
		if !tx[e] {
			return e
		}
	}
	return ""
}

func entities(s string) []string {
	var out []string
	locs := tokenRe.FindAllStringIndex(s, -1)
	for i, loc := range locs {
		t := s[loc[0]:loc[1]]
		r := []rune(t)
		low := strings.Trim(strings.ToLower(t), ".-'")
		if low == "i" || stopwords[low] || negators[low] {
			continue
		}
		sentenceStart := i == 0 || sentenceEnd.MatchString(s[locs[i-1][0]:loc[0]])
		upper := 0
		for _, c := range r {
			if unicode.IsUpper(c) {
				upper++
			}
		}
		isAcronym := upper >= 2
		isProper := !sentenceStart && unicode.IsUpper(r[0])
		if isAcronym || isProper {
			out = append(out, low)
		}
	}
	return out
}

func content(ts []string) []string {
	var out []string
	for _, t := range ts {
		if stopwords[t] || negators[t] {
			continue
		}
		out = append(out, stem(t))
	}
	return out
}

func stem(t string) string {
	t = strings.TrimSuffix(t, "'s")
	for _, suf := range stemSuffixes {
		if len(t) > len(suf)+3 && strings.HasSuffix(t, suf) {
			t = strings.TrimSuffix(t, suf)
			// resetting -> resett -> reset
			if n := len(t); n > 2 && t[n-1] == t[n-2] && !strings.ContainsRune("aeiouls", rune(t[n-1])) {
				t = t[:n-1]
			}
			break
		}
	}
	return t
}

func antonymHit(onlyA, onlyB []string) string {
	for _, x := range onlyA {
		for _, y := range onlyB {
			if antonyms[x+":"+y] || antonyms[y+":"+x] {
				return x + "/" + y
			}
			for _, p := range negPrefixes {
				if x == p+y || y == p+x {
					return x + "/" + y
				}
			}
		}
	}
	return ""
}

func scriptShare(s string) float64 {
	letters, nonLatin := 0, 0
	for _, c := range s {
		if unicode.IsLetter(c) {
			letters++
			if c > unicode.MaxLatin1 && !unicode.Is(unicode.Latin, c) {
				nonLatin++
			}
		}
	}
	if letters == 0 {
		return 0
	}
	return float64(nonLatin) / float64(letters)
}

func setDiff(a, b []string) (onlyA, onlyB, common []string) {
	sa, sb := set(a...), set(b...)
	for k := range sa {
		if sb[k] {
			common = append(common, k)
		} else {
			onlyA = append(onlyA, k)
		}
	}
	for k := range sb {
		if !sa[k] {
			onlyB = append(onlyB, k)
		}
	}
	return
}

func diffList(a, b []string) string {
	onlyA, onlyB, _ := setDiff(a, b)
	if len(onlyA)+len(onlyB) == 0 {
		return ""
	}
	return strings.Join(onlyA, ",") + " vs " + strings.Join(onlyB, ",")
}

func set(xs ...string) map[string]bool {
	m := make(map[string]bool, len(xs))
	for _, x := range xs {
		m[x] = true
	}
	return m
}

func pairs(ps ...string) map[string]bool {
	m := make(map[string]bool, len(ps))
	for _, p := range ps {
		m[p] = true
	}
	return m
}

func filter(ts []string, keep map[string]bool) []string {
	var out []string
	for _, t := range ts {
		if keep[t] {
			out = append(out, t)
		}
	}
	return out
}

// multisetDiff is like setDiff but counts repeats ("live API keys charge live cards").
func multisetDiff(a, b []string) (onlyA, onlyB, common []string) {
	ca, cb := map[string]int{}, map[string]int{}
	for _, x := range a {
		ca[x]++
	}
	for _, x := range b {
		cb[x]++
	}
	for k, n := range ca {
		switch {
		case n > cb[k]:
			onlyA = append(onlyA, k)
			if cb[k] > 0 {
				common = append(common, k)
			}
		default:
			common = append(common, k)
		}
	}
	for k, n := range cb {
		if n > ca[k] {
			onlyB = append(onlyB, k)
		}
	}
	return
}
