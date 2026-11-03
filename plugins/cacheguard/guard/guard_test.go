package guard

import "testing"

func TestVerify(t *testing.T) {
	cases := []struct {
		name, a, b string
		reuse      bool
		reason     string
	}{
		{"identical", "How do I reset my password?", "How do I reset my password?", true, ""},
		{"format only", "how do i enable 2fa", "How do I enable 2FA?", true, ""},
		{"politeness", "What is the refund policy? Thanks!", "What is the refund policy?", true, ""},
		{"paraphrase", "How do I reset my password?", "What is the process for resetting my password?", true, ""},
		{"negation word", "Can I deploy the agent without a license key?", "Can I deploy the agent with a license key?", false, RuleNegation},
		{"not", "Does this API not require authentication?", "Does this API require authentication?", false, RuleNegation},
		{"antonym", "Does the free plan exclude SSO?", "Does the free plan include SSO?", false, RuleAntonym},
		{"prefix antonym", "Are deleted records irrecoverable?", "Are deleted records recoverable?", false, RuleAntonym},
		{"number", "What is the rate limit on the 10k plan?", "What is the rate limit on the 100k plan?", false, RuleNumber},
		{"version", "How do I call the v3 API?", "How do I call the v2 API?", false, RuleNumber},
		{"relative time", "What broke this week?", "What broke last week?", false, RuleTime},
		{"entity", "What is the refund policy for US customers?", "What is the refund policy for EU customers?", false, RuleEntity},
		{"language", "Как сбросить пароль?", "How do I reset my password?", false, RuleLanguage},
		{"substitution", "How do I rotate the API key for the production environment?", "How do I rotate the API key for the staging environment?", false, RuleSubstitution},
		{"repeated word", "Can sandbox API keys charge live cards?", "Can live API keys charge live cards?", false, RuleSubstitution},
		{"pronoun I is not an entity", "Where is the audit log?", "How do I find the audit log?", true, ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			v := Verify(c.a, c.b)
			if v.Reuse != c.reuse || (!c.reuse && v.Reason != c.reason) {
				t.Fatalf("Verify(%q, %q) = %+v, want reuse=%v reason=%q", c.a, c.b, v, c.reuse, c.reason)
			}
		})
	}
}

func TestDisabledRule(t *testing.T) {
	a, b := "How do I rotate the API key for the production environment?", "How do I rotate the API key for the staging environment?"
	if v := VerifyWith(a, b, Options{Disabled: map[string]bool{RuleSubstitution: true}}); !v.Reuse {
		t.Fatalf("substitution disabled but still rejected: %+v", v)
	}
}

var sink Verdict

func BenchmarkVerifyReuse(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sink = Verify("What is the process for resetting my password?", "How do I reset my password?")
	}
}

func BenchmarkVerifyReject(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sink = Verify("Can I deploy the agent without a license key?", "Can I deploy the agent with a license key?")
	}
}
