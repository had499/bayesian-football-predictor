# WP062 — position / goalkeeper / severity refinements to the injury signal: ALL NULL

WP059's feature is one coefficient on TOTAL unavailable minutes. Three refinements tested,
each with coefficients chosen on windows <=200 and evaluated on 201-343 (n=1,305).

| arm | params | HELD-OUT vs CLOSE | vs OUTCOME |
|---|---|---|---|
| **`out_tot` (incumbent)** | 0.20 | **−0.000113 [−0.000170,−0.000059]** | −0.00025 [−0.00064,+0.00014] |
| POSITION (attack/defence split) | 0.40, 0.30 | −0.000111 [−0.000162,−0.000062] | −0.00024 |
| + GOALKEEPER | 0.20, **0.00** | −0.000113 | −0.00025 |
| + SEVERITY (>=30 days in) | 0.20, 0.45 | −0.000115 | −0.00026 |
| long absences ONLY | 0.30 | −0.000013 [−0.000027,+0.000001] | −0.00005 |

Decision rule set in advance (five arms are reported, so picking the eval-best would itself
be selection): adopt only if a refinement beats `out_tot` by at least what `out_tot` is
worth. None comes close. **The crude feature is already optimal.**

## Three substantive readings, not just nulls

**GOALKEEPING IS ANSWERED.** The GK coefficient was selected at exactly **0.00** on the
selection half — the data actively prefers no GK term. A missing first-choice keeper is worth
precisely his minute share. This is the third independent closure: WP016's defence split
(suppression + shot-stopping) made RPS WORSE, WP056's GK-continuity feature was null, and now
a GK-absence multiplier is null. Goalkeeping carries nothing special in this model and the
gap WP016 flagged should stop being treated as open.

**POSITION IS IRRELEVANT** — losing a striker and losing a centre-back are equivalent, with
an attack/defence split scoring fractionally WORSE than the undifferentiated total. So the
signal is general squad-quality loss, not role-specific damage, and "share of recent minutes
unavailable" is the right abstraction with no finer structure to exploit.

**SEVERITY DOES NOT CONCENTRATE IT.** Restricting to players >=30 days into a spell gives
−0.000013 against the total's −0.000113, i.e. ~12% of the signal from the most serious
absences. The effect is spread across all absences, consistent with WP059's finding that
`out_tot` beat `out_w_new` and with the collapse of my "unpriced change" mechanism.

Severity was deliberately measured leak-free as DAYS ELAPSED since the spell began (known at
prediction time) rather than total spell length, which depends on the `until` date WP059c had
to audit for look-ahead.
