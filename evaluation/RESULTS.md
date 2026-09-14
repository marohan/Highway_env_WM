# Measured results (2026-09-14)

Re-run after fixing priorities 1–4. All numbers produced on this checkout with
`.venv_mac`; reproduce with the commands at the bottom.

These supersede an earlier set of recorded figures, which were stale and wrong in
both directions (they reported the agent at speed 2.75 with 0 collisions; the
agent actually drives *faster* than IDM and the safety cost shows up as
collisions, not as crawling).

---

## 1. Pareto vs. the IDM oracle

**Pre-registered acceptance criterion.** IDM/MOBIL is not a peer. It reads exact
positions and velocities of every vehicle from the simulator, has no sensing
pipeline and no self-model uncertainty, and is the same model driving the
surrounding traffic, so it is effectively coordinating with copies of itself.
Parity is the wrong bar; a threshold relative to it is the right one:

> PASS if `collision_rate <= IDM + 0.05` **and** `mean_speed >= 0.70 x IDM`.

30 episodes per row, 40 s each, paired seeds 1000–1029,
`vehicles_count=20, density=1.5`, `policy_frequency=1`. Wilson 95% CI.

| agent | v_limit | rear_rss | collision rate | 95% CI | mean speed | retention | verdict |
|---|---|---|---|---|---|---|---|
| **IDM/MOBIL (oracle)** | — | — | 0.000 | [0.00, 0.11] | 20.40 | 1.00 | reference |
| l1-only (no L2) | 30 | on | 0.667 | [0.49, 0.81] | 25.50 | 1.25 | FAIL |
| flow-L2 (legacy heuristic) | 30 | n/a | 0.200 | [0.10, 0.37] | 22.86 | 1.12 | FAIL |
| viability-L2 | 30 | on | 0.333 | [0.19, 0.51] | 22.89 | 1.12 | FAIL |
| viability-L2 | 26 | on | 0.133 | [0.05, 0.30] | 21.80 | 1.07 | FAIL |
| viability-L2 | 23 | on | 0.067 | [0.02, 0.21] | 20.99 | 1.03 | FAIL (marginal) |
| **viability-L2** | **20** | **on** | **0.000** | **[0.00, 0.11]** | **19.87** | **0.97** | **PASS** |
| viability-L2 | 23 | off (pre-fix) | 0.067 | [0.02, 0.21] | 21.06 | 1.03 | FAIL (marginal) |
| viability-L2 | 30 | off (pre-fix) | 0.333 | [0.19, 0.51] | 22.98 | 1.12 | FAIL |

Readings:

- **The criterion is met at `v_limit=20`:** zero collisions in 30 episodes at 97%
  of the oracle's speed. This is provisional — with 0 events the CI upper bound
  is still 0.11, and the project's own pre-registered power standard asks for ~30
  collision *events*, not 30 episodes.
- **A genuine frontier exists and is monotone.** v_limit 20 → 23 → 26 → 30 maps
  to collisions 0.000 → 0.067 → 0.133 → 0.333 and speed 19.87 → 20.99 → 21.80 →
  22.89. The agent is not a crawler; it trades safety for speed on a clean curve.
- **`k_factor` is a dead knob.** k = 1.0 / 2.0 / 3.0 all give collision 0.333 and
  speed 22.8–23.0 at v_limit=30. Under viability-L2, `k` only enters the flow
  tie-break and L1.5, which rarely bind. The old `evaluate_pareto.py` swept
  `k`/`t_react`/`a_lead_max`, i.e. it was sweeping the wrong axis. `v_limit` is
  the axis that moves the frontier.
- **L2 is load-bearing.** Removing deliberation entirely (l1-only) gives 0.667
  collisions. The three-layer split is doing real work.
- **viability-L2 vs. flow-L2 at n=30 is not separable** (resolved in section 4 below
  under the event-count protocol — it separates, and not in the project's favour).
  At n=30: 0.333 [0.19,0.51] vs
  0.200 [0.10,0.37] at matched v_limit; at matched *speed* (~21.8 vs 22.9)
  viability is at 0.133 vs flow's 0.200. Overlapping CIs in both directions. The
  central claim of the project remains unproven, not disproven. Separating them
  requires the event-count protocol.

## 2. Section-11 gate — the forward-bias prediction

Prediction: braking authority (5 m/s²) exceeds accelerating authority (3 m/s²),
so a pure option count should prefer a gap position biased **toward the leader**,
and the pain gradient should shrink that bias to ~0.

Measured as `logZ(position)` across the gap (the closed-loop version cannot
resolve it: the ±2 m/s action lattice yields a continuum of fixed points rather
than a unique equilibrium). Bias is reported relative to the alive-band
midpoint, which controls for the residual forward/rear margin asymmetry.

gap = 60 m, v_ego = 20 m/s, T = 8, K = 1:

| beta | rear_rss OFF (pre-fix) | rear_rss ON (fixed) |
|---|---|---|
| 0.00 | +8.0 m (band [-26,+14], mid -6.0) | **+7.0 m** (band [-16,+14], mid -1.0) |
| 0.25 | +6.0 m | +7.0 m |
| 0.50 | +6.0 m | +7.0 m |
| 0.75 | +6.0 m | **+1.0 m** |
| 1.00 | +6.0 m | **+1.0 m** |

> **CORRECTION.** The table above was measured with the ego swept-volume bug
> still present (see section 5). With the swept volume implemented, the beta=0
> point moves to **-5.0 m** at gap=60/v_ego=20 and the clean "+7 collapsing to
> +1" pattern does not survive. Post-fix numbers are below. The pre-fix reading
> in this paragraph is retained only to document what changed.

**With the symmetric rear RSS (pre-swept-volume), the prediction reproduced:** +7.0 m at beta <= 0.5,
collapsing to +1.0 m at beta >= 0.75. Same signature as the parent project's
+14.4 px → +1.7 px. Without the fix, the offset does not respond to beta at all
(stuck at +6.0) and the alive band is grossly asymmetric ([-26,+14]) — the
constraint artifact masks the effect under test.

Robustness (bias at beta = 0 / 0.5 / 1.0, argmax minus band midpoint):

| gap | v_ego | b=0 | b=0.5 | b=1.0 |
|---|---|---|---|---|
| 44 | 16 | +1.0 | -1.0 | -1.0 |
| 44 | 20 | -3.0 | -1.0 | -1.0 |
| 44 | 24 | dead band | | |
| 60 | 16 | +3.0 | -1.0 | -5.0 |
| 60 | 20 | +7.0 | +7.0 | +1.0 |
| 60 | 24 | -3.0 | +1.0 | +1.0 |
| 80 | 16 | +13.0 | +3.0 | +3.0 |
| 80 | 20 | +15.0 | +3.0 | +1.0 |
| 80 | 24 | -9.0 | -1.0 | +7.0 |

### Post-swept-volume re-measurement (current code)

Bias = argmax minus alive-band midpoint, rear_rss=ON:

| gap | v_ego | b=0 | b=0.5 | b=1.0 |
|---|---|---|---|---|
| 60 | 16 | +9.0 | -1.0 | -1.0 |
| 60 | 20 | **-5.0** | +1.0 | +1.0 |
| 60 | 24 | +1.0 | +1.0 | +1.0 |
| 80 | 16 | +19.0 | +1.0 | +1.0 |
| 80 | 20 | +9.0 | +3.0 | +3.0 |
| 80 | 24 | -9.0 | +7.0 | +7.0 |

**Verdict on the gate: partially reproduced, not cleanly.** In 3 of 6 cells the
beta=0 bias is strongly positive and collapses toward ~0 as predicted (60/16:
+9 -> -1; 80/16: +19 -> +1; 80/20: +9 -> +3). In the other 3 it is zero or
negative at beta=0, including the headline cell (60/20) which flips to -5.0 m.
The magnitude scales with gap size and is largest when the ego is slower than
traffic, which is mechanically consistent with the hypothesis, but the sign is
not robust across the operating envelope.

This is a weaker result than the pre-fix table suggested. Reporting it as a
clean reproduction would have been reporting a bug.

Earlier reading (pre-fix): the prediction holds **in the regime it was stated for** — ego at
or below traffic speed, gap wide enough that the alive band is not degenerate —
and grows with gap size (+7 at 60 m, +15 at 80 m). It **inverts when the ego is
overtaking** (v_ego = 24 > traffic 20), which is mechanically sensible: closing
on the leader inflates the forward margin requirement and pushes the preferred
position back. That regime dependence was not part of the pre-registration and
should be stated as a finding, not smoothed over. The 2 m lattice also quantizes
the argmax coarsely.

## 3. Tie-set instrumentation — how much does logZ actually decide?

Logged over the 30-episode runs at v_limit=30:

| k | plans | with >1 safe option | logZ decided | heuristic decided | all-dead | mean tie size |
|---|---|---|---|---|---|---|
| 1.0 | 809 | 552 | 475 (86%) | 77 | 19 | 1.10 |
| 2.0 | 804 | 539 | 462 (86%) | 77 | 20 | 1.12 |
| 3.0 | 803 | 558 | 462 (83%) | 96 | 18 | 1.12 |

**83–86% of multi-option decisions are resolved by `logZ` alone**; the tuned flow
heuristic breaks only ~14% of them, and the mean tie-set size is 1.10–1.12. The
"it is really just a tuned rule set wearing a viability costume" objection is
answered for this scenario: the counted objective, not the hand-weighted
tie-break, is selecting the maneuver in the large majority of cases.

Caveat: `all_dead` fires 18–20 times per run (~2.3% of plans) — every candidate
scores LOGZ_DEAD, `Z` is flat at zero, and the choice falls through to the
heuristic with no viability signal at all. Those are exactly the moments where
the agent is already in trouble.

---

## 4. Event-count protocol — the decisive comparison

The pre-registered power standard asks for >=30 collision EVENTS per condition and
Poisson rate ratios with CIs, not proportion comparisons. Run with 10 parallel workers,
seed-paired across conditions, seeds from 20000. Total wall time 2123 s
(760 episodes, ~18900 s of simulated driving).

| condition | episodes | events | exposure (s) | s / collision | per 100 s | mean speed |
|---|---|---|---|---|---|---|
| flow-L2, v_limit 26 | 200 | 30 | 6885 | 229.5 | 0.436 | 21.72 |
| viability-L2, v_limit 26 | 160 | 37 | 5033 | 136.0 | 0.735 | 21.97 |
| flow-L2, v_limit 30 | 120 | 31 | 3644 | 117.5 | 0.851 | 22.71 |
| viability-L2, v_limit 30 | 120 | 39 | 3344 | 85.7 | 1.166 | 22.91 |

Poisson rate ratios, viability / flow (exact 95% CI):

| comparison | rate ratio | 95% CI | speed ratio | verdict |
|---|---|---|---|---|
| v_limit 26 | 1.69 | [1.01, 2.83] | 1.012 | flow safer |
| v_limit 30 | 1.37 | [0.83, 2.27] | 1.009 | not separable |
| **pooled** | **1.57** | **[1.10, 2.23]** | **1.011** | **flow safer** |

**This is the project's central claim, measured at its own pre-registered power
standard, and it comes out negative.** Replacing the flow heuristic with the
viability count makes the agent collide about 1.6x as often (110 s per collision
vs 173 s) while driving 1.1% faster. The replacement was pre-registered as the one
quantitative core of the project -- "equal survival, better flow" -- so this is the
result that matters, and the delta has the wrong sign.

What this does and does not refute:

- It does **not** refute viability counting as an objective. It refutes *this
  implementation* of it on *this* task.
- It is **not** explained away by the objective being inert. The tie-set
  instrumentation (section 3) shows logZ decides 83-86% of multi-option
  decisions with mean tie size 1.1. The counted objective is genuinely in
  control, and it is the thing losing. That combination is the informative part.
- Both arms share the same L0, L1, gap controller and target-speed computation.
  The only thing the replacement changes is **maneuver ranking**, i.e. lane
  change decisions. The regression is therefore localized to lane-change choice,
  which narrows the debugging surface considerably.

Leading suspects, in order of how much they would explain:

1. **Abstraction mismatch.** The DP counts futures of a 1 Hz, +-2 m/s lattice
   while execution is a continuous controller at 15 Hz. The agent is counting
   the futures of a system it is not running.
2. **Ego swept volume is still missing from the DP.** The design calls for a
   lane-change transition to occupy both the origin and target lane for that
   step; `viability.get_succ` only checks the destination cell, so maneuvers
   that clip a neighbour are counted as survivable. `cfg.no_sweep` gates only
   the *other* vehicles' intent sweep. This is the single most likely cause of a
   lane-change-localized regression.
3. **`all_dead` at ~2.3% of plans** -- logZ goes flat exactly when the agent is
   already in trouble, handing those moments to the fallback with no signal.

## 5. Ego swept volume implemented — re-measurement

Suspect 2 from section 4 was fixed: a lane-change transition now requires the
ORIGIN lane cell at the destination (v, x) to be alive as well, not just the
target cell (`viability.build_succ_tables` emits an origin-lane gather table,
`dp_single_world` gates lane-change contributions by it).

A second bug surfaced while wiring it: the DP applied the gate to every
transition it folded in, but `score_actions` reads `Z[t_read][succ(s0, a)]`
directly, skipping the s0 -> succ step. **The one action actually being chosen
was the one action never sweep-checked.** `sweep_gate()` in `score_actions`
closes that.

Same protocol, same seeds (20000+), same machine. The flow arm reproduces
bit-identically (9 events / 1275 s in round 1 both times), which confirms the
comparison is clean.

| condition | episodes | events | exposure (s) | s / collision | mean speed |
|---|---|---|---|---|---|
| flow-L2, v26 | 200 | 30 | 6885 | 229.5 | 21.72 |
| viability-L2, v26 | 160 | 34 | 5157 | 151.7 | 21.93 |
| flow-L2, v30 | 120 | 31 | 3644 | 117.5 | 22.71 |
| viability-L2, v30 | 120 | 36 | 3454 | 95.9 | 22.72 |

Rate ratios (exact 95% CI):

| comparison | rate ratio | 95% CI | verdict |
|---|---|---|---|
| v26, viability / flow | 1.51 | [0.90, 2.56] | not separable |
| v30, viability / flow | 1.23 | [0.74, 2.05] | not separable |
| **pooled, viability / flow — pre-fix** | **1.57** | **[1.10, 2.23]** | **flow safer** |
| **pooled, viability / flow — post-fix** | **1.40** | **[0.98, 2.01]** | **not separable** |
| pooled, post-fix / pre-fix (effect of the fix) | 0.90 | [0.64, 1.26] | not separable |

Seconds per collision, pooled: flow 172.6 | viability pre-fix 110.2 | viability
post-fix 123.0. `all_dead` also fell from ~20 to ~11 per 20-episode run.

**Honest reading.** The fix moved the point estimate in the right direction and
the flow-vs-viability comparison is no longer adverse at the 95% level. It is
**not** a win:

- The improvement itself (RR 0.90 [0.64, 1.26]) is indistinguishable from noise.
  Losing significance after a modest point-estimate shift is weak evidence, not
  a reversal — the adverse direction is unchanged.
- viability-L2 still collides more often than the heuristic it replaced (123 s
  vs 173 s per collision) at the same speed (1.01x).
- The claim "viability-L2 beats the flow heuristic" remains **unproven**. It is
  now unproven-inconclusive rather than unproven-contradicted.

Separating a residual 1.4x at 95% would need roughly 3-4x the exposure
(~100+ events per arm). That is about 2 hours on 10 workers with this harness,
and is the honest next measurement if the claim is to be settled either way.
Suspect 1 (the 1 Hz / +-2 m/s lattice vs. a 15 Hz continuous controller) is
untouched and is now the leading explanation.

## 6. Suspect 1 fixed: half-lane lattice — and it buys nothing

The grid/controller mismatch was the leading remaining explanation. The DP
modelled `LANE_LEFT` as completing inside one 1 s step; `EgoController` caps
`heading_ref` at 0.08 rad, giving ~1.8 m/s lateral at highway speed and a real
lane change of roughly 2.5–3 s. The agent was counting futures of a teleporting
body it does not have.

Fix: the lane axis is now discretised at **half-lane** resolution — 4 lane
centres plus the 3 straddle positions between them, 7 cells (`occupancy.N_CELLS`).
`LANE_LEFT` from a centre lands on a straddle cell; a second `LANE_LEFT`
completes the change. A lane change therefore costs two steps, and the swept
volume stops being a bolted-on multiply: a straddle cell is blocked whenever
*either* adjacent lane is blocked, which is what "the body occupies both lanes
during the change" means. `cfg.no_sweep` now controls whether straddle cells
union their neighbours. Verified: one car in lane 1 blocks cells {1,2,3} with
sweep on and {2} with it off; a straddle cell inherits the blocked lane's gap
(37 m) while the free lane centre stays at 1000 m.

Same protocol, same seeds. The flow arm reproduces bit-identically for the third
time (9 events / 1275 s in round 1).

| condition | episodes | events | exposure (s) | s / collision | mean speed |
|---|---|---|---|---|---|
| flow-L2, v26 | 200 | 30 | 6885 | 229.5 | 21.72 |
| viability-L2 half-lane, v26 | 160 | 35 | 5102 | 145.8 | 21.95 |
| flow-L2, v30 | 120 | 31 | 3644 | 117.5 | 22.71 |
| viability-L2 half-lane, v30 | 120 | 35 | 3488 | 99.7 | 22.66 |

Pooled rate ratios across the three viability variants (exact 95% CI):

| variant | vs flow | 95% CI | s / collision |
|---|---|---|---|
| v0 baseline | 1.57 | [1.10, 2.23] | 110.2 |
| v1 swept-volume gate | 1.40 | [0.98, 2.01] | 123.0 |
| **v2 half-lane lattice** | **1.41** | **[0.98, 2.02]** | **122.7** |
| flow-L2 reference | 1.00 | — | 172.6 |

| contrast | rate ratio | 95% CI |
|---|---|---|
| v2 / v1 (effect of fixing Suspect 1) | **1.00** | [0.71, 1.42] |
| v2 / v0 (both fixes combined) | 0.90 | [0.64, 1.26] |

**Suspect 1 is eliminated, and it was not the cause.** Making the lattice agree
with the controller's actual lane-change duration changed the collision rate by
a factor of 1.00 [0.71, 1.42] — literally nothing. The gap to the flow heuristic
sits at 1.41x, statistically indistinguishable from where the swept-volume fix
left it.

`logZ` also became *less* decisive under the half-lane lattice: 73% of
multi-option decisions (down from 81%), mean tie size 1.19 (up from 1.14),
because lane-change options now score closer together.

Keep the change anyway — the swept volume is structural instead of bolted on,
and the transition model no longer contradicts the controller. But it is an
architectural correction, not a performance one, and should be reported as such.

### Where this leaves the project

Two rounds of targeted, well-motivated fixes each moved the number by 0–10% and
the gap is stable at ~1.4x. Remaining explanation space, in order:

1. `all_dead` at ~2.3% of plans — logZ goes flat exactly when the agent is
   already in trouble.
2. The longitudinal transition model: `IDLE` holds v exactly while the
   controller tracks `v_target`; `KEEP_SPEED` is scored as `max(z_idle, z_slow)`,
   an optimistic max over two different dynamics.
3. **The objective itself.** On this task, counting surviving futures may simply
   be a worse maneuver-ranking signal than gap-based flow scoring. The
   instrumentation rules out the "it was never in control" escape: logZ decides
   73–86% of multi-option choices. It is in control, and it is losing.

Hypothesis 3 is now the one the evidence most directly supports, and it is a
publishable negative result: viability counting is a principled objective that
needs no reward design, and on dense-highway maneuver selection it does not beat
a tuned gap heuristic. The flappy/highway pair then reads as "works where the
action lattice is the real action space, does not transfer to a domain where the
executed system is continuous."

## 7. S1 Option trap — the differentiating scenario, finally run

S1 was designated as the primary differentiating evidence for the viability reading,
with the behaviours pre-registered:

> `flow-L2`, `adaptive-RSS only`: **enters** (satisfies RSS now and is faster).
> `viability-L2`: **does not enter**, because the number of futures collapses.

It had never been implemented. `evaluation/scenario_s1.py` implements it: the ego
sits in a 70 m hole in a 20 m/s lane-1 platoon; lane 0 is empty for `d_trap`
metres and therefore faster and RSS-clean right now, but ends in a jam. Beyond
the ego's hole the platoon spacing is 22 m, below the RSS return threshold.
Other vehicles are scripted at constant velocity so the trap cannot dissolve
itself. `viability_myopic` (T=1) is the control: same penalties, same RSS, no
horizon — if it refuses too, the refusal is not option value.

24 runs: `d_trap` ∈ {90, 130, 170} m × jam speed ∈ {4, 12} m/s × 4 agents.

| agent | entered | crashed | mean distance |
|---|---|---|---|
| l1-only | 6 / 6 | **6 / 6** | 180 m |
| flow-L2 | 3 / 6 | 0 / 6 | 755 m |
| **viability-L2 (T=8)** | **6 / 6** | 1 / 6 | 667 m |
| viability myopic (T=1) | 1 / 6 | 0 / 6 | 738 m |

`logZ(CHANGE_LEFT) − logZ(KEEP_SPEED)` at the moment entry was chosen:
**+0.85, +0.50, −0.04, +0.91, +1.17, +0.09** — positive in 5 of 6.

**The prediction is inverted, not merely unmet.** viability-L2 enters more
readily than anything except the no-deliberation baseline, and its objective
*actively favours* entering by up to 1.17 nats. flow-L2 is the conservative one
(refuses 3/6). The myopic control is the most conservative of all (refuses 5/6),
which means **the horizon is what causes entry** — the exact opposite of the
pre-registered mechanism.

### Why: a traffic jam is not death

Entering a lane that ends in a jam does not reduce the number of *survivable*
futures. You brake and you live. It reduces the number of *fast* futures — and
speed was deliberately removed from the objective and demoted to an ε-tie-break
(the design explicitly forbade `logZ + λ·speed` in order to protect the purity of
the objective).

So option counting is **structurally blind to congestion traps**, and the stated
rationale for S1 ("the number of futures collapses past the entry") is wrong about
the mechanics of its own objective. An empty lane genuinely has more survivable
futures — accelerate, decelerate, hold, change lane, all alive — which is why
the count prefers it, correctly by its own definition and uselessly for the task.

Entering was in fact usually right: viability beat flow on distance in 3 of 6
conditions (812 vs 799, 813 vs 804, 737 vs 724), tied in 2, and lost once.

### The one time the objective was right, the tolerance discarded it

The single viability crash is `d_trap = 170` with the hard jam — the trap sits
beyond the 512-ray sensor's 150 m reach, so it is discovered late and at close
range, where it threatens death rather than delay. There the count did turn
negative: **−0.04 nats**. But `cfg.tie_eps = 0.05`, so the candidate stayed in
the tie-set, the flow heuristic picked the empty fast lane, and the agent hit the
jam (193 m travelled vs flow's 758 m).

**The only correct trap signal the objective ever produced was smaller than the
tolerance designed to ignore it.** That is a concrete, cheap thing to fix and it
was invisible before this scenario existed.

### A note on S1 as specified

S1 was specified with the trap 200 m ahead. The project's own sensor reaches 150 m.
As written, S1 is unobservable to the agent under test, and the first version of
this scenario reproduced that: every agent entered and crashed for perception
reasons, with nothing to say about planning. The sweep therefore straddles the
sensor boundary deliberately, and the `in` / `OUT` column marks which regime each
row is in.

### What this does to the section 6 verdict

It does not overturn it, but it narrows it. The 1.4x adverse rate ratio was
measured on dense random traffic, which contains almost no traps. S1 was supposed
to be the regime where option counting earns that back. It does not — not because
the implementation is weak, but because the objective cannot represent the kind of
trap that matters on a highway. **The negative result now has a mechanism, not
just a magnitude.**

## Reproduce

```bash
python3 -m venv .venv_mac && .venv_mac/bin/pip install gymnasium highway-env numpy imageio matplotlib scipy

# section-11 gate (numpy only, ~1 min)
python3 test_viability_bias.py

# Pareto vs IDM (~25 min)
.venv_mac/bin/python evaluation/evaluate_vs_idm.py --episodes 30 \
  --agents idm,l1only,flow,viability:2.0:20,viability:2.0:23,viability:2.0:26,viability:2.0:30

# event-count protocol, 10 workers (~35 min)
.venv_mac/bin/python evaluation/evaluate_events.py --workers 10 --target-events 30 \
  --conditions flow:26,viability:26,flow:30,viability:30

# post-swept-volume re-measurement (same command, results_events_sweep.json)

# S1 option trap (~7 min)
.venv_mac/bin/python evaluation/scenario_s1.py

# rear-RSS ablation
.venv_mac/bin/python evaluation/evaluate_vs_idm.py --episodes 30 --rear-rss 0 \
  --agents viability:2.0:23,viability:2.0:30
```
