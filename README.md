# highway_env_WM

**Do SELF, DEATH and OTHER survive contact with a task?**

A successor to `flappy_WM`, which built an agent with no reward function out of three
structures — the functional form of its own dynamics (**SELF**), the absolute
irreversibility of collision (**DEATH**), and the fact that others occupy space and
drift with inertia (**OTHER**). The objective was never hand-written; it was derived
from the absorbing boundary as `log Z`, the count of survivable futures.

In Flappy Bird, *survival is the task*. Staying alive and doing well are the same
sentence. This project moves the same three elements into `highway-env`, where they
come apart: you must not only stay alive, you must **get somewhere, reasonably fast**.
That is also how organisms are structured — survival first and instinctive, efficiency
evaluated within the cases where survival already holds.

This repository is the record of what transferred, what did not, and why.

---

## Result in one paragraph

**The three elements transfer.** Self-dynamics are identified online in tens of
seconds rather than millions of samples; the lexicographic status of DEATH is
measurably load-bearing (remove the deliberation layer and the collision rate goes to
0.667 per episode); and the system drives collision-free at **97% of an omniscient
IDM/MOBIL oracle's speed** using 512 rays and an online self-model.

**What does not transfer is the task-layer ranking.** `log Z` counts *survivable*
futures, not *fast* ones, so it is blind to anything that delays you without killing
you. A dedicated trap scenario shows this directly: the objective actively *prefers*
driving into a lane that ends in a traffic jam, by up to 1.17 nats. A traffic jam is
not death. Flappy Bird has no state that is survivable-but-costly; a highway is full
of them, and that — not dimensionality or the number of other agents — is what
separates the two domains.

---

## Given vs. acquired

No claim here is interpretable without this table. The *structure* of all three
elements is injected by the designer. Only *parameters* are acquired online.

| Element | In flappy_WM | Here | Acquired online |
|---|---|---|---|
| **SELF** | 1-DOF rise/fall | 2nd-order lateral spring–damper + constant-accel longitudinal | `a_accel`, `a_decel`, `Kp`, `Kd` — Bayesian posteriors with a variance floor |
| **DEATH** | Contact with pipe/ground | Rectangle intersection, extended to RSS margin violation | nothing — pure axiom |
| **OTHER** | Static pipes | Extended bodies, drift, intent, occlusion | speeds and intent (EKF + IMM), occlusion phantoms |
| **WORLD** | (implicit) | lane width 4.0 m, road bounds, 5×2 m body | nothing — hardcoded. A fourth, unacknowledged concept |

DEATH is necessarily a priori: you cannot learn your own death without dying, and the
premise of the project is not dying. Organisms do not learn their own death either —
nociception and fear were installed by an outer loop (evolution) and operate as axioms
within an individual lifetime. The designer stands in for that outer loop.

---

## Architecture

A lexicographic stack. The safety layer **removes** candidates rather than penalizing
them: survival is a constraint, not a cost.

| Layer | Timescale | Role |
|---|---|---|
| `L0` Reflex | 1 step, deterministic | TTC < 1.5 s → emergency brake |
| `L1` Invariant | 1 step, hard guarantee | Closed-form RSS. Deletes violating maneuvers from the candidate set |
| `L1.5` Soft cost | 1 step | Pessimistic bounds, phantoms, rear RSS |
| `L2` Deliberation | T = 8 s, posterior expectation | **Ranks** the survivors — the subject of this work |
| Controller | 15 Hz continuous | Maneuver → steering and acceleration (cascaded PD) |

The world model clusters raw ray returns into object centroids, tracks them with an
EKF, infers lane-change intent, and emits occlusion phantoms at depth discontinuities.
Self-parameters carry Bayesian posteriors whose variance inflates on prediction
surprise, so a changed body is re-learned rather than assumed away.

### The viability reading of L2

```
Z[T][s] = alive(s, T)
Z[t][s] = alive(s, t) · exp(−β · pain(s, t)) · Σ_a Z[t+1][succ(s, a)]

score(a) = (1 − w) · min_k logZ_k  +  w · mean_k logZ_k
```

This is a **counting relaxation of the viability kernel**. Classical viability theory
asks whether a surviving trajectory *exists* (a set-membership test); this counts how
many there are. That single relaxation yields a gradient toward the interior of the
safe set for free, which an indicator function cannot provide. It is the most
defensible technical claim in the project.

The world ensemble samples the ego's braking authority **only in the bad direction**
(μ, μ−σ, μ−2σ) plus the top-2 IMM intent hypotheses of other vehicles, aggregated
maximin with a 15% mean term so that a wide posterior does not flatten the score.

The lane axis is discretised at **half-lane resolution** (4 lane centres + 3 straddle
cells). A lane change therefore costs two steps, matching the controller's real
~2.5–3 s maneuver, and the swept volume becomes structural: a straddle cell is blocked
whenever *either* adjacent lane is blocked.

---

## OTHER is read twice

This is the conceptual centre of the project.

In `flappy_WM`, OTHER was a static pipe — a source of death and nothing else. On a
highway it does not reduce to that.

| Reading | OTHER is… | Derived signal |
|---|---|---|
| **viability** | occupancy — space you die in | the number of survivable futures (`log Z`) |
| **flow** | a medium — traffic has a speed | progress potential (lane flow speed, gap equilibrium) |

**Both signals come from OTHER, not from a reward function.** The flow reading is
routinely dismissed as "a tuned heuristic", but it is no less derived from the three
elements than the count is. The difference is *which aspect* of OTHER is read.
`flappy_WM` needed only the first reading because pipes do not flow.

This reframing changes what the measurements mean. Pit the two readings against each
other as task-layer rankers and flow wins — but *a signal that reads progress winning
on a task whose objective is progress* is close to a tautology. The surprising thing is
not that viability loses; it is how close it gets.

---

## Results

All numbers reproduce from this checkout. Full tables, confidence intervals and
methodology are in [`evaluation/RESULTS.md`](evaluation/RESULTS.md).
Total measured: **2,304 episodes**, roughly 57,000 s of simulated driving.

### Pareto frontier against an omniscient oracle

IDM/MOBIL is not a peer. It reads exact positions and velocities of every vehicle
from the simulator, has no sensing pipeline and no self-model uncertainty, and is the
same model driving surrounding traffic — it is effectively coordinating with copies of
itself. Parity is the wrong bar, so a **threshold** was pre-registered before running:

> PASS if `collision_rate ≤ IDM + 0.05` **and** `mean_speed ≥ 0.70 × IDM`.

30 episodes each, seed-paired, Wilson 95% CI:

| Agent | `v_limit` | Collisions/ep | 95% CI | Mean speed | Retention | |
|---|---|---|---|---|---|---|
| IDM/MOBIL (oracle) | — | 0.000 | 0.00–0.11 | 20.40 | 1.00 | reference |
| L1 only (no deliberation) | 30 | 0.667 | 0.49–0.81 | 25.50 | 1.25 | fail |
| flow reading | 30 | 0.200 | 0.10–0.37 | 22.86 | 1.12 | fail |
| viability reading | 30 | 0.333 | 0.19–0.51 | 22.89 | 1.12 | fail |
| viability reading | 26 | 0.133 | 0.05–0.30 | 21.80 | 1.07 | fail |
| viability reading | 23 | 0.067 | 0.02–0.21 | 20.99 | 1.03 | fail |
| **viability reading** | **20** | **0.000** | **0.00–0.11** | **19.87** | **0.97** | **PASS** |

The frontier is real and monotone — the agent is not a crawler, it trades safety for
speed on a clean curve. Note that `k_factor` (the pessimism coefficient) is a **dead
knob**: k = 1, 2, 3 all give exactly 0.333. The axis that moves the frontier is the
speed cap.

Passing at `v_limit = 20` is provisional: with **zero** observed events the CI upper
bound is still 0.11.

### Event-count comparison of the two readings

Episode-rate comparisons are underpowered. Each condition was driven until ≥30
collision *events* were observed, then compared as a Poisson rate ratio with an exact
CI, seed-paired:

| Ranking signal | Events | Exposure | s / collision | Mean speed | Rate ratio | 95% CI |
|---|---|---|---|---|---|---|
| flow | 61 | 10,529 s | 172.6 | 22.1 | 1.00 | reference |
| viability | 70 | 8,590 s | 122.7 | 22.3 | 1.41 | 0.98–2.02 |

About 1.4× more collisions for 1.1% more speed. The interval straddles 1.0, so this
is not separable at 95%.

### Is the objective actually deciding?

The obvious escape — "`log Z` was inert and the tie-break did all the work" — is closed
by instrumentation. Across the runs, **`log Z` alone decides 73–86% of multi-option
decisions**, with a mean tie-set size of 1.10–1.19. The objective is holding the wheel.

One exception matters: in ~2.3% of plans **every candidate scores as dead**, `Z` goes
flat at zero and the choice falls through with no signal. That looks like a defect and
is arguably a feature — see below.

### Body swap — online identification is load-bearing

Sections above isolate the ranking layer by pinning the self-model to ground truth.
`evaluation/body_swap.py` unpins it and changes the ego's actual physical authority
(`acceleration_range`), then compares three arms seed-paired over 360 episodes: an
**adaptive** agent that starts from `mu = −10, sigma = 10` and actively probes an empty
road until `sigma < 0.5`; a **pinned-nominal** agent that believes it still has its old
brakes; and a **pinned-oracle** agent told the truth.

Active calibration recovers the true braking authority in **6 steps** — identified
2.52 / 3.52 / 5.01 / 7.01 against true 2.5 / 3.5 / 5.0 / 7.0.

Pooled over the three swapped bodies (n = 90 each):

| arm | collisions | rate | mean speed |
|---|---|---|---|
| adaptive | 18 / 90 | 0.200 | 21.07 |
| pinned-nominal | 24 / 90 | 0.267 | 21.69 |
| pinned-oracle | 17 / 90 | 0.189 | 21.07 |

McNemar on matched (body, seed) pairs: **adaptive crashed alone 0 times,
pinned-nominal crashed alone 6 times, exact p = 0.031**, at a cost of 0.61 m/s.
Against the oracle the difference is 1 vs 0 discordant pairs, p = 1.000 — six seconds
of probing is indistinguishable from being told the answer.

The conservatism is graded by how wrong the body is. Speed relative to the
pinned-nominal agent: **−1.02 m/s** at half braking authority, −0.89 at 3.5, −0.11 at
the nominal body where there is nothing to discover, +0.07 at 7.0. Nothing in the code
says "if the brakes are weak, drive slower"; it falls out of identifying `a_decel`,
which grows the RSS distance, which grows the target gap, which lowers target speed.

One precision: the conservatism arrives through the **mean**, not the variance. The
ensemble spread is wider for the adaptive arm but does not track degradation (0.74 at
|a| = 2.5, 0.93 at |a| = 7.0), so the "surprise inflates sigma → pessimism → caution"
pathway remains unvalidated.

### S1 — the Option trap

Aggregate benchmarks cannot separate the two readings because dense random traffic
contains almost no traps. [`evaluation/scenario_s1.py`](evaluation/scenario_s1.py)
builds one: the left lane is empty, faster and RSS-clean *now*, but ends in a jam
`d_trap` metres ahead; outside the ego's own hole the platoon spacing is below the RSS
return threshold, so entering is hard to undo. Other vehicles are scripted at constant
velocity so the trap cannot dissolve itself.

The pre-registered prediction was: **flow enters, viability refuses**, because the
number of futures collapses past the entry. `viability_myopic` (T = 1) is the control —
same penalties, same RSS checks, no horizon.

24 runs, `d_trap` ∈ {90, 130, 170} m × jam speed ∈ {4, 12} m/s:

| Ranking signal | Entered | Crashed | Mean distance |
|---|---|---|---|
| L1 only | 6 / 6 | **6 / 6** | 180 m |
| flow | 3 / 6 | 0 / 6 | 755 m |
| **viability (T = 8)** | **6 / 6** | 1 / 6 | 667 m |
| viability myopic (T = 1) | 1 / 6 | 0 / 6 | 738 m |

`logZ(enter left) − logZ(keep lane)` at the moment entry was chosen:
**+0.85, +0.50, −0.04, +0.91, +1.17, +0.09** — positive in five of six.

**The prediction is inverted, not merely unmet.** The viability reading enters more
readily than anything except the no-deliberation baseline; flow is the conservative one;
the *most* conservative is the control with the horizon removed. The horizon is what
causes entry.

Why: entering a lane that ends in a jam does not reduce the number of **survivable**
futures. You brake and you live. It reduces the number of **fast** futures — and speed
was deliberately removed from the objective and demoted to an ε-tie-break. The count is
blind to congestion traps by construction.

The single viability crash is the one case where the objective was right. With the trap
at 170 m — beyond the 512-ray sensor's 150 m reach — it is discovered late and at close
range, where it threatens death rather than delay, and the score did turn negative:
**−0.04 nats**. But `tie_eps = 0.05`, so the candidate stayed in the tie-set and the
heuristic picked the empty lane. *The only correct trap signal the objective ever
produced was smaller than the tolerance designed to ignore it.*

---

## What we conclude: where `log Z` belongs

Taken together the measurements point one way. `log Z` is not a task-layer ranking
function; it is a **margin signal belonging to the death layer**.

```
transplanted                          what the evidence supports

  L0  reflex                            L0  reflex
  L1  RSS hard filter                   L1  RSS hard filter
  L2  logZ + flow tie-break    ──►          + logZ margin / alarm
  controller                            L2  flow ranking
                                        controller
```

`flappy_WM` never needed this distinction because it had no task layer. Adding one
reveals what `log Z` actually measures: **survival margin**, not progress efficiency.

In that position it does things flow can never do. It reports, in a unit (nats, the log
count of survivable futures), *how many ways out the model still has*. The `all_dead`
2.3% becomes a feature — the system saying out loud "there is no surviving path in my
model", which a gap score has no way to express. And because the ensemble is built from
the `a_decel` posterior, degraded braking shrinks the option count **before** any
collision occurs; hand-tuned constants have no such path.

---

## Limitations

**Measurement.** The passing configuration rests on *zero* observed events. The 1.4×
gap between the two readings is not separable at 95% and would need 3–4× the exposure
to settle. Surrounding vehicles yield to the ego, so some of the measured survival is
other drivers' goodwill. Sections 1–7 of `RESULTS.md` pin the self-model priors to
ground truth to isolate the ranking layer, which also leaves the pessimism ensemble
nearly degenerate (a 4–13% spread) — so every number there was produced with the
robust/adaptive machinery effectively suppressed. The body-swap experiment unpins it
(see below); the variance/pessimism pathway specifically is still unvalidated.

**Structure.** Ego pose and velocity are read noiselessly from the simulator, and lane
width, road bounds and body dimensions are hardcoded — the "512 rays only" framing is
not supported by the code. Space beyond sensor range is counted as free, so a trap
outside 150 m is unavoidable for any ranking function. Deliberation runs on a 1 Hz
lattice while execution is a 15 Hz continuous controller; correcting the lane-change
duration in the lattice did not move performance, so this mismatch is real but was not
the dominant factor.

**In principle.** That the agent *acquired* the three concepts, that it *understands*
death, and that there is no reward — all three are unprovable. Everything was injected,
not dying is the premise, and a lexicographic order is itself a non-Archimedean utility
function. The three elements are not sufficient for driving; the contribution here is
locating the precise point at which they stop being sufficient.

---

## Repository layout

| Path | Role |
|---|---|
| `main.py` | Active calibration, then a dense-traffic run; writes a video |
| `world_model.py` | Bayesian scalars, EKF/IMM multi-object tracking, occlusion phantoms |
| `ray_sensor.py` | 512-ray lidar with spatial filtering |
| `occupancy.py` | Predicted occupancy field on the half-lane lattice |
| `viability.py` | Exact backward DP, `log Z`, world-ensemble aggregation |
| `planner.py` | L0 / L1 / L1.5 / L2; both readings of OTHER |
| `controller.py` | Maneuvers → continuous action |
| `config.py` | `ViabilityCfg` and the ablation switches |
| `test_viability_bias.py` | Gap-position bias measurement (actuation-asymmetry probe) |
| `evaluation/evaluate_vs_idm.py` | Pareto frontier against the oracle |
| `evaluation/evaluate_events.py` | Event-count protocol, parallel workers |
| `evaluation/scenario_s1.py` | Option trap |
| `evaluation/RESULTS.md` | **All measured numbers, with methodology** |
| `models.py`, `lexicographic_planner.py` | Legacy, superseded and unused |

---

## Reproducing

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

```bash
# calibration + a dense-traffic run, renders Viability_Driving.mp4
.venv/bin/python main.py

# gap-bias probe (numpy only, ~1 min)
.venv/bin/python test_viability_bias.py

# Pareto vs the IDM oracle (~25 min)
.venv/bin/python evaluation/evaluate_vs_idm.py --episodes 30 \
  --agents idm,l1only,flow,viability:2.0:20,viability:2.0:23,viability:2.0:26,viability:2.0:30

# event-count protocol, 10 parallel workers (~35 min)
.venv/bin/python evaluation/evaluate_events.py --workers 10 --target-events 30 \
  --conditions flow:26,viability:26,flow:30,viability:30

# S1 option trap (~7 min)
.venv/bin/python evaluation/scenario_s1.py
```

Ablations are switches on `ViabilityCfg`: `no_option_value` (T = 1), `no_pessimism`
(K = 1), `no_pain_gradient` (β = 0), `no_memory`, `no_sweep`, `rear_rss`, `legacy_L2`
(the flow reading).

---

## Related work

This is not "neither RL nor rules"; it is an intersection, and saying so makes the
claim stronger.

- **Viability theory** (Aubin 1991) — direct ancestor. `Z` is a counting relaxation of
  the viability kernel indicator. The gradient it yields is a gradient of *survival
  margin*, not of task performance; this repository measures that distinction.
- **HJ reachability** (Mitchell & Tomlin; Bansal et al. 2017) — the lattice DP is a
  discrete counting version, and the world ensemble is robust reachability. That this
  literature uses the kernel **as a safety filter only** and writes the performance
  objective separately turns out to be the right division of labour, not a shortcut.
- **Control as inference** (Todorov 2007; Toussaint 2009; Levine 2018) — `Z` is
  literally a partition function; this is soft value iteration with
  `r(s) = log alive − β·pain` and a uniform action prior. "No reward" is precisely
  "the reward is pinned to a 0/1 survival indicator", and that pinning is exactly why
  it cannot see time.
- **Empowerment** (Klyubin et al. 2005; Mohamed & Rezende 2015) — the closest prior
  work on reward-free option-preserving behaviour. S1 reproduces its known failure mode
  concretely: the empty lane has high empowerment and is far from the destination.
- **RSS** (Shalev-Shwartz et al. 2017) — used unmodified in L1. Defining DEATH as
  "irrecoverable margin violation" rather than "contact" is what makes the absorbing
  boundary propagate a useful distance backwards.
- **Shielding / CBF / Simplex** (Alshiekh et al. 2018; Ames et al.) — "remove, don't
  penalize" is exactly shielding. The engineering counterpart of the biological
  two-layer structure already exists as a standard.
- **Pessimistic model-based control** (PETS 2018; MOReL, MOPO 2020) — sigma-point
  ensembles with min aggregation. Wiring self-ignorance directly to behavioural
  conservatism is a validated pattern, and is why the SELF layer worked best here.
- **Occlusion-aware planning** (Orzechowski et al. 2018) — the phantom mechanism is a
  reinvention of a named technique.
- **Active inference** (Friston) — expected free energy's risk/ambiguity split maps onto
  the alive/pain split.
- **Dual control** (Feldbaum; persistent excitation) — the proper name for active
  calibration. Sample efficiency against RL is real, but not a differentiator against
  the non-learning baseline, which also uses zero samples.

**Where this sits on its own:** using the viability kernel as a *count* rather than a
membership test, evaluating that count over a self-model posterior, and separating —
inside one system — its failure as a task ranker from its success as a margin signal.
The negative result comes with a mechanism, which is the contribution.

---

## Next

1. **Re-place `log Z` as a margin signal.** Take it out of the ranking slot, wire it to
   alarms and monitoring, and measure its predictive power for incidents under the same
   protocol.
2. **Push the body swap further.** Identification is now shown to earn its keep for
   `a_decel` (below). The variance/pessimism pathway is not: make `k_factor` a live
   knob by starting from genuinely wide posteriors and re-measure.
3. **A domain where traps are death.** A jam is not death, but a joint limit, a fuel
   floor and a dead-end corridor are. This work specified the boundary condition; the
   inside of that boundary is still untested.

---

## License

MIT — see [`LICENSE`](LICENSE).
