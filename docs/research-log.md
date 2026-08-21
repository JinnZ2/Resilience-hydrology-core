# Research Log

How claims in this repository are made, tested, and revised.

## The loop

```
hypothesize  →  run  →  compare result to claim
                              │
              ┌───────────────┴───────────────┐
        result supports                 result falsifies
              │                               │
     mark SUPPORTED,                  edit the claim to
     state what would                 match the result,
     falsify it next                  archive the old wording
              │                               │
              └───────────────┬───────────────┘
                              ▼
                   list the unknowns this
                   run exposed  →  rerun
```

Two rules make the loop work:

1. **The old claim is never deleted.** It is archived in [`legacy/`](../legacy/)
   with its original date, so a revision reads as a revision and priority stays
   with whoever wrote it first. See [`legacy/README.md`](../legacy/README.md).
2. **A claim in the working tree names its status.** SUPPORTED, FALSIFIED,
   REVISED, or UNTESTED — and if it is a model output, the command that
   produces it.

Status vocabulary used below:

| Status | Meaning |
|---|---|
| SUPPORTED | The run agrees with the claim. Not proof — just not yet falsified. |
| FALSIFIED | The run contradicts the claim. Claim revised, old wording archived. |
| REVISED | Claim rewritten to match what the code actually shows. |
| UNTESTED | Asserted, never run. No evidence either way. |
| OPEN | A question this round exposed and could not answer. |

**A model run is not evidence about the world.** Every entry below tests a claim
against *this repository's code*. Agreement means the documentation is honest
about the model. It says nothing about whether the model matches the atmosphere —
that takes field data, which is the largest open item here (see O1).

---

## Round 1 — 2026-08-15: documentation audit against code

All three simulations were run and their outputs compared to what the README and
docs asserted. Environment: Python 3, numpy/matplotlib/scipy per
[`requirements.txt`](../requirements.txt).

### H1 — "Output: 0.034-0.14 mm/day depending on conditions and energy available"

- **Source**: root README, first written 2025-12-07; archived verbatim at
  [`legacy/2025-original/README.md`](../legacy/2025-original/README.md).
- **Prediction**: `01_basic_dew.py` produces daily rates inside 0.034–0.14 mm/day.
- **Run**: `python simulations/01_basic_dew.py --climate {arid,semi_arid,mediterranean,tropical_dry}`
- **Result**:

  | Climate | System OFF (mm/day) | System ON (mm/day) |
  |---|---|---|
  | arid | 0.1000 | 0.3000 |
  | semi_arid | 0.0910 | 0.2730 |
  | mediterranean | 0.0540 | 0.1620 |
  | tropical_dry | 0.0800 | 0.2400 |

- **Verdict**: **FALSIFIED.** The model's OFF range is 0.054–0.100 and its ON
  range is 0.162–0.300. Neither bracket is 0.034–0.14. The claimed range is not
  reproducible from any code in this repository at any preset.
- **Where the numbers came from**: `0.034` appears in
  [`02_crop_response.py`](../simulations/02_crop_response.py) as the assumed
  system water input. The origin of `0.14` was not found — logged as O2.
  **O2 is now closed, by [`method-log.md`](method-log.md) M-01**, written
  independently and merged on 2026-08-16: both figures are printed outputs of an
  ion-coupling PDE model that was never committed as code, surviving only in
  [`legacy/2025-original/firmware__02_crop_response.md`](../legacy/2025-original/firmware__02_crop_response.md)
  — 0.034 mm/day for natural gradient coupling, 0.14 mm/day for active ion
  injection at 5.8 kWh/day. The number outlived its model and was re-attached to
  unrelated code. Credit to M-01; that entry has precedence on this finding.
- **Revised claim**: the README now states the model's ON range,
  0.16–0.30 mm/day, labels it as model output rather than measurement, and names
  the command that reproduces it.

### H2 — "we amplify the natural process... system amplifies natural process 2-4x"

- **Source**: README and the `DewSimulator` docstring in
  [`01_basic_dew.py`](../simulations/01_basic_dew.py).
- **Prediction**: amplification is a computed consequence of the modelled physics.
- **Run**: read `DewSimulator.simulate_night`; compare the improvement figure
  across all four climates.
- **Result**: `amplification = 3.0 if system_on else 1.0` — a hard-coded
  constant. Every climate reports "Improvement: +200%", exactly, because 3.0 is
  an input, not a result. The same constant is in the 2025-12-07 original
  ([`legacy/2025-original/simulations__01_basic_dew_simulation.py`](../legacy/2025-original/simulations__01_basic_dew_simulation.py))
  and was carried through the 2026-03 rewrite unchanged.
- **Verdict**: **FALSIFIED as a result; retained as an assumption.** The
  simulation cannot test the central hypothesis of the project, because the
  hypothesis is one of its inputs. The 3x figure has no derivation and no
  measurement behind it in this repository.
- **Revised claim**: README and simulation docs now state that 3x is an assumed
  amplification factor and that the ON/OFF comparison is a statement about that
  assumption, not evidence for it. → **O1, O3**

### H3 — "0.034 mm/day of atmospheric water materially helps crops through drought"

- **Source**: README ("Crop yield impact during drought"), `02_crop_response.py`
  chart subtitle.
- **Prediction**: yields rise meaningfully with system water during a 60-day drought.
- **Run**: `python simulations/02_crop_response.py`, then a sweep of
  `system_water_mm_day` from 0.034 to 3.0.
- **Result** (after the H6 fix below): wheat 47.4% → 47.5%, olive 90.5% → 91.2%,
  tomato 26.1% → 26.2%. All gains are under one percentage point. The sweep:

  | System water (mm/day) | wheat | olive | tomato |
  |---|---|---|---|
  | 0.000 | 0.474 | 0.905 | 0.261 |
  | 0.034 | 0.475 | 0.912 | 0.262 |
  | 0.140 | 0.480 | 0.932 | 0.265 |
  | 1.000 | 0.530 | 0.998 | 0.302 |
  | 2.000 | 0.690 | 1.000 | 0.469 |
  | 3.000 | 0.871 | 1.000 | 0.689 |

- **Verdict**: **FALSIFIED at the claimed rate.** Crop demand in the model runs
  1.5–4.5 mm/day. At 0.034 mm/day the system supplies roughly 1% of demand, and
  the model responds accordingly. Material yield rescue needs 2–3 mm/day —
  **60–90x** the claimed output.
- **Revised claim**: the docs now state the honest version: at modelled output
  rates the effect on seasonal yield is under one percentage point, and the
  interesting question is survival/supplemental use, not yield rescue. → **O4**

### H4 — "Physics models validated", "Hardware is proven"

- **Source**: root README status list, 2025-12-07.
- **Prediction**: if the models are "validated" and the hardware "proven", the
  repository contains at least one comparison of a model output against a
  measurement, with the conditions of that measurement recorded.
- **Run**: searched the repository for any comparison of model output against
  measurement.
- **Result**: none exists. The only field data in the repository is
  [`docs/trailer-build.md`](trailer-build.md): two nights (85 ml, 110 ml), a
  "[data...]" placeholder, and a stated 95 ml/night average over 7 days — with
  notes that condensation froze on night 4 and the battery died on day 6. The
  average cannot be derived from the data shown, and the failure notes imply
  fewer than 7 usable nights.
- **Verdict**: **FALSIFIED.** "Validated" and "proven" describe a comparison
  that has not been performed. There is no unit conversion in the repository
  between ml/night from a collector of unstated area and mm/day per m² from the
  model, so the one dataset cannot currently be compared to the one model.
- **Revised claim**: status is now stated as: models run and are internally
  consistent; they are unvalidated against field data. The trailer results are
  marked as a partial single-site log. → **O1, O5**

### H5 — "Finds the optimal 40-bit seed for each climate zone"

- **Source**: [`simulations/README.md`](../simulations/README.md), CLAUDE.md,
  root README ("find optimal seeds for your climate").
- **Prediction**: different climates yield different optimal seeds.
- **Run**: `python simulations/03_seed_optimization.py`; repeated runs for one
  climate; direct evaluation of the objective at fixed seeds.
- **Result**: every climate returns amplification bytes `[0, 0, 0, ...]` — the
  minimum of the search range, i.e. *the optimum is the system turned all the
  way down*. Bytes 3 and 4 come back as different numbers on every run
  (`[0,0,0,76,91]`, `[0,0,0,109,187]`, `[0,0,0,220,44]`) at an identical score
  of 0.3500.
- **Cause**: `evaluate_seed` never reads bytes 3 and 4 — `wavelength` and
  `crop_bias` are decoded and then unused, so 2 of the 5 bytes are unconstrained
  noise. For the other three, the objective is monotonically decreasing: the
  precipitation gain per unit amplification is `weight * RH * delta_T * 0.01`
  (at most 0.020 under any preset), while the energy penalty is 0.05 per unit
  and the safety penalty a further 0.05 on the largest amplification. Gain never
  exceeds cost, so the minimum always wins. Reaching a break-even for the `amp_pH`
  term would need `RH * delta_T > 12.5`; the highest any preset reaches is 5.0.
- **Verdict**: **FALSIFIED.** The optimizer is not finding climate-specific
  seeds. It is reporting a corner solution plus two random numbers, and it
  reports "optimal" with no indication that the answer is degenerate.
- **Revised claim**: the docs now describe this file as a seed-search *scaffold*
  with a known-degenerate objective, and state that its published seeds should
  not be used. **The weights were not silently retuned** — inventing weights that
  produce an interior optimum would manufacture the desired answer. Fixing this
  needs a defensible cost model. → **O6**

### H6 — "stress_tolerance: higher = more tolerant"

- **Source**: inline comment in `CropWaterModel.CROPS`
  ([`02_crop_response.py`](../simulations/02_crop_response.py)), 2025-12-07.
- **Prediction**: raising `stress_tolerance` raises yield at equal stress.
- **Run**: held `avg_stress` fixed and varied the parameter; then ran wheat with
  its tolerance set to 0.7, 1.5, 2.0.
- **Result**: at `avg_stress = 0.651`, wheat yielded 0.459 at tolerance 0.7,
  0.249 at 1.5, and 0.193 at 2.0. Higher tolerance produced *lower* yield. In the
  full comparison this put "very tolerant" olive and "sensitive" tomato in
  roughly the same band, and it is why the pre-fix run reported tomato (46.1%)
  outranking wheat (24.8%).
- **Cause**: `yield_reduction = avg_stress ** (1.0 / tolerance)`. Since
  `avg_stress` is in [0, 1], a *larger* exponent gives a *smaller* reduction, so
  the reciprocal inverted the parameter's meaning.
- **Verdict**: **FALSIFIED — code defect, fixed.** Changed to
  `avg_stress ** tolerance`.
- **Rerun** (`python simulations/02_crop_response.py`):

  | Crop | Yield before fix | Yield after fix |
  |---|---|---|
  | wheat (tolerance 1.5) | 24.8% | 47.4% |
  | olive (tolerance 2.0, "very tolerant") | 44.5% | 90.5% |
  | tomato (tolerance 0.7, "sensitive") | 46.1% | 26.1% |

  The crop ordering now matches the labels: olive > wheat > tomato. This is a
  sanity check on the fix, not evidence that the numbers are right — the
  absolute yields remain unvalidated. All H3 figures above are post-fix.

---

## Round 2 — 2026-08-15: constrained variable search

Round 1 ended with "which variables would most change the answer?" unanswered.
[`simulations/04_variable_search.py`](../simulations/04_variable_search.py) was
written to ask it: sample a constrained variable space, evaluate each sample,
and report the *range* the best outcomes occupy rather than a single optimum —
with an explicit check for whether that range is really just a bound.

Doing that required a model with somewhere for ecological variables to act.
`01_basic_dew.py` computes `RH * delta_T * 0.02 * amplification`, so correlating
anything against it can only rediscover its own two inputs. The new file uses a
surface energy balance instead (radiative loss + active cooling = convective
gain + latent release + conduction), solved for surface temperature, with
condensation from the vapour-pressure gradient. Wind, cloud, canopy openness,
tilt, emissivity and insulation enter physically.

Both new models remain unvalidated. The energy balance is structurally more
defensible than the linear formula, which is not the same as being right.

### H7 — dew forms at the climate presets this repository ships

- **Source**: `CLIMATES` in [`01_basic_dew.py`](../simulations/01_basic_dew.py)
  and [`03_seed_optimization.py`](../simulations/03_seed_optimization.py) —
  RH 0.25 (arid), 0.35 (semi-arid), 0.45 (mediterranean), 0.40 (tropical dry);
  first written 2025-12-07.
- **Prediction**: an energy-balance model run at those humidities produces dew.
- **Run**: `python simulations/04_variable_search.py --climate repo_daytime_rh --samples 2000`
- **Result**: **zero condensing samples out of 534 feasible.** Radiative cooling
  reached 3.2 K below air temperature on average (best case 10.9 K), against a
  dew-point depression of 12–18 K at those humidities. The surface never reached
  the dew point, so no water forms at any tilt, emissivity, or insulation.
- **Verdict**: **FALSIFIED — two distinct defects.**
  1. **No dew-point check.** `01_basic_dew.py` returns 0.09–0.30 mm/day at these
     same humidities because its formula multiplies RH by the diurnal
     temperature range and never asks whether condensation is thermodynamically
     possible. It reports water forming under conditions where it cannot.
  2. **Ambiguous humidity.** The presets carry one RH per climate with no time
     of day attached. Dew is governed by near-surface RH in the hours before
     dawn, which is far higher than the daytime value at the same site — the air
     cools toward its dew point overnight while absolute humidity changes
     little. Whichever was meant, the same number cannot serve both roles.
- **Revised claim**: the new file documents pre-dawn RH windows separately from
  the repository's existing presets and states why they differ. The old presets
  are kept and reachable as `--climate repo_daytime_rh` specifically to
  reproduce this null result. → **O8**

### H8 — "the system amplifies the natural process 3x"

Round 1 (H2) established the 3x factor is assumed rather than derived. This
round asks the quantitative follow-up: *could* the field hardware deliver it?

- **Prediction**: if 3x is achievable, the field build's energy budget buys
  enough cooling to triple the passive yield on a favourable night.
- **Run**: swept active cooling power on a favourable semi-arid night
  (291 K, RH 0.80, light wind, clear sky, tilt 30°, COP 0.7), via
  `python simulations/05_transition_paths.py --site semi_arid_summer`
  and direct evaluation of `DewEnergyBalance` across `electrical_w_m2`.
- **Result**:

  | Cooling delivered (W/m²) | Electrical (W/m²) | Yield (mm) | vs passive |
  |---|---|---|---|
  | 0 | 0.0 | 0.115 | 1.00x |
  | 2 | 2.9 | 0.129 | 1.11x |
  | 5 | 7.1 | 0.148 | 1.28x |
  | 20 | 28.6 | 0.246 | 2.13x |
  | 40 | 57.1 | 0.373 | 3.23x |

  Radiative cooling at equilibrium on that night is **35.3 W/m²**. The trailer
  build's energy budget — an 18650 cell and a 5 W panel across roughly 0.25 m²
  of collector, about 3 W/m² electrical over a 12-hour night — buys 2.1 W/m² of
  cooling at COP 0.7. That is **6% of the radiative term**, worth **1.11x**.
- **Verdict**: **FALSIFIED at the field hardware's energy budget.** Reaching 3x
  needs roughly 40 W/m² of cooling, about 57 W/m² electrical — **19x the power
  the field build has**. The Peltier is not competing with the radiative term;
  it is a rounding error on it.
- **Consistent with the search**: across 6,000 samples, `electrical_w_m2` ranked
  **last** of twelve variables (S1 0.005) and `cop_cooling` second-last (0.007).
  Within a realistic energy budget, the active-cooling lever is inert.
- **Revised claim**: "system ON vs OFF" cannot mean Peltier cooling at this power
  budget. What the passive design does — tilt, siting, surface, insulation — is
  where the available leverage actually is. → **O11**

### Findings — what the search says to do instead

Not falsifications; model-derived leads. From
`--climate semi_arid --samples 6000 --condensing-only` (1,072 feasible samples;
ordering reproduced on `--climate arid`, and identical across repeated runs at
the default seed):

*(Regenerated after the Round 3 collection-efficiency correction; see that
round for what changed and why.)*

| Variable | Kind | S1 | Best range | Note |
|---|---|---|---|---|
| rh | climate | 0.155 | 0.85–0.89 | at upper bound |
| tilt_deg | design | 0.098 | 18–53° | interior optimum |
| local_vapor_boost | siting | 0.092 | 0.08–0.14 | at upper bound |
| sky_view_factor | siting | 0.084 | 0.69–0.97 | at upper bound |
| cloud_cover | climate | 0.057 | 0.02–0.35 | at lower bound |
| wind_speed | climate | 0.024 | 0.8–4.4 m/s | interior optimum |
| electrical_w_m2 | control | 0.005 | — | no constraint |

1. **Tilt is the largest design lever and no build guide specifies it.** It
   trades drainage against sky view — steeper sheds droplets into the collector
   but sees less cold sky — giving a genuine interior optimum rather than a
   bound. Neither [`build-guide.md`](build-guide.md) nor
   [`trailer-build.md`](trailer-build.md) mentions collector angle. → **O9**
2. **The ecological variables are real levers.** Canopy openness
   (`sky_view_factor`) and upwind soil/plant moisture (`local_vapor_boost`)
   rank third and fourth, above every hardware variable except tilt. Siting and
   land management appear to matter more than the electronics.
3. **Wind is non-monotonic**, as it should be — it feeds vapour to the surface
   and warms the surface at the same time, so both calm and windy nights
   underperform. This behaviour was not built in; it emerges from the two
   channels competing, which is mild evidence the balance model is structured
   sensibly.
4. **Humidity dominates whether dew happens at all**, and it is a corner
   solution: more is always better, all the way to the bound. Only 18% of
   samples condensed at all. The design variables only matter on nights that
   condense — which is why the script separates the two questions.
5. **Three of the top four are pinned at bounds this project invented.** For
   `local_vapor_boost` the bound is entirely arbitrary — the 0.15 cap was chosen
   for lack of any measurement. The search says "as much as you'll allow",
   which is a statement about the box, not the world. → **O10**

### Measurement priorities

The script ranks the variables with high leverage that have never been measured.
This is the concrete form of O1:

1. **`rh`** (S1 0.140) — the field build has *no humidity sensor at all*, and
   humidity is the single largest driver of whether dew forms. A DS18B20 pair
   cannot answer this. Adding one sensor changes more than any other measurement.
2. **`local_vapor_boost`** (S1 0.086) — an entirely assumed channel with no
   measurement anywhere.
3. **`sky_view_factor`** (S1 0.078) — costs one upward photo per site to record.
4. **`cloud_cover`** (S1 0.052) — recoverable retrospectively from weather
   records for the November 2025 nights.
5. **`wind_speed`** (S1 0.023) — non-monotonic, so a value is needed, not a bound.

---

## Round 3 — 2026-08-15: transition analysis, and the first model-vs-measurement test

Round 2 said what matters. It did not say what to do about the collector already
sitting in a field, which is a different question — a deployed unit has sunk
costs, an owner, and a budget.
[`simulations/05_transition_paths.py`](../simulations/05_transition_paths.py)
asks it: given the build as deployed and a catalogue of modifications with real
costs, what is the cheapest ordered set of changes, and which of them survive
our own uncertainty?

Every modification is scored across a Monte Carlo that varies **both the weather
and our own [ASSUMED] coefficients**. Ranking by a single predicted yield would
launder O10's uncertainty into false confidence; what gets reported instead is
how often a change helped across draws.

### Model correction found while building this

`04_variable_search.py` had collection efficiency as
`EFF_MAX * (1 - exp(-tilt/TILT_CHAR))`, which is exactly **zero at zero tilt**.
A flat plate is not a zero-yield object — dew forms on it and some reaches the
vessel; it just drains badly. The consequence was silent and serious: every
configuration that had not been angled produced exactly nothing, including the
deployed build, whose tilt was never recorded. Added `EFF_MIN = 0.15`
[ASSUMED].

**Round 2 was re-run after the fix and its top two swapped**: humidity moves
from second to first (S1 0.140 → 0.155) and tilt from first to second
(0.152 → 0.098). The Round 2 table above carries the corrected figures. Tilt
remains the largest *design* lever — the variable a builder controls — which is
what the build guidance rests on, so no recommendation changes. But the earlier
claim that tilt was the single biggest driver overall was an artifact of the
zero-at-flat bug inflating the gap between angled and unangled collectors.

### H9 — can the model reproduce the only measurement this project has?

The first time a model output and a field measurement have been put side by
side. This is what O1 has been asking for, attempted with the data that exists.

- **The measurement**: 85 ml and 110 ml on two nights, northern Minnesota,
  November 2025 ([`trailer-build.md`](trailer-build.md)).
- **Prediction**: if the energy-balance model describes this collector, its
  output for that site and month brackets the two recorded volumes.
- **Run**: `python simulations/05_transition_paths.py --site field_nov_mn --samples 3000`,
  plus direct evaluation of a best-case configuration at the same site.
- **Result**:

  | Configuration | Model output |
  |---|---|
  | Build as assumed deployed (flat, uninsulated, obstructed sky, Peltier on) | mean 0.05 mL/night, best night 4.6 |
  | Well-configured at the same site (30° tilt, open sky, insulated, frost ignored) | mean 34.7 mL/night, best night 139.5 |

- **Verdict**: **UNDETERMINED — and the reason is missing metadata, not missing
  physics.** The measurement is three orders of magnitude above the first row
  and sits comfortably inside the second. Both readings are consistent with the
  notebook, because **the collector's area and tilt angle were never recorded**.
  We cannot tell whether the model is wrong or whether the assumed baseline
  configuration is wrong.
- **What this costs**: a tape measure and a protractor, applied once in 2025,
  would have made this a real test. Instead the single most valuable dataset the
  project owns cannot discriminate between "the physics is wrong" and "the
  collector was tilted and nobody wrote it down". → **O9, O1**
- **Note on the null-result direction**: the model does NOT rule out the
  reported volumes. A best night of 139.5 mL brackets 85 and 110 comfortably. If
  anything this is weak encouragement for the energy balance — it can produce
  the observed magnitudes under a plausible configuration, which the linear model
  in `01_basic_dew.py` was never in a position to demonstrate either way.

### Transition findings

Site `field_nov_mn`, 2000 draws, budget tranches $0 / $10 / $25 / $50 per unit.

**1. The deployment ran in the wrong month.** At that site in November the model
puts **47% of nights below freezing** and **4% making water**. The same site in
September: **0% frozen, 16% making water**. The night-4 frost failure in the
field log is a modelled outcome of the season, not bad luck — and no hardware
change repairs it. Choosing *when* to run is free and outranks everything that
can be bolted on.

**2. Ordering dominates the parts list.** Each hardware modification was scored
alone against the deployed build, and again in sequence after the free
decisions:

| Modification | Alone | After season + siting fixed |
|---|---|---|
| Angle to 30° | +0.1 | **+10.9** mL/night |
| Wet mulch upwind | +0.4 | **+10.8** mL/night |
| Foam block under mount | +0.4 | **+2.9** mL/night |
| Double collector area | +0.1 | **+31.1** mL/night |
| High-emissivity coating | +0.0 | +2.0 mL/night |

A bracket cannot improve a night that was never going to condense. **A build
guide that lists these parts without the ordering is selling upgrades that will
appear not to work** — and would generate exactly the kind of disappointing
field results that get blamed on the concept rather than the sequence.

**3. The first stage is self-funding.** Removing the Peltier recovers $15, which
pays for the bracket, the foam, and the mulch with $3 left over. Stage one costs
**minus three dollars** and takes the modelled output from ~0 to 25 mL/night. No
funding decision, no procurement, no permission — a screwdriver and a decision
about where the collector sits.

| Stage | Spend (cumulative) | Modelled yield |
|---|---|---|
| Free decisions + self-funded parts | **−$3** | 25.0 mL/night |
| + high-emissivity coating | $5 | 26.9 |
| + larger panel | $23 | 31.1 |
| + double the area | $48 | 62.1 |

**4. Doubling the area is the only lever that never disappoints.** It is
arithmetic, not physics: it cannot fail to work, and it does not depend on any
[ASSUMED] coefficient. Everything above it in the list is cheaper per mL but
rests on the model being roughly right.

**5. One recommendation rests on an invented channel.** The mulch bed scores
+10.8 mL/night in sequence, entirely through `local_vapor_boost` — a mechanism
this project made up and has never measured (O10). The script flags it in its
own output rather than presenting it alongside the defensible steps. It is a
trial to run, not a recommendation to follow.

### Design changes made as a result

The point of the loop is that findings change the artifact. What was altered:

- **[`docs/build-guide.md`](build-guide.md)** rewritten to lead with the two
  free decisions and to specify a tilt angle; Peltier moved from a component to
  an explicit "do not fit". Original preserved at
  [`legacy/2025-original/BUILD-README.md`](../legacy/2025-original/BUILD-README.md).
- **[`docs/trailer-build.md`](trailer-build.md)** — the "next iteration" list
  (frost heater, bigger panel) was replaced. Both existed to keep the Peltier
  running; both are now the wrong spend. Original preserved at
  [`legacy/2025-original/trailer-build.md`](../legacy/2025-original/trailer-build.md).
- **[`firmware/esp32_validation/`](../firmware/esp32_validation/)** — new
  firmware logging collector surface temperature, humidity, and tipping-bucket
  volume, with collector area and tilt as mandatory site constants that refuse
  to pass silently when unset. `esp32_basic` is untouched and still fine for
  what it does; it simply cannot test anything.

---

## Round 4 — 2026-08-16: the logs themselves, tested

The merge with `main` brought in a second falsification record,
[`method-log.md`](method-log.md), written independently by a parallel session
that reached the same round-1 findings by different routes. The question was
which to keep. Rather than argue it, both were audited with
[`../tools/log_audit.py`](../tools/log_audit.py) — the same standard these logs
impose on the models.

### H10 — "this log does what it claims to do"

- **Source**: the rules at the top of this file, and the equivalent at the top
  of `method-log.md`.
- **Prediction**: every entry cites a command a reader can run, carries the
  fields its own format requires, and resolves from any code that cites it.
- **Run**: `python tools/log_audit.py --verify-commands`
- **Result** (before corrections):

  | | method-log | research-log |
  |---|---|---|
  | entries citing a runnable command | 2/8 | 5/9 |
  | entries missing a required field | 0/8 | **2/9** |
  | cited commands that execute | 2/2 | 4/4 |

- **Verdict**: **FALSIFIED for this log, on completeness.** H4 and H8 were
  missing the Prediction field this format requires — written by the same
  session that wrote the rule. `method-log.md` passed its own field check 8/8
  with nobody checking. Corrected: H4, H8 and H9 now carry predictions and
  literal commands, taking this log to 7/9 runnable and 0 incomplete.
- **The measured difference between the formats**: `method-log` is more
  disciplined (per-claim ledger, fixed status vocabulary, IDs cited in code
  comments); this log is more reproducible (explicit `Run:` line that wants a
  command, not a description). Full comparison in
  [`log-format-comparison.md`](log-format-comparison.md).
- **Decision**: keep both. Where the two overlap — H1/M-01, H2/M-02, H5/M-03,
  H6/M-04 — they were derived independently and agree. **That is a replication,
  and it is the strongest evidence in this repository**; everything else here is
  one model run once. Deleting either log to tidy up would destroy it.

### The audit was wrong twice before it was right

Recorded because it is the failure mode this log exists to catch, occurring in
the instrument built to check the log.

Run 1 reported method-log citing almost no commands — the extractor matched only
inline backticks, and that log uses fenced bash blocks. Run 2 reported one of its
commands as failing — the extractor had split a `for` loop into lines and run a
fragment with an unbound variable. Both errors made the other session's work look
worse than it is. Neither was caught by the audit; both were caught by checking a
surprising result against the source.

**An automated check is a claim like any other.** A green audit is evidence about
the audit as much as about the thing audited. → **O14**

---

## Round 5 — 2026-08-21: ENSO

A very strong El Nino is forecast to peak between October 2026 and January 2027,
with a ~69% chance of exceeding every event since 1950. Sources, and the limits
of their chain of custody, are in [`enso-context.md`](enso-context.md).

This is the first question this project has had that *required* the
energy-balance model. `01_basic_dew.py` has no cloud term, so it cannot represent
El Nino at all: to it, a drought is just a lower RH number.

### H11 — "drought is when a dew collector earns its keep"

The implicit claim behind the whole project, never stated as a hypothesis and
therefore never tested.

- **Source**: [`../README.md`](../README.md) framing ("During drought,
  conventional irrigation fails... this approach"), and
  [`build-guide.md`](build-guide.md) title, from 2025-12-07 onward.
- **Prediction**: conditions that define El Nino drought make dew formation
  easier, or at worst leave it unchanged. Drought skies are clear, and clear
  skies are what radiative cooling needs.
- **Run**: `python simulations/06_enso_response.py --all-regions --decompose`
- **Result**: modelled dew yield per m², neutral vs strong El Nino:

  | Region | Neutral | El Nino | Change | Productive nights |
  |---|---|---|---|---|
  | Australia | 8.0 | 1.2 | **−85%** | 13% → 3% |
  | Southern Africa | 8.0 | 1.7 | **−78%** | 13% → 4% |
  | Central America | 8.0 | 1.8 | **−77%** | 13% → 4% |
  | SE Asia | 8.0 | 2.6 | **−68%** | 13% → 6% |
  | Southern US (wetter phase) | 8.0 | 12.9 | **+61%** | 13% → 18% |

  Channel decomposition, mL/night:

  | Region | Drying | Clearing | Warming | Sum | Combined |
  |---|---|---|---|---|---|
  | Australia | −7.1 | +4.3 | −1.5 | −4.3 | −6.8 |
  | Southern Africa | −6.6 | +3.7 | −1.5 | −4.4 | −6.3 |
  | SE Asia | −6.3 | +4.3 | −1.0 | −3.0 | −5.4 |
  | Central America | −6.6 | +3.3 | −1.0 | −4.3 | −6.2 |
  | Southern US | +15.3 | −5.0 | +0.5 | +10.8 | +4.9 |

- **Verdict**: **FALSIFIED.** The prediction was half right and the half that
  was wrong is the one that matters. Clearing *is* a real, substantial benefit —
  +3.3 to +4.3 mL/night, which is not a rounding error. It is simply outweighed
  by drying, roughly 1.5–2x, in every drought region tested. **The system
  produces least exactly where and when it is most needed.**
- **Interaction**: combined is consistently worse than the naive sum in the
  drought regions (−6.8 vs −4.3 for Australia) and better than it in the wet one
  (+4.9 vs +10.8). Both directions have the same cause: the condensation term is
  nonlinear and gated by the dew point. A night that no longer reaches the dew
  point cannot be rescued by a clearer sky, and a night already condensing
  freely gains less from more vapour.
- **Revised claim**: the framing "water during drought" is not supported for
  ENSO-driven drought. What the model supports is narrower and still useful:
  supplementary water in humid-but-water-insecure conditions, and in regions on
  the *wet* side of a strong El Nino. The README and build guide state the
  seasonal-timing rule already (Round 3); this extends it from months to years.
- **Not evidence**: perturbation magnitudes are [ASSUMED]. The signs are sourced
  from the documented teleconnection; the sizes are invented. The finding rests
  on the *ratio* of drying to clearing, which is more robust than either
  magnitude, but is not independent of them. → **O15**

### Deliberately not used as evidence

Published work finds El Nino **intensifies** fog in the Namib (Li et al. 2025)
and raises fog-water yield in the Atacama, while coastal California sees *less*
fog. Those describe advection fog, driven by sea-surface temperature. This model
describes radiative dew. Different mechanism, and the fog literature does not
even agree with itself on sign across sites.

Importing the Namib result as support for a dew claim would be M-01 repeating —
a number detached from the model that produced it. Cited as context in
`enso-context.md`, excluded from the evidence here. That two fog sites disagree
about the sign is a reason to measure, not to borrow.

---

## Open questions

Carried forward. Each names what would close it.

- **O1 — No model-to-measurement comparison exists.** The single largest gap.
  Closing it needs: collector surface area for the trailer build, nightly
  collected volume, and simultaneous logged ground/air temperature and humidity —
  enough to convert ml/night to mm/day per m² and compare against
  `DewSimulator.simulate_night` for that site's actual conditions. The firmware
  logs the temperatures already; nothing logs collected volume or area.
- ~~**O2 — Provenance of the 0.034–0.14 mm/day range.**~~ **CLOSED 2026-08-16
  by method-log M-01.** Both figures came from an uncommitted ion-coupling PDE
  model; only its printed output survives in `legacy/2025-original/`. The number
  outlived the model that produced it. Kept here struck through rather than
  deleted, per the archive rule — a closed question is evidence about how the
  project closes questions.
- **O3 — Is there any amplification at all?** The 3x factor is assumed
  everywhere and derived nowhere. The cheapest test in the repository is the
  paired-container experiment sketched at the end of
  [`legacy/2025-original/firmware__02_crop_response.md`](../legacy/2025-original/firmware__02_crop_response.md)
  ($25, two identical soil containers, one modulated): does the treated container
  collect measurably more dew than the control? Until that runs, ON/OFF plots
  should be read as illustrations of an assumption.
- **O4 — What is the honest use case at ~0.2 mm/day?** The crop-rescue framing
  does not survive H3. Drinking water, seedling/nursery support, and keeping a
  small root zone alive are plausible and untested framings. Each needs its own
  model before it goes in the README.
- **O5 — Trailer data is incomplete.** Two nights recorded of a claimed seven,
  with a frost failure on night 4 and a power failure on day 6. Either the
  missing nights are recoverable from notes, or the average should be restated
  as "2 nights, 85 and 110 ml".
- **O6 — The seed objective needs a defensible cost model.** Amplification gain
  must be expressible in the same units as energy cost before "optimal" means
  anything, and bytes 3–4 (wavelength, crop bias) need either a role in the
  objective or removal from the seed format. The 40-bit layout in
  [`atmospheric-seed-theory.md`](atmospheric-seed-theory.md) (ion amplitude,
  altitude modulation frequency, pattern wavelength, temporal pattern, energy
  budget) does not match what `decode_seed` actually decodes — that mismatch
  should be resolved first.
- **O7 — Most of the 2025-12-07 research was never implemented.** The archived
  session covers adaptive multi-day strategies, layered altitude sensing, network
  architecture, and deployment protocols; the working tree implements none of it.
  Worth an inventory pass to decide what is a live direction and what is
  abandoned, so the theory doc stops implying capability that no code provides.
- **O8 — The climate presets need a time of day.** `01_basic_dew.py` and
  `03_seed_optimization.py` share one RH per climate that is used as if it
  governed nighttime condensation. Either they are daytime values (in which case
  the dew calculation is using the wrong number) or pre-dawn values (in which
  case they are implausibly low for dew formation). Closing this needs a decision
  on which quantity is meant, pre-dawn values sourced per climate, and a
  dew-point check added to `01_basic_dew.py` so it stops reporting water under
  conditions that forbid it. Raised by H7.
- **O9 — Collector tilt is unspecified everywhere.** It is the largest design
  lever in the search (S1 0.152) with a real interior optimum near 19–53°, and
  no build document mentions an angle. Cheap to close: record the angle on the
  existing trailer collector, and add a specified tilt to the build guide. Until
  then, tilt is an uncontrolled variable in every field result the project has.
- **O10 — Assumed coefficients now carry conclusions.** `04_variable_search.py`
  tags its unjustified numbers `[ASSUMED]`: the convective transfer coefficients
  (2.5 W/m²K still, 3.0 per m/s), the tilt-to-collection-efficiency shape, and
  the 0.15 cap on `local_vapor_boost`. Three of the search's top four variables
  sit at bounds this project invented, so those bounds are shaping the answer.
  The vapour-boost channel matters most: it is both influential and completely
  unmeasured.
- **O12 — Seasonal operating window per climate.** Round 3 found that *when* you
  run dominates what you bolt on, and nothing in this repository states a dew
  season for any climate. Each preset needs a start and end month, derived from
  the frost line and pre-dawn humidity, so a builder is not left to discover
  November the hard way. The September/November windows used in
  `05_transition_paths.py` are [ASSUMED] and cover one site only.
- **O13 — Collection efficiency has never been measured.** `EFF_MIN` and
  `EFF_MAX` (0.15 and 0.95) bracket how much of the dew that forms actually
  reaches the vessel, and both are invented. Since tilt is the largest design
  lever and acts entirely through this function, the shape of this curve is
  carrying more weight than any other assumption here. Measurable directly:
  weigh a plate before and after a dew night, compare against what drained.
- **O15 — ENSO perturbation magnitudes are invented.** `06_enso_response.py`
  moves RH, cloud cover and temperature by amounts this project chose, because
  per-region pre-dawn anomalies for a strong El Nino could not be retrieved
  (see `enso-context.md`, Provenance — the primary NOAA and IRI sources were
  unreachable from this environment). H11's conclusion depends on drying
  outweighing clearing by 1.5–2x; that ratio would survive moderate errors in
  either magnitude but has not been tested against real anomaly data. Closing it
  needs observed composite RH and cloud anomalies for a strong El Nino in at
  least one drought region.
- **O16 — Does the anti-correlation hold outside ENSO?** H11 shows supply and
  need moving in opposite directions for ENSO-driven drought. Whether that
  generalises to drought driven by other mechanisms — a failed monsoon, a
  blocking high, long-term aridification — is unknown and matters more than the
  ENSO case, because it decides whether the project's premise is wrong in
  general or only for this one driver.
- **O14 — The audit only checks form, not truth.** `tools/log_audit.py` verifies
  that an entry cites a command and that the command exits zero. It does not
  check that the command produces the numbers the entry claims. An entry could
  cite a working command and report fabricated results and still pass. Closing
  this means capturing expected output per entry and diffing it on rerun —
  which would also catch entries whose numbers silently went stale when the code
  changed, the way Round 2's figures did after the Round 3 fix.
- **O11 — Is the real product passive?** H8 shows active cooling is worth 1.11x
  at the field energy budget and would need ~19x the power for the claimed 3x.
  Meanwhile tilt, siting, and canopy openness carry the leverage. The honest
  reframing may be that this is a passive radiative-cooling collector whose
  performance is set by *where and how you put it*, with the electronics
  demoted to logging. That is a different project from the one the docs
  describe, and deciding between them needs O1's field data.

---

## Adding an entry

Append to the current round; start a new round when the environment or code
changes materially. Keep it to the shape used above:

```markdown
### H<n> — "<the claim, quoted exactly as written>"

- **Source**: where the claim lives, and the date it was first written.
- **Prediction**: what must be true if the claim holds.
- **Run**: the exact command.
- **Result**: what actually came out — numbers, not impressions.
- **Verdict**: SUPPORTED / FALSIFIED / UNTESTED, and why.
- **Revised claim**: the new wording, and what was archived to `legacy/`.
- **Open**: the unknowns this exposed → O<n>.
```

If a run falsifies a claim, revise the claim before the next commit — an
unrevised falsified claim in the working tree is the one failure mode this log
exists to prevent. Archive the old wording; never overwrite it in place.
