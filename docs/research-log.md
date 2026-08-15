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
  [`legacy/docs/README_2025-12-07.md`](../legacy/docs/README_2025-12-07.md).
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
  system water input, so the low end is likely a hand-carried constant rather
  than a model output. The origin of `0.14` was not found anywhere in the
  repository or its history. → **O2**
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
  ([`legacy/simulations/01_basic_dew_simulation.py`](../legacy/simulations/01_basic_dew_simulation.py))
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

- **Run**: swept active cooling power on a favourable semi-arid night
  (291 K, RH 0.80, light wind, clear sky, tilt 30°, COP 0.7).
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

| Variable | Kind | S1 | Best range | Note |
|---|---|---|---|---|
| tilt_deg | design | 0.152 | 19–53° | interior optimum |
| rh | climate | 0.140 | 0.84–0.89 | at upper bound |
| local_vapor_boost | siting | 0.086 | 0.08–0.14 | at upper bound |
| sky_view_factor | siting | 0.078 | 0.69–0.97 | at upper bound |
| cloud_cover | climate | 0.052 | 0.02–0.35 | at lower bound |
| wind_speed | climate | 0.023 | 0.8–4.4 m/s | interior optimum |
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

## Open questions

Carried forward. Each names what would close it.

- **O1 — No model-to-measurement comparison exists.** The single largest gap.
  Closing it needs: collector surface area for the trailer build, nightly
  collected volume, and simultaneous logged ground/air temperature and humidity —
  enough to convert ml/night to mm/day per m² and compare against
  `DewSimulator.simulate_night` for that site's actual conditions. The firmware
  logs the temperatures already; nothing logs collected volume or area.
- **O2 — Provenance of the 0.034–0.14 mm/day range.** `0.034` is traceable to
  the crop model's input constant. `0.14` is not traceable to anything in this
  repository. If it came from a measurement or an external source, that source
  should be cited in the README; if it cannot be found, the number should stay
  retired.
- **O3 — Is there any amplification at all?** The 3x factor is assumed
  everywhere and derived nowhere. The cheapest test in the repository is the
  paired-container experiment sketched at the end of
  [`legacy/notes/2025-12-07_seed-expansion-session.md`](../legacy/notes/2025-12-07_seed-expansion-session.md)
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
