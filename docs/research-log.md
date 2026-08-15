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
