# Method Log

A running record of claims this project has made, what happened when they were
tested, and what the claim was changed to. Falsified entries are **kept, not
deleted** — a claim that failed is evidence, and the reasoning that produced it
still sets precedence for how the next claim gets made.

## How to use this file

The loop:

```
hypothesize -> run -> compare to claim -> falsified? -> edit the claim
            -> list what you did not know -> rerun
```

Two rules that make the loop worth running:

1. **Edit the claim, not the model.** If the run disagrees with the README, the
   README is what's wrong until you have a reason to believe otherwise. Tuning
   constants until the output matches a number you already published is not a
   test — it destroys the evidence.
2. **Never move a number between models.** A figure computed by model A is not
   an output of model B, even if both describe "the system." M-01 below is what
   that mistake looks like after a year.

Every entry gets an ID (`M-nn`). Code that is knowingly wrong or unvalidated
cites its ID in a comment, so the code and this file stay tied together.

Status values: `OPEN` (untested) · `FALSIFIED` (run disagreed with claim) ·
`CORRECTED` (claim edited to match the run) · `FIXED` (code was wrong, repaired)
· `UNKNOWN` (cannot be tested with what we have).

Run environment for everything below: Python 3.11, numpy 2.4.6, scipy 1.17.1,
matplotlib 3.11.1, 2026-08-14.

---

## M-01 — The headline output figure

**Claim (as published)**
> "Output: 0.034-0.14 mm/day depending on conditions and energy input."
> — `README.md`, `CLAUDE.md`

**Hypothesis** — Running `simulations/01_basic_dew.py` across the four climate
presets should produce values inside 0.034–0.14 mm/day.

**Run**

```bash
for c in arid semi_arid mediterranean tropical_dry; do
  python simulations/01_basic_dew.py --climate $c
done
```

| Climate | System OFF | System ON |
|---|---|---|
| arid | 0.100 | 0.300 |
| semi_arid | 0.091 | 0.273 |
| mediterranean | 0.054 | 0.162 |
| tropical_dry | 0.080 | 0.240 |

(mm/day)

**Result: FALSIFIED.** Not one ON value falls inside the published range — every
one exceeds the 0.14 upper bound. The value 0.034 is not produced by any preset
in any mode. The published range is not a range this code can output.

**Where the numbers actually came from** — traced through
`legacy/2025-original/firmware__02_crop_response.md`:

| Figure | Origin | Energy cost |
|---|---|---|
| 0.034 mm/day | "natural gradient coupling" mode of an **ion-coupling PDE model** | ~0 kWh/day |
| 0.14 mm/day | "active ion injection" mode of that same model | 5.8 kWh/day |

That model is not in this repository. It was never ported. `01_basic_dew.py` is
an unrelated empirical scaling relation (`RH × ΔT × 0.02`) that replaced it. The
headline figure survived the model that produced it and was re-attached to code
that cannot generate it — and the "depending on energy input" clause is a
leftover from a 0-vs-5.8 kWh/day comparison that no longer exists anywhere in
this repo.

**Claim edited to**
> Modelled output, `01_basic_dew.py`: 0.054–0.100 mm/day unamplified,
> 0.162–0.300 mm/day with the assumed 3× gain. Unvalidated against field data.
> The retired 0.034/0.14 figures belong to an ion-coupling model not present in
> this repository.

**Unknowns surfaced**
- The 0.02 coefficient in `DEW_COEFF` has no derivation and no source. It is
  identical in every version of the file back to the first commit — carried, not
  checked.
- The ion-coupling PDE model was never version-controlled. Only its printed
  outputs survive, in the legacy notes. It cannot be rerun. → **M-05**

**Status: CORRECTED.**

---

## M-02 — The 3× amplification factor

**Claim** — "System amplifies natural process 2-4x" (`01_basic_dew.py` docstring;
implemented as a constant 3.0).

**Run** — Same runs as M-01.

**Result** — Every climate reports "Improvement: +200%", exactly, always. That is
not a finding: `amplification` is a scalar multiplier applied to the same
`natural_dew` in both branches, so the ON/OFF comparison is arithmetically
guaranteed to return the constant regardless of climate, humidity, or ΔT. The
comparison plot cannot fail and therefore tests nothing.

**Not falsified — untestable as written.** 3.0 is an assumption presented as an
output. No measurement, in this repo or the legacy notes, supports 2–4×.

**Claim edited to**
> `AMPLIFICATION = 3.0` is an assumed gain, not a measured one. The ON/OFF plot
> displays that assumption; it does not test it.

**Unknowns surfaced**
- What physical mechanism produces the gain, and does it depend on ΔT or RH? If
  it does, a constant is the wrong shape entirely.
- Measuring this needs paired collectors, powered and unpowered, same night,
  same site. Nothing in `docs/trailer-build.md` was run that way. → **M-07**

**Status: OPEN.** Cannot be closed without field data.

---

## M-03 — "Optimal seed per climate zone"

**Claim** — "Finds the optimal 40-bit seed for each climate zone"
(`simulations/README.md`); "Each build includes seeds optimized for [4 climates]"
(`docs/build-guide.md`).

**Hypothesis** — Different climates should yield materially different seeds.

**Run** — `python simulations/03_seed_optimization.py`, plus a sensitivity sweep
over each byte.

**Result: FALSIFIED, twice over.**

*Every* climate returns amplification bytes `[0, 0, 0, ...]` → `amp_T = amp_pH =
amp_light = 1.00`, the floor of the range. Term breakdown at the arid optimum:

| Amplification | precip | energy | safety | score |
|---|---|---|---|---|
| 1.0 (byte 0) | +0.0500 | −0.1500 | +0.4500 | **+0.3500** |
| 5.0 (byte 255) | +0.2500 | −0.7500 | +0.2500 | **−0.2500** |

Turning amplification up buys +0.20 of precipitation and costs −0.80 in energy
and safety. The objective is monotonically decreasing in amplification, so the
optimiser's answer is always "run the system at minimum" — i.e. **don't run it**.
The optimum is not a peak, it is the wall of the search box.

Second failure: byte sensitivity, measured as score range over each byte:

| Byte | Parameter | Score range |
|---|---|---|
| 0 | amp_T | 0.3400 |
| 1 | amp_pH | 0.3200 |
| 2 | amp_light | 0.3400 |
| 3 | wavelength | **0.0000** |
| 4 | crop_bias | **0.0000** |

`wavelength` and `crop_bias` never enter `evaluate_seed`. Two of five bytes are
unconstrained, so differential evolution returns whatever it happened to be
holding. The per-climate variation the earlier output appeared to show — 694 nm
arid, 2935 nm semi-arid, 3782 nm mediterranean — was **random number generator
state, presented as climate adaptation**. Rerunning gives different values with
no seed change and no code change.

**Claim edited to**
> `03_seed_optimization.py` demonstrates the seed-decode/optimise scaffolding.
> Its objective function is degenerate and does not select a seed. The values it
> prints are not tuned configurations.

**Code changed (not to fix the result — to stop it lying):** the script now
reports the score-term breakdown, measures per-byte sensitivity, and prints an
explicit warning when bytes are unconstrained or the optimum sits on a bound. The
objective itself was left exactly as it was. Repairing the weights to make the
answer look reasonable would have manufactured the conclusion.

**Unknowns surfaced**
- The weights (1.0 precip, 0.1 energy, 0.5 safety) mix mm/day, kWh/day and a
  dimensionless safety index with no stated exchange rate. What is a mm of water
  worth in kWh? Until that is answered the objective is not merely degenerate,
  it is meaningless — and picking weights that flip the sign would be choosing
  the answer.
- The code's 5-byte layout (amp_T, amp_pH, amp_light, wavelength, crop_bias)
  does not match the 40-bit layout in `docs/atmospheric-seed-theory.md` (ion
  amplitude, altitude modulation, horizontal wavelength, temporal modulation,
  energy budget). Same bit count, different meanings, neither derived from the
  other. → **M-08**

**Status: FALSIFIED.** Left in the repo as scaffolding with its failure labelled.

---

## M-04 — Crop stress tolerance was inverted

**Claim** — `stress_tolerance` in `02_crop_response.py`, commented
"higher = more tolerant": olive 2.0 (`# Very tolerant`), wheat 1.5, tomato 0.7
(`# Sensitive`).

**Run** — Held average stress fixed at 0.6 and swept the exponent:

| tolerance | `1 - 0.6**(1/tol)` (as written) | `1 - 0.6**tol` (corrected) |
|---|---|---|
| 0.7 | 0.518 | 0.301 |
| 1.5 | 0.289 | 0.535 |
| 2.0 | 0.225 | 0.640 |

**Result: FALSIFIED — code bug, not a claim bug.** Stress is in [0,1], so raising
it to `1/tolerance` makes the *reduction larger* as tolerance rises. The
parameter did the exact opposite of its name. Before the fix, "sensitive" tomato
(46.1%) out-yielded "tolerant" wheat (24.8%) under identical drought.

**Fixed** — exponent is now `tolerance`, not `1.0 / tolerance`.

**Rerun** — 60-day drought, 0.034 mm/day:

| Crop | tolerance | Yield before fix | Yield after fix |
|---|---|---|---|
| wheat | 1.5 | 24.8% | 47.4% |
| olive | 2.0 | 44.5% | 90.5% |
| tomato | 0.7 | 46.1% | 26.1% |

Ordering now matches the labels: olive > wheat > tomato.

**Note:** this bug was invisible for as long as it was because nobody stated the
prediction "olive should out-yield tomato" *before* reading the output. The
numbers looked plausible, so they were accepted. Predict first, then run.

**Status: FIXED.**

---

## M-05 — Does the modelled water actually matter to a crop?

**Claim** — "0.034 mm/day → 5-20% yield improvement"
(`legacy/2025-original/firmware__02_crop_response.md`, line 4330).

**Hypothesis** — 0.034 mm/day during a 60-day drought produces a 5–20% yield gain.

**Run** — `02_crop_response.py` at three input levels, post-M-04 fix:

| Water input | wheat | olive | tomato |
|---|---|---|---|
| 0.034 mm/day (retired figure) | +0.3% | +0.8% | +0.4% |
| 0.27 mm/day (`01_basic_dew.py` semi-arid, ON) | +2.5% | +5.3% | +3.2% |
| 3.0 mm/day (~crop peak demand) | +83.8% | +10.5% | +163.8% |

(relative yield change vs. no system)

**Result: FALSIFIED.** At 0.034 mm/day the effect is 0.3–0.8%, an order of
magnitude below the claimed 5–20%. Even at the current dew model's most
optimistic output the gain is 2.5–5.3%, at the very bottom of the claimed band.

The scale problem is the point: crop demand in these models is 0.5–4.5 mm/day.
0.034 mm/day is **under 1% of peak demand**. A drought is not closed by 1% of
demand, and no amount of amplification in the current model bridges two orders of
magnitude.

**Claim edited to**
> At modelled output levels, atmospheric water yields single-digit-percent
> relative yield changes during drought. This is a supplement, not drought
> mitigation. Meaningful yield rescue requires ~3 mm/day, roughly 100× the
> retired figure and ~10× the current model's optimistic case.

**Unknowns surfaced**
- Both crop models assume drought means *zero* rain and system water only. Real
  partial-drought behaviour is untested.
- The soil model has no percolation, runoff, or evaporation term. Water in
  equals water available. Very small daily inputs are exactly where that
  simplification is least safe — 0.034 mm/day may not reach the root zone at all
  before evaporating, in which case the true effect is not "small," it is zero.

**Status: CORRECTED.**

---

## M-06 — GPIO5 pin collision

**Claim** — implicit: `firmware/esp32_basic/main.py` works as wired.

**Run** — Static read of pin assignments (no hardware available in this
environment).

**Result: FALSIFIED.** `TemperatureLogger.__init__` puts the air sensor's OneWire
bus on `Pin(5)`; `setup_sd_card()` puts the SD card chip-select on `Pin(5)`. Both
drive the same pin. With an SD card fitted, the air sensor readings are corrupt
or absent, and `delta_t` — the only quantity this node exists to measure —
silently becomes `None`.

Worse than a crash: `read_temperatures()` catches the exception, logs `NA`, and
the loop continues. A field node would run for weeks producing a CSV of nulls
with no visible error.

**Fixed** — SD chip-select moved to GPIO15. Pins hoisted to named constants, and
`TemperatureLogger.__init__` now raises if `SD_CS` collides with a sensor pin, so
the next pin change fails loudly instead of silently.

**Unknowns surfaced**
- **Not verified on hardware.** GPIO15 is a strapping pin on the ESP32 (it must
  be high at boot for normal flash mode); it is a conventional SD-CS choice and
  the SD card's pull-up should be benign, but this has not been confirmed on a
  board. Anyone with hardware: confirm the node boots with a card fitted, and
  record the result here.
- The trailer build (`docs/trailer-build.md`, Nov 2025) predates this file. It is
  unknown whether it ran with an SD card, and therefore whether its logged
  temperatures were affected.

**Status: FIXED (unverified on hardware).**

---

## M-07 — Field results are not comparable to model output

**Claim** — "Average: 95ml/night" over 7 nights, northern Minnesota, Nov 2025
(`docs/trailer-build.md`); "Prototype hardware tested", "Physics models
validated" (`README.md`).

**Run** — Attempted to compare the field figure to the model. Could not.

**Result: UNKNOWN — the comparison cannot be made.**

The model reports mm/day, which is volume per unit area. The field log reports
ml/night with **no collector area recorded**. Without area the two are not the
same kind of quantity and no conversion exists. As a rough illustration only: if
the collecting surface were 1 m², 95 ml/night ≈ 0.095 mm/night — which would land
just above the model's *unamplified* mediterranean-through-arid band (0.054–0.100)
and roughly 3× *below* its amplified band. But the area is a guess, so this
comparison establishes nothing in either direction.

Three further blockers:

1. **No control.** No unpowered collector ran alongside. The 95 ml/night is total
   collection, not collection *attributable to the system*. M-02's 3× gain
   remains untested by this run.
2. **Climate mismatch.** Northern Minnesota in November is not any of the four
   presets — all of which are warm, dry-season climates. The model has no preset
   for the conditions actually tested.
3. **Two of seven nights failed** (icing night 4, battery death day 6, per the
   build notes), so the 95 ml average is over an unstated subset.

**Claim edited to**
> A prototype collected water in the field. Its output has not been compared to
> model predictions, and no claim of model validation is supported. "Physics
> models validated" is withdrawn.

**What to record next time** — the minimum for a comparable run: collector area
(m²), a paired unpowered control at the same site, per-night ambient T_day /
T_night / RH, and every night including failures, labelled.

**Status: UNKNOWN.** Highest-value gap in the project — it is the only entry that
a modest, cheap experiment could close outright.

---

## M-08 — Two incompatible 40-bit seed layouts

**Claim** — The 40-bit seed encodes atmospheric control parameters.

**Result: CONTRADICTION, unresolved.** Two layouts coexist with no stated
relationship:

| Bits | `docs/atmospheric-seed-theory.md` | `03_seed_optimization.py` |
|---|---|---|
| 0–7 | near-surface ion production amplitude | amp_T (1.0–5.0) |
| 8–15 | altitude modulation frequency | amp_pH (1.0–5.0) |
| 16–23 | horizontal pattern wavelength | amp_light (1.0–5.0) |
| 24–31 | temporal modulation pattern | wavelength (500–5000 nm) |
| 32–39 | energy budget allocation | crop_bias (0–1) |

The theory doc's layout belongs to the ion-coupling model of M-01 — the one no
longer in the repo. The code's layout belongs to the temperature/pH/light
approach that replaced it. They share a bit count and nothing else. Note that
"wavelength" appears in both, meaning different things (a horizontal atmospheric
pattern scale vs. an optical wavelength in nm), which is how the collision went
unnoticed.

**Unknowns**
- Is 40 bits a requirement, or an artifact carried over from the orbital-seed
  analogy the theory doc borrows from? Nothing in either model derives it.
- Which layout, if either, should the repo keep? This cannot be settled by
  running code — it is a design decision, recorded here as open rather than
  quietly resolved by whichever file gets edited next.

**Status: OPEN.**

---

## Summary

| ID | Subject | Status |
|---|---|---|
| M-01 | 0.034–0.14 mm/day headline figure | CORRECTED |
| M-02 | 3× amplification | OPEN |
| M-03 | Per-climate optimal seeds | FALSIFIED |
| M-04 | Crop stress tolerance inverted | FIXED |
| M-05 | 5–20% yield improvement | CORRECTED |
| M-06 | GPIO5 collision | FIXED (unverified on hardware) |
| M-07 | Field results vs. model | UNKNOWN |
| M-08 | Conflicting seed layouts | OPEN |

**The pattern worth carrying forward:** M-01, M-03, M-05 and M-08 are all the
same failure. A number or a structure outlived the model that produced it, got
re-attached to different code, and was repeated until it read as established.
Nothing was fabricated at any single step — each restatement was a reasonable
copy of the previous one. The error was that no step re-derived the figure from
the code actually present.

The cheap defence is provenance. Every number in this repo should be traceable
to either a command someone can run today, or a measurement someone recorded with
its conditions. If it is neither, it is a hypothesis, and it should be labelled
as one and given an ID here.

## Adding an entry

```markdown
## M-nn — Short subject

**Claim** — What was asserted, and where it is written.
**Hypothesis** — The prediction, stated BEFORE the run.
**Run** — Exact command. Paste real output.
**Result** — FALSIFIED / CONFIRMED / UNKNOWN, and why.
**Claim edited to** — The replacement wording, now live in the docs.
**Unknowns surfaced** — What you learned you did not know.
**Status** — OPEN / FALSIFIED / CORRECTED / FIXED / UNKNOWN
```

Do not delete an entry when it is superseded. Add the new one, and link back.
