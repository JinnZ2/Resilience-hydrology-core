# Simulations

Python models for atmospheric water harvesting.

Each entry below states what the model does **and where it is known to be
wrong**. Full falsification records are in [`../docs/method-log.md`](../docs/method-log.md).

## Requirements

```bash
pip install -r ../requirements.txt
```

## Files

### 01_basic_dew.py

Simulates dew collection with and without the system, across four climate presets.

```bash
python 01_basic_dew.py
python 01_basic_dew.py --climate arid --days 14
```

**Output**: Graph + summary statistics, system ON vs OFF.

| Climate | OFF | ON |
|---|---|---|
| arid | 0.100 | 0.300 |
| semi_arid | 0.091 | 0.273 |
| mediterranean | 0.054 | 0.162 |
| tropical_dry | 0.080 | 0.240 |

(mm/day)

**Known limits**
- `DEW_COEFF = 0.02` has no derivation and no source (M-01). Absolute values are
  unvalidated.
- `AMPLIFICATION = 3.0` is an assumed gain, not a measured one. Because it's a
  constant multiplier on both branches, the ON/OFF comparison always reports
  exactly +200% regardless of climate — it displays the assumption rather than
  testing it (M-02).
- Not modelled: dew point, surface temperature, condenser area, collection
  efficiency, wind, radiative cooling.

### 02_crop_response.py

Models crop yield impact during drought for wheat, olive, and tomato.

```bash
python 02_crop_response.py
python 02_crop_response.py --water 0.27 --drought-days 60
```

**Output**: Bar chart comparing yield with and without atmospheric water.

The `--water` default (0.034 mm/day) is a **retired figure** inherited from a
model no longer in this repo, kept as the default only so historical runs stay
reproducible. To use the current dew model's output, pass `--water 0.27` (M-01).

Effect size at each level, 60-day drought:

| Water input | wheat | olive | tomato |
|---|---|---|---|
| 0.034 mm/day | +0.3% | +0.8% | +0.4% |
| 0.27 mm/day | +2.5% | +5.3% | +3.2% |
| 3.0 mm/day | +83.8% | +10.5% | +163.8% |

Crop demand in this model is 0.5–4.5 mm/day, so modelled system output is under
1% of peak demand. The claimed "5–20% yield improvement" was falsified (M-05).

**Known limits**
- Drought is modelled as zero rain; partial drought is untested.
- The soil model has no percolation, runoff, or evaporation. Small daily inputs
  are exactly where that simplification is least safe.
- The `stress_tolerance` exponent was inverted until 2026-08 — results predating
  that fix have the crop ranking backwards (M-04).

### 03_seed_optimization.py

Differential-evolution search over a 5-byte (40-bit) seed.

```bash
python 03_seed_optimization.py
```

**This model is degenerate. It does not select a seed (M-03).**

Two independent failures, both now reported by the script itself:

1. The energy and safety penalties outweigh the precipitation term at every
   amplification level, so the optimum is always minimum amplification — the
   objective's answer is "don't run the system." The result sits on the boundary
   of the search box, not at a peak.
2. Bytes 3 and 4 (`wavelength`, `crop_bias`) never enter the objective. Measured
   score sensitivity is exactly 0.0000 for both, so the optimiser returns
   arbitrary values. Per-climate variation in those bytes is RNG state, not
   climate adaptation.

The objective was deliberately **left unfixed** — retuning the weights to produce
a plausible-looking answer would manufacture the conclusion. The script now
prints a term breakdown, per-byte sensitivity, and explicit warnings instead.

Kept as scaffolding for the seed decode/optimise loop. Do not quote its output as
tuned configurations.

### 04_variable_search.py

Searches a constrained variable space for the *ranges* that improve yield, and
ranks variables — including ecological ones — by how much they actually move the
outcome.

```bash
python 04_variable_search.py                              # semi-arid, 4000 samples
python 04_variable_search.py --list-variables              # the constraint table
python 04_variable_search.py --condensing-only             # isolate design levers
python 04_variable_search.py --climate arid --samples 12000 --condensing-only
python 04_variable_search.py --fix tilt_deg=30 --energy-budget 10
python 04_variable_search.py --climate repo_daytime_rh     # reproduces the H7 null result
```

**Output**: a five-part report — sensitivity ranking, optimal ranges, interior
optima, ecological levers, and measurement priorities — plus a PNG.

Unlike the other three files this one uses a surface energy balance (radiative
loss + active cooling = convective gain + latent release + conduction) rather
than a linear formula, because the ecological variables need somewhere physical
to act. Coefficients are tagged `[STANDARD]` or `[ASSUMED]` in the source.

What it reports, and why each part exists:

- **Sensitivity** — first-order index `Var(E[Y|X])/Var(Y)`, which catches
  variables with a peak in the middle that a correlation coefficient misses.
- **Optimal ranges** — where the top 10% of outcomes sit, with a `narrow` score
  saying whether the variable is actually selective.
- **Corner-solution detection** — flags any "optimum" that is really just a
  bound. This exists because `03_seed_optimization.py` shipped exactly that
  failure undetected (research log, H5).
- **Interior optima** — the variables with a genuine best range. Trust the
  range; the peak *point* is the noisiest number in the report.
- **Measurement priorities** — high-leverage variables nobody has measured,
  which is the actionable form of O1.

Main results so far are in the research log, Round 2. The short version: tilt is
the biggest design lever and no build guide specifies it; canopy openness and
upwind soil moisture outrank every hardware variable except tilt; and active
cooling within the field build's energy budget is worth 1.11x, not 3x.

⚠️ Everything it prints is a property of the model, which has never been
compared against a field measurement. The rankings are hypotheses about where to
look, not findings about dew.

### 05_transition_paths.py

Takes an already-built collector and returns the cheapest ordered set of changes
to improve it — with costs, hours, and who has to act.

```bash
python 05_transition_paths.py                              # the actual deployment site
python 05_transition_paths.py --list-mods                  # the modification catalogue
python 05_transition_paths.py --site semi_arid_summer
python 05_transition_paths.py --budget 0 10 25 50 100
```

**Output**: no-regret moves, information moves, a staged plan by budget tranche,
an ordering-effects table, a breakdown by actor, and an explicit "what not to
do".

Every modification is scored across a Monte Carlo varying **both the weather and
this project's own `[ASSUMED]` coefficients**, so a recommendation that survives
is one that does not depend on us being right about the numbers we invented.
Modifications are re-scored after each step is applied, so interactions are
handled rather than assumed additive.

Three results worth knowing:

- **The first stage costs −$3.** Removing the Peltier pays for the bracket, the
  foam, and the mulch.
- **Ordering beats the parts list.** Angling the collector is worth +0.1 mL/night
  on its own and +10.9 after the free season and siting decisions. A guide that
  lists parts without the ordering sells upgrades that appear not to work.
- **Information outranks hardware.** The entire measurement kit is $23 and about
  four hours, and it is what makes every other number in this repository
  checkable.

Details in the research log, Round 3.

### 06_enso_response.py

What a strong El Niño does to modelled dew yield, region by region, with the
competing channels separated.

```bash
python 06_enso_response.py --all-regions --decompose
python 06_enso_response.py --region southern_africa --samples 4000
python 06_enso_response.py --list-regions
```

**Output**: yield change per region, and a decomposition into drying, clearing
and warming.

El Niño drought does two opposite things to radiative dew — it dries the air
(less vapour to condense) and clears the sky (stronger radiative cooling). This
is the first question in the project that *required* the energy balance:
`01_basic_dew.py` has no cloud term at all, so to it a drought is just a smaller
RH number.

Result: **drying wins by 1.5–2x in every drought region tested.** Modelled yield
falls 68–85% across Australia, southern Africa, South-East Asia and Central
America, and rises 61% in the southern US, which a strong El Niño makes wetter.
Clearing is a real benefit, just an outmatched one.

The uncomfortable conclusion is in the research log as H11: this system produces
least exactly where and when it is most needed.

⚠️ The perturbation magnitudes are `[ASSUMED]` — only their signs are sourced
(see [`../docs/enso-context.md`](../docs/enso-context.md), which also records why
the primary NOAA sources could not be retrieved). Read the sign and the ranking,
not the absolute mL.
