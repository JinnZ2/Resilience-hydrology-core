# Simulations

Python models for atmospheric water harvesting.

**These are models, not measurements.** None of them has been compared against
field data, and each carries assumptions that the runs below cannot test. What
was checked and what failed is recorded in
[../docs/research-log.md](../docs/research-log.md).

## Requirements

```bash
pip install -r ../requirements.txt
```

## Files

### 01_basic_dew.py

Simulates 7 days of dew collection with/without the system.

```bash
python 01_basic_dew.py
python 01_basic_dew.py --climate arid --days 14
```

**Output**: Graph + summary statistics showing system ON vs OFF comparison.
Modelled output runs 0.054–0.100 mm/day OFF and 0.162–0.300 mm/day ON,
depending on the climate preset.

⚠️ The ON case multiplies by a hard-coded `amplification = 3.0`. That factor is
an assumption with no derivation or measurement behind it, so every climate
reports exactly "+200%" and the comparison cannot test whether amplification
happens at all (research log, H2).

### 02_crop_response.py

Models crop yield impact during drought for wheat, olive, and tomato.

```bash
python 02_crop_response.py
```

**Output**: Bar chart comparing yield with/without atmospheric water input (0.034 mm/day).

⚠️ At 0.034 mm/day the modelled effect is under one percentage point of yield —
crop demand in the model is 1.5–4.5 mm/day, so the system supplies roughly 1% of
it. Material yield rescue requires 2–3 mm/day (research log, H3). The
`stress_tolerance` parameter was inverted until 2026-08-15; results published
before that date are wrong (H6).

### 03_seed_optimization.py

Searches for a 40-bit seed per climate zone using differential evolution.

```bash
python 03_seed_optimization.py
```

**Output**: Seed bytes and parameters for arid, semi-arid, mediterranean, and tropical dry climates.

⚠️ **Superseded for practical purposes by `04_variable_search.py`**, which does
the same job (search a constrained space) without the degeneracy below, and
reports ranges rather than points.

⚠️ **The published seeds are not usable.** The objective is monotonically
decreasing in amplification, so every climate returns the minimum of the search
range — "optimal" here means the system turned all the way down. Bytes 3 and 4
(`wavelength`, `crop_bias`) are decoded but never read by `evaluate_seed`, so
they come back as different random values on every run at an identical score.
The file is kept as a search scaffold; it needs a defensible cost model before
its output means anything (research log, H5 and O6).

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
