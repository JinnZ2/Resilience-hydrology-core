# Resilience Hydrology Core

Physics-based atmospheric water harvesting using natural environmental gradients.

## Project Overview

This system amplifies natural dew/fog formation to collect water. It uses
temperature, pH, and light gradients — no pumps, no wells, no infrastructure.

It was framed as drought mitigation. That framing is falsified for ENSO-driven
drought (research-log H11): dew needs humid air, El Nino droughts are dry-air
droughts, and modelled yield falls 68-85% in the regions most at risk. Supply and
need move in opposite directions. Qualify accordingly.

Modelled output (`simulations/01_basic_dew.py`): 0.054-0.100 mm/day unamplified,
0.162-0.300 mm/day with the assumed 3x gain. Unvalidated against field data.

The previously documented "0.034-0.14 mm/day" figure was withdrawn — it came
from an ion-coupling model that is not in this repository. See
`docs/method-log.md` M-01 before quoting any output number.

## Repository Structure

```
simulations/          Python models (numpy/matplotlib/scipy)
  01_basic_dew.py       Basic dew collection simulation (ON vs OFF comparison)
  02_crop_response.py   Crop yield impact during drought
  03_seed_optimization.py  Seed search; reports its own degeneracy (M-03)
  04_variable_search.py    Constrained variable search: sensitivity, optimal
                           ranges, ecological levers, measurement priorities
  05_transition_paths.py   Cheapest ordered changes from a deployed build to a
                           better one, scored under our own uncertainty
  06_enso_response.py      Dew yield under a strong El Nino, by region, with the
                           drying-vs-clearing channels separated
  07_alternative_systems.py  Mechanism comparison for dry air: dew vs active
                           condensation vs sorption, with the feasibility walls
firmware/             MicroPython code for ESP32 hardware nodes
  esp32_basic/          Basic temperature logger (DS18B20 sensors)
docs/                 Documentation, build guides, research notes
  method-log.md         Claims, tests, falsifications, open unknowns (read first)
  research-log.md       Second falsification record; rounds 2-5 continue past
                        the method log (H-nn entries, O-nn open questions)
  enso-context.md       ENSO state, sources, and the drought-premise problem
  alternative-systems.md  What works when air is too dry for dew (sorption)
  build-guide.md        Hardware builds by budget ($50-$2000)
  trailer-build.md      Real-world trailer dew collector results
  atmospheric-seed-theory.md  Research notes on seed expansion physics
tools/                Repository self-checks
  log_audit.py          Audits both falsification logs: does every claim cite a
                        runnable command, carry its required fields, and resolve
                        from the code that cites it?
legacy/               Superseded files, frozen — the precedence record
  README.md             Index: what each file was, what replaced it, why kept
  2025-original/        Pre-standardisation state
```

## Tech Stack

- **Simulations**: Python 3, numpy, matplotlib, scipy
- **Firmware**: MicroPython on ESP32
- **Sensors**: DS18B20 (temperature), OneWire protocol

## Conventions

- Python files use snake_case for functions, variables, and file names
- Classes use PascalCase
- Simulation files are numbered: `01_` through `06_`
- Units: mm/day for water output, Kelvin for temperatures in code, Celsius in display
- Climate presets: arid, semi_arid, mediterranean, tropical_dry

## Claim Discipline

This repo keeps a falsification record in `docs/method-log.md`. It exists because
the same failure has already occurred four times: a number outlived the model
that produced it, was re-attached to different code, and got repeated until it
read as established (M-01, M-03, M-05, M-08).

Rules:

- **Check the method log before quoting any number.** Several published figures
  have been withdrawn there.
- **Every number must trace to a command runnable today, or a recorded
  measurement with its conditions.** If it's neither, it's a hypothesis — label
  it and give it an `M-nn` ID.
- **When a run disagrees with the docs, edit the claim, not the model.** Tuning
  constants until the output matches an already-published number destroys the
  evidence. M-03 is deliberately left broken for this reason.
- **State the prediction before reading the output.** M-04 hid for a year because
  wrong numbers looked plausible.
- **Code that is knowingly wrong or unvalidated cites its `M-nn` ID in a
  comment**, so code and log stay tied together.
- **Run `python tools/log_audit.py` after touching either log.** It checks the
  logs against their own rules and exits nonzero on a violation. It found two
  entries in `research-log.md` missing a required field, written by the session
  that wrote the rule.
- **Two logs, both live.** `method-log.md` (M-nn) came first and holds
  precedence; `research-log.md` (H-nn/O-nn) continues into rounds 2-3. They were
  written independently and independently reached the same round-1 findings.
  Where they overlap, cite the M-entry as the original.
- **Superseded files go to `legacy/` frozen, never deleted** — including, and
  especially, ones whose claims were falsified. Nothing current imports from
  `legacy/`; cite it, don't copy numbers out of it.

## Running Simulations

```bash
pip install -r requirements.txt
python simulations/01_basic_dew.py --climate arid --days 14
python simulations/02_crop_response.py --water 0.27
python simulations/03_seed_optimization.py   # reports its own degeneracy (M-03)
python simulations/04_variable_search.py --condensing-only
python simulations/05_transition_paths.py
python simulations/06_enso_response.py --all-regions --decompose
python simulations/07_alternative_systems.py --sweep --budget-check

Numbered filenames start with a digit, so they cannot be imported normally.
`05_transition_paths.py`, `06_enso_response.py` and `07_alternative_systems.py`
load `04_variable_search.py`
via importlib; follow that pattern if another file needs to reuse a model.
```

## Key Classes

- `DewSimulator` — Core dew formation model (simulations/01_basic_dew.py)
- `CropWaterModel` — Crop water stress during drought (simulations/02_crop_response.py)
- `SeedOptimizer` — Evolutionary seed optimization (simulations/03_seed_optimization.py)
- `DewEnergyBalance` — Surface energy-balance dew model: the physically
  structured alternative to `DewSimulator`, and the only model here that checks
  whether the surface actually reaches the dew point
  (simulations/04_variable_search.py)
- `VariableSearch` — Constrained sampling and sensitivity analysis
  (simulations/04_variable_search.py). Its `Variable` registry is the single
  place where variable bounds, kinds, and measurement status are declared.
- `TransitionEvaluator` / `Modification` — Retrofit scoring under weather and
  coefficient uncertainty (simulations/05_transition_paths.py)
- `EnsoComparison` — Paired-quantile dew comparison across ENSO states
  (simulations/06_enso_response.py)
- `TemperatureLogger` — ESP32 sensor logger (firmware/esp32_basic/main.py)
- `ValidationLogger` — Surface temp + humidity + volume logger
  (firmware/esp32_validation/main.py)

## Hardware

- ESP32 dev board + 2x DS18B20 sensors + Peltier cooler
- Total cost: $45-180 depending on build
- Field tested: northern Minnesota, November 2025 — collected water, but the run
  is **not comparable to model output** (no collector area, no control). See M-07.
- **The Peltier cooler is no longer recommended.** At these builds' energy
  budget it delivers ~6% of the radiative cooling the surface already does for
  free — worth ~1.11x against a claimed 3x (research-log H8). Removing it
  recovers $15 and funds the changes that do work.
- **Dew is a wall, not a slope (H12).** Below the point where dew-point
  depression exceeds achievable radiative cooling, yield is exactly zero, not
  small. The wall sits near 70% RH at 15 C and near 90% at 32 C. Design
  improvements multiply zero below it.
- **Two regimes, not one (O18).** Dew suits cool humid nights; severe drought is
  hot and dry and needs sorption, which works to ~11% RH on solar heat. The repo
  has not yet decided which project it is. Do not extend the dew build guide as
  if it covered both.
- **The drought premise is falsified for ENSO drought (H11).** Dew needs humid
  air; El Nino droughts are dry-air droughts. Modelled yield falls 68-85% in the
  regions a strong El Nino puts at risk. Do not describe this system as drought
  mitigation without that qualification.
- Build order matters more than the parts: season, then siting, then tilt. Each
  hardware change is near-worthless before those and large after them (Round 3).
- Pin map: GPIO4 ground sensor, GPIO5 air sensor, GPIO15 SD chip-select.
  GPIO15 was moved off GPIO5 to resolve a collision with the air sensor (M-06);
  the fix is unverified on hardware.
