# Resilience Hydrology Core

Physics-based atmospheric water harvesting using natural environmental gradients.

## Project Overview

This system amplifies natural dew/fog formation to collect water during drought.
It uses temperature, pH, and light gradients — no pumps, no wells, no infrastructure.

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
  03_seed_optimization.py  Optimal 40-bit seed finder using differential evolution
firmware/             MicroPython code for ESP32 hardware nodes
  esp32_basic/          Basic temperature logger (DS18B20 sensors)
docs/                 Documentation, build guides, research notes
  method-log.md         Claims, tests, falsifications, open unknowns (read first)
  build-guide.md        Hardware builds by budget ($50-$2000)
  trailer-build.md      Real-world trailer dew collector results
  atmospheric-seed-theory.md  Research notes on seed expansion physics
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
- Simulation files are numbered: `01_`, `02_`, `03_`
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
- **Superseded files go to `legacy/` frozen, never deleted** — including, and
  especially, ones whose claims were falsified. Nothing current imports from
  `legacy/`; cite it, don't copy numbers out of it.

## Running Simulations

```bash
pip install -r requirements.txt
python simulations/01_basic_dew.py --climate arid --days 14
python simulations/02_crop_response.py --water 0.27
python simulations/03_seed_optimization.py   # reports its own degeneracy (M-03)
```

## Key Classes

- `DewSimulator` — Core dew formation model (simulations/01_basic_dew.py)
- `CropWaterModel` — Crop water stress during drought (simulations/02_crop_response.py)
- `SeedOptimizer` — Evolutionary seed optimization (simulations/03_seed_optimization.py)
- `TemperatureLogger` — ESP32 sensor logger (firmware/esp32_basic/main.py)

## Hardware

- ESP32 dev board + 2x DS18B20 sensors + Peltier cooler
- Total cost: $45-180 depending on build
- Field tested: northern Minnesota, November 2025 — collected water, but the run
  is **not comparable to model output** (no collector area, no control). See M-07.
- Pin map: GPIO4 ground sensor, GPIO5 air sensor, GPIO15 SD chip-select.
  GPIO15 was moved off GPIO5 to resolve a collision with the air sensor (M-06);
  the fix is unverified on hardware.
