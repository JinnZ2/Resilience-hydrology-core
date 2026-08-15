# Resilience Hydrology Core

Physics-based atmospheric water harvesting using natural environmental gradients.

## Project Overview

This system amplifies natural dew/fog formation to collect water during drought.
It uses temperature, pH, and light gradients — no pumps, no wells, no infrastructure.

Modelled output: 0.054-0.100 mm/day passive, 0.162-0.300 mm/day with the system
on, across the four climate presets. These are simulation outputs and have never
been compared against field measurements. The "system on" figures inherit a
hard-coded 3x amplification assumption.

An earlier headline figure of 0.034-0.14 mm/day was withdrawn as not
reproducible from the code — see `docs/research-log.md`, H1.

## Repository Structure

```
simulations/          Python models (numpy/matplotlib/scipy)
  01_basic_dew.py       Basic dew collection simulation (ON vs OFF comparison)
  02_crop_response.py   Crop yield impact during drought
  03_seed_optimization.py  Optimal 40-bit seed finder using differential evolution
firmware/             MicroPython code for ESP32 hardware nodes
  esp32_basic/          Basic temperature logger (DS18B20 sensors)
docs/                 Documentation, build guides, research notes
  research-log.md       Claims tested, falsified, revised; open questions
  build-guide.md        Hardware builds by budget ($50-$2000)
  trailer-build.md      Real-world trailer dew collector results
  atmospheric-seed-theory.md  Research notes on seed expansion physics
legacy/               Superseded originals, archived by date — never deleted
  README.md             Index: what each archived file is and what replaced it
  simulations/          Original dew models (2025-12-07)
  firmware/             Original ESP32 logger, incl. unfinished SD-card path
  notes/                Full 4,404-line seed expansion research session
  docs/                 Original READMEs, preserving pre-revision claims
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

### Evidence conventions

This project runs on an explicit scientific loop: hypothesize → run → compare →
revise the claim → list unknowns → rerun. Two rules follow from it:

- **Never delete a superseded file or claim.** Archive it under `legacy/` with
  its original date and add an entry to `legacy/README.md`. Precedence stays
  with whoever wrote it first, so a revision must remain readable as a revision.
- **Every number in the docs is either a model output with the command that
  reproduces it, or is marked untested.** No numbers without provenance. When a
  run contradicts a documented claim, revise the claim in the same commit and
  log it in `docs/research-log.md` (format is at the bottom of that file).

## Running Simulations

```bash
pip install -r requirements.txt
python simulations/01_basic_dew.py
python simulations/02_crop_response.py
python simulations/03_seed_optimization.py
```

## Key Classes

- `DewSimulator` — Core dew formation model (simulations/01_basic_dew.py)
- `CropWaterModel` — Crop water stress during drought (simulations/02_crop_response.py)
- `SeedOptimizer` — Evolutionary seed optimization (simulations/03_seed_optimization.py).
  Known degenerate: the objective always prefers minimum amplification, and 2 of
  its 5 seed bytes are unused. Do not treat its output as usable seeds.
- `TemperatureLogger` — ESP32 sensor logger (firmware/esp32_basic/main.py)

## Hardware

- ESP32 dev board + 2x DS18B20 sensors + Peltier cooler
- Total cost: $45-180 depending on build
- Field tested: northern Minnesota, November 2025 — one site, 2 nights recorded
  (85 ml, 110 ml). Collector area was not measured, so these cannot yet be
  converted to mm/day and compared against the models. Closing that gap is the
  project's top open item (`docs/research-log.md`, O1).
