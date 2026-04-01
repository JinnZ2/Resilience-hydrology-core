# Resilience Hydrology Core

Physics-based atmospheric water harvesting using natural environmental gradients.

## Project Overview

This system amplifies natural dew/fog formation to collect water during drought.
It uses temperature, pH, and light gradients — no pumps, no wells, no infrastructure.

Output: 0.034-0.14 mm/day depending on conditions and energy input.

## Repository Structure

```
simulations/          Python models (numpy/matplotlib/scipy)
  01_basic_dew.py       Basic dew collection simulation (ON vs OFF comparison)
  02_crop_response.py   Crop yield impact during drought
  03_seed_optimization.py  Optimal 40-bit seed finder using differential evolution
firmware/             MicroPython code for ESP32 hardware nodes
  esp32_basic/          Basic temperature logger (DS18B20 sensors)
docs/                 Documentation, build guides, research notes
  build-guide.md        Hardware builds by budget ($50-$2000)
  trailer-build.md      Real-world trailer dew collector results
  atmospheric-seed-theory.md  Research notes on seed expansion physics
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
- `SeedOptimizer` — Evolutionary seed optimization (simulations/03_seed_optimization.py)
- `TemperatureLogger` — ESP32 sensor logger (firmware/esp32_basic/main.py)

## Hardware

- ESP32 dev board + 2x DS18B20 sensors + Peltier cooler
- Total cost: $45-180 depending on build
- Field tested: northern Minnesota, November 2025
