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

⚠️ **The published seeds are not usable.** The objective is monotonically
decreasing in amplification, so every climate returns the minimum of the search
range — "optimal" here means the system turned all the way down. Bytes 3 and 4
(`wavelength`, `crop_bias`) are decoded but never read by `evaluate_seed`, so
they come back as different random values on every run at an identical score.
The file is kept as a search scaffold; it needs a defensible cost model before
its output means anything (research log, H5 and O6).
