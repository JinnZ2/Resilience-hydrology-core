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
