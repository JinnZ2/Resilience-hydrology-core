# Simulations

Python models for atmospheric water harvesting.

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

### 02_crop_response.py

Models crop yield impact during drought for wheat, olive, and tomato.

```bash
python 02_crop_response.py
```

**Output**: Bar chart comparing yield with/without atmospheric water input (0.034 mm/day).

### 03_seed_optimization.py

Finds the optimal 40-bit seed for each climate zone using differential evolution.

```bash
python 03_seed_optimization.py
```

**Output**: Optimal seed bytes and parameters for arid, semi-arid, mediterranean, and tropical dry climates.
