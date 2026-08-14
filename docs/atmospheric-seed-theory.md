# Atmospheric Seed Expansion Theory

Research notes on encoding precipitation patterns in minimal seeds.

> **Status: unimplemented theory.** The physics described here belongs to an
> ion-coupling PDE model that was never committed to this repository. Only its
> printed outputs survive, in
> [`../legacy/2025-original/firmware__02_crop_response.md`](../legacy/2025-original/firmware__02_crop_response.md).
> The retired "0.034 / 0.14 mm/day" figures came from *this* model, not from the
> code now in `simulations/` (see [method-log.md](method-log.md) M-01).
>
> The 40-bit layout below also does **not** match the layout
> `simulations/03_seed_optimization.py` actually decodes. Same bit count,
> different meanings, no stated correspondence (M-08). Note that "wavelength"
> appears in both meaning different things — a horizontal atmospheric pattern
> scale here, an optical wavelength in nm there.

## Core Insight

If we can:

1. Encode desired fog/cloud/precipitation patterns in a minimal seed
2. Let atmospheric physics expand it into actual weather
3. Use ion/charge modulation as the atmospheric equivalent of orbital delta-V impulses
4. Create fractal patterns that grow from local to regional scales

...we could potentially orchestrate gentle, distributed hydration instead of brute-force cloud seeding or water transportation.

## Atmospheric vs Orbital Analogies

| Orbital System | Atmospheric System |
|---|---|
| Delta-V impulses | Ion/charge injections |
| Orbital harmonics | Atmospheric resonance modes |
| Phase rate monitoring | Humidity/temperature tracking |
| 3-satellite prime network | 3-altitude layered sensing |
| Solar storm noise | Weather front interference |
| Seed -> orbital schedule | Seed -> precipitation pattern |

## The Physics "Decompressor"

```
Ion concentration (n_i) -> affects temperature (Q_ion = -alpha * n_i)
Temperature -> affects saturation (S = e / e_s(T))
Supersaturation -> triggers droplet nucleation (J_ion = gamma * n_i * max(S-1, 0))
Droplets -> grow to precipitation size
```

Control variable: Ion production rate S_ion(z, t)

## 40-Bit Seed Encoding

```
[0-7]:   Near-surface ion production amplitude
[8-15]:  Altitude modulation frequency
[16-23]: Horizontal pattern wavelength
[24-31]: Temporal modulation pattern
[32-39]: Energy budget allocation
```

Physics expands this to:

1. Local fog formation (minutes-hours)
2. Cloud development (hours)
3. Gentle precipitation (hours-days)
4. Soil moisture redistribution (days)
