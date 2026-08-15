# Atmospheric Seed Expansion Theory

Research notes on encoding precipitation patterns in minimal seeds.

> **Status: theory, untested.** Nothing on this page has been implemented or
> measured. It is condensed from the original research session of 2025-12-07,
> archived in full at
> [`legacy/notes/2025-12-07_seed-expansion-session.md`](../legacy/notes/2025-12-07_seed-expansion-session.md)
> — 4,404 lines, of which this page keeps a small fraction. **Priority for this
> material dates to that file.** Read it rather than this page if you intend to
> build on the ideas; most of them (adaptive strategies, layered altitude
> sensing, network architecture, deployment protocols) exist only there.

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

⚠️ This layout does **not** match what
[`simulations/03_seed_optimization.py`](../simulations/03_seed_optimization.py)
actually decodes (`amp_T`, `amp_pH`, `amp_light`, `wavelength`, `crop_bias`).
Two independent seed formats share one name. Resolving which is authoritative is
an open item — see docs/research-log.md, O6.

Physics expands this to:

1. Local fog formation (minutes-hours)
2. Cloud development (hours)
3. Gentle precipitation (hours-days)
4. Soil moisture redistribution (days)
