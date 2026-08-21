# Build A — Dew Collector (cool humid sites)

A passive radiative dew collector. No pump, no power, no consumables.

**This is one of two build paths.** It works where nights are cool and humid and
produces *nothing* where they are hot and dry — see
[`build-guide.md`](build-guide.md) to choose, or
[`build-sorbent.md`](build-sorbent.md) for the dry-air build.

> **Revised 2026-08-15.** This guide used to lead with hardware and budget
> tiers. It now leads with two free decisions, because the transition analysis
> found that every hardware upgrade in it is close to worthless until those two
> decisions are made — and large afterwards. The original guide is preserved at
> [`legacy/2025-original/BUILD-README.md`](../legacy/2025-original/BUILD-README.md);
> what changed and why is in [`research-log.md`](research-log.md), Round 3.

## First: is dew the right mechanism for your site?

**Check this before spending anything.** Dew is not a mechanism that works
poorly in dry air — it stops entirely. Below the point where the dew-point
depression exceeds the 3–9 K that radiative cooling can deliver, the yield is
exactly zero, and nothing in this guide changes that.

Modelled yield, mL/m² per night, by **pre-dawn** air temperature and humidity:

| RH | 15 °C | 22 °C | 32 °C |
|---|---|---|---|
| ≤50% | 0 | 0 | 0 |
| 70% | 8 | 1 | 0 |
| 80% | 59 | 17 | 0 |
| 90% | 179 | 130 | 6 |

- **Cool nights above ~70% RH** → build what this guide describes.
- **Hot nights, or below ~60% RH** → this guide will not produce water for you.
  Go to [`build-sorbent.md`](build-sorbent.md), which harvests down to about
  11% RH — where severe drought actually sits.

Note that these are *pre-dawn* values, which are far higher than the daytime
humidity at the same site. If you do not know your pre-dawn RH, that is the
first thing to measure — a $6 sensor settles it
([`firmware/esp32_validation/`](../firmware/esp32_validation/)).

## Build in this order

The order is the finding. Steps 1 and 2 cost nothing and unlock everything after
them. In the model, a collector angled to 30° gains **+0.1 mL/night** on its own
and **+10.9 mL/night** once steps 1 and 2 are done. A bracket cannot improve a
night that was never going to condense.

### 1. Run it in the dew season — free

A site that freezes on half its nights is not a dew site on those nights, and no
hardware fixes that. The one field deployment this project has ran in northern
Minnesota in **November**, where the model puts **47% of nights below freezing**
and only **4% making water**. The same site in **September**: 0% frozen, 16%
making water.

Frost is not a malfunction to be engineered around. It is a signal that you are
running in the wrong month.

### 2. Put it under open sky, out of the wind — free

Canopy openness ranks above every hardware variable except tilt. A collector
tucked beside a trailer or under a tree loses the cold sky it must radiate to —
that sky is the entire cooling mechanism.

Wind both feeds vapour to the surface and warms it, so there is a middle band
rather than "less is better". A light breeze is fine; exposed and gusty is not.

### 3. Angle the collector to about 30° — $3

The largest design lever in the model, with a genuine best range of roughly
**19–53°**. Steeper drains droplets into the vessel; too steep and the surface
sees less cold sky. A scrap bracket does it.

**Write the angle down.** It has never been recorded on any build, which is why
the one field measurement cannot be checked against any model (see
"Why this matters" below).

### 4. Insulate the mount — $4

A foam block between collector and support stops the mount conducting heat back
into the surface you are trying to keep cold. This also addresses the night-4
frost failure in the trailer log.

### 5. Do not fit a Peltier cooler — saves $15

At the energy budget these builds actually have — roughly 3 W/m² electrical from
a small panel and one 18650 across a long night — active cooling delivers about
**6% of the radiative cooling the surface already does for free**. It is worth
about **1.11x**, and reaching the 3x this project used to claim would need
roughly **19x the power available**.

If your build already has one, removing it recovers the part cost and pays for
steps 3 and 4 with change left over. See [`research-log.md`](research-log.md),
H8.

## Budget tiers

| Budget | What you get |
|---|---|
| **$0** | Steps 1–2, plus step 5 if you already own a Peltier (this tier *pays you*) |
| **< $15** | Steps 1–5 complete: season, siting, tilt, insulation, no Peltier |
| **< $40** | Add the measurement kit below — the highest-value spend in the project |
| **< $75** | Double the collector area. Area is the only lever that scales linearly and never disappoints |

Only the first tier is fully written up, as
[trailer-build.md](trailer-build.md). Field Node, Complete Field System, and
Farm-Scale Deployment are *planned, not written*.

## The measurement kit — $23

Nothing here produces water. It is still the best money in this guide, because
without it no claim the project makes can be checked.

| Item | Cost | Closes |
|---|---|---|
| Tape measure on the collector, write down the area | free | half of O1 |
| Protractor on the bracket, write down the tilt | free | O9 |
| Third DS18B20 bonded to the collector plate | $5 | tests the physics directly |
| SHT31 humidity sensor | $6 | the biggest driver, currently unmeasured |
| Tipping-bucket gauge and counter | $12 | turns anecdotes into a series |

Firmware for the last three is in
[`firmware/esp32_validation/`](../firmware/esp32_validation/).

**Why this matters.** The trailer build reported 85 ml and 110 ml on two nights.
Nobody recorded the collector's area or angle. Run the model on a *poorly*
configured collector at that site and you get 0.05 mL/night; run it on a
*well* configured one and you get a mean of 35 with a best night of 140 — which
brackets the measurement neatly. So we cannot tell whether the model is wrong or
the assumed configuration is wrong. Two numbers nobody wrote down are the
difference between a validated model and a stalled project.

## Build Difficulty
- ⭐ = Hand tools, no electronics knowledge
- ⭐⭐ = Basic soldering, can follow tutorials
- ⭐⭐⭐ = Comfortable with Arduino/code
- ⭐⭐⭐⭐ = Can design and debug systems

Steps 1–5 are all ⭐. The measurement kit is ⭐⭐.

## Climate Zones
The simulations carry presets for:
- Arid (hot deserts)
- Semi-arid (dry grasslands)
- Mediterranean (dry summers)
- Tropical dry (monsoon climate)

> **Withdrawn:** builds do *not* ship seeds optimised per climate. The seed
> optimiser's objective is degenerate — it returns minimum amplification for
> every climate, and two of its five bytes have no effect on the result at all.
> See [method-log.md](method-log.md) M-03 and [research-log.md](research-log.md)
> H5. For siting and configuration
decisions use `simulations/04_variable_search.py` and
`simulations/05_transition_paths.py` instead.

## Already built one?

`python simulations/05_transition_paths.py` takes an existing unit and returns
the cheapest ordered set of changes, with costs, hours, and who has to act. Its
first stage costs **negative three dollars**.

## Start Building
Pick your build, get the parts, follow the guide.

Post your results — success or failure, and *with the area and angle recorded* —
so others can learn.

## Safety
This amplifies natural processes. It's not weather modification.
It's not dangerous. But don't be stupid.

## Questions?
Open an issue. Field measurements — especially collector area alongside nightly
volume — are the single most useful thing you can contribute.
