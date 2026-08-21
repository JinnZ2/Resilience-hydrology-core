# Which Build?

Two builds, two climates. Pick by your site, not by preference — the mechanisms
work in different air, and the wrong one produces nothing at all.

## Answer one question first

**What is the relative humidity at your site just before dawn?**

Not the daytime value, which is much lower at the same site. Pre-dawn, when air
is coldest and closest to saturation. If you do not know it, measuring it is the
first job: a $6 sensor settles it
([`../firmware/esp32_validation/`](../firmware/esp32_validation/)).

Everything below depends on that number, and guessing it wrong wastes the build.

## Then choose

| Your site | Build | Why |
|---|---|---|
| **Cool nights, above ~70% pre-dawn RH** | **[A — Dew collector](build-dew.md)** | Free, passive, no consumables. Nothing to run out of. |
| **Warm nights, above ~90% RH** | **[A — Dew collector](build-dew.md)** | Works, but the margin is thin. Measure before committing. |
| **Below ~60% RH, any temperature** | **[B — Sorbent harvester](build-sorbent.md)** | Dew yields *exactly zero* here. Sorption still works. |
| **Below ~30% RH** | **[B](build-sorbent.md)**, and read the salt section | Calcium chloride stops working; the choices get narrower and the safety notes matter more. |
| **Below ~11% RH** | Neither, honestly | Past where any of this is known to work. Storage and demand reduction will beat harvesting. |

Between 60% and 70%, either could be marginal. Measure for a week before
spending.

## Why the split exists

Dew forms when a surface radiates heat to the sky until it falls below the dew
point of the air. In dry air the dew point is far below ambient — 22 K below at
32 °C and 25% RH — and radiative cooling only delivers 3–9 K. The surface never
gets there, so the yield is not small, it is **zero**. No tilt, coating or budget
crosses that wall.

A sorbent does not need the air to reach saturation. It pulls vapour out
chemically at humidity where condensation is impossible, then gives it back when
heated. It costs energy — as heat, from the sun — and it needs a consumable salt
and more care about what ends up in the water.

Free but gated, versus costly but always available. That is the whole decision.
The physics is in [`alternative-systems.md`](alternative-systems.md); the
falsification that forced the split is
[`research-log.md`](research-log.md) H11–H13.

## Honest comparison

| | A — Dew | B — Sorbent |
|---|---|---|
| Works at | ≥70% RH cool, ≥90% warm | 11–100% RH, salt-dependent |
| Energy | none | sunlight, as heat |
| Consumables | none | salt (regenerates for years) |
| Cost | ~$15 without instruments | ~$8–45 depending on salt |
| Yield | 8–180 mL/m²/night, humidity-dependent | 240–720 mL/m²/day modelled |
| Drinking-water risk | low | **salt carryover — read the safety notes** |
| Complexity | one afternoon | a weekend, plus a cycle to tune |
| Built and measured by this project | partially (2 nights) | **never** |

Neither has been validated against field data by this project. Build A's model
is ours and unvalidated; Build B's material properties are other people's
measurements of other people's hardware. Both are honest starting points and
neither is a promise.

## Both builds want the same instruments

Whatever you build, the measurement kit is the highest-value $23 in this
project — it is what turns a build into evidence:

| Item | Cost | Why |
|---|---|---|
| Humidity sensor (SHT31) | $6 | Decides which build you should have made |
| Temperature sensors (DS18B20) | $5 | Surface vs air is what the models predict |
| Tipping-bucket or graduated container | $12 | Turns anecdotes into a series |
| Tape measure and protractor | free | Collector area and angle. Their absence is why the one existing field result cannot be interpreted at all |

Firmware: [`../firmware/esp32_validation/`](../firmware/esp32_validation/).

## Post your results

Successes and failures both, with the conditions recorded. The trailer build in
[`trailer-build.md`](trailer-build.md) records icing and a dead battery and is
more useful for it.

If you build the sorbent version, **you will be the first**, and the numbers you
take will be the best evidence this repository has.

## Safety

Build A amplifies a natural process and is about as dangerous as a bucket.

Build B involves concentrated brine and produces drinking water — read
[the safety section](build-sorbent.md#safety--read-before-drinking-anything)
before drinking anything it makes. The short version: never drink water that
tastes salty, use food-grade calcium chloride, and do not use lithium chloride
for drinking water without testing.
