# Trailer Dew Collector (v0.1)

## What I built
[PHOTO]

## Why
I live in a trailer and needed drinking water without infrastructure.

## Parts
- 2× DS18B20 temp sensors ($10)
- ESP32 dev board ($8)  
- Peltier cooler ($15)
- 18650 battery + solar ($12)
- Collection container (had it)

Total: $45

## Results (Nov 2025, northern MN)
- Night 1: 85ml
- Night 2: 110ml
- Nights 3-7: not recorded

**Recorded: 2 nights, 85 and 110 ml.** An earlier version of this page reported
"Average: 95ml/night" over 7 days. Only two nights are written down, and the
failure notes below (frost on night 4, dead battery on day 6) imply fewer than
seven usable nights, so the average is withdrawn until the missing nights turn
up. See docs/research-log.md, O5.

**Not recorded, and needed**: collector surface area. Without it these volumes
cannot be converted to mm/day per m², which is the unit the simulations output —
so this dataset cannot currently be compared to any model in the repository.
Anyone repeating this build: measure the area (research log, O1).

> **Not comparable to model output.** Collector area was never recorded, so
> ml/night cannot be converted to the mm/day the simulations report. No unpowered
> control ran alongside, so this figure is total collection, not collection
> attributable to the system. Two nights failed, so any average covers an
> unstated subset. This build does **not** validate the models. See
> [method-log.md](method-log.md) M-07 for what a comparable run needs: collector
> area in m², a paired unpowered control, per-night T_day / T_night / RH, and
> every night logged including failures.

## Code
See [firmware/esp32_basic/](../firmware/esp32_basic/) — `main.py` logs the
ground/air temperature gradient. There is no separate trailer-specific firmware.

Note that this firmware carried a **GPIO5 pin collision** (air sensor OneWire vs
SD chip-select) until 2026-08. If this build logged to an SD card, its air
temperatures may be unreliable — see [method-log.md](method-log.md) M-06. That
would affect any attempt to reconstruct the conditions of these two nights.

For a node that can actually test the models, see
[`../firmware/esp32_validation/`](../firmware/esp32_validation/).

## Failures
- Condensation froze on night 4 (need insulation)
- Battery died day 6 (need bigger panel)

## Next iteration

**Revised 2026-08-15.** This section used to read "add heater for frost
protection, double solar panel size". Both were aimed at keeping the Peltier
running, and the analysis says the Peltier is the part to remove. Original
wording preserved at
[`legacy/2025-original/trailer-build.md`](../legacy/2025-original/trailer-build.md);
reasoning in [`research-log.md`](research-log.md), H8 and Round 3.

In order, cheapest first:

1. **Run it in September, not November** — free. The model puts 47% of November
   nights at this site below freezing and only 4% making water. September: 0%
   frozen, 16% productive. The night-4 frost was the season, not the hardware.
2. **Move it clear of the trailer** — free. The trailer blocks the cold sky the
   collector has to radiate to.
3. **Angle it to ~30° and write the angle down** — $3.
4. **Foam block under the collector** — $4. This is the real fix for night 4.
5. **Remove the Peltier** — recovers $15, which pays for steps 3 and 4. At this
   build's energy budget it contributes about 6% of the cooling the surface
   already does for free.
6. **Measure the collector area** — free, and see below.

The frost heater and the bigger panel are not on this list. Both spend money to
support a component that the model says is not earning its place.

## What this build cannot tell us yet

Two numbers were never recorded: the **collector area** and the **tilt angle**.

Without them these results cannot be compared to any model in the repository.
The model run on a poorly configured collector at this site gives 0.05 mL/night;
run on a well configured one it gives a mean of 35 mL with a best night around
140 mL, which brackets the 85 and 110 recorded here. Both readings are
consistent with the data, and nothing in the notebook distinguishes them.

A tape measure and a protractor would have settled the configuration question. A
paired unpowered control would have settled the attribution question — without
one, even a perfectly recorded volume cannot separate "the system worked" from
"dew formed, as it does on any cold surface". See
[`research-log.md`](research-log.md) H9 and O1, and
[`method-log.md`](method-log.md) M-07.
