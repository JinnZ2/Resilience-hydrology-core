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

## Results (7 days, Nov 2025, northern MN)
- Night 1: 85ml
- Night 2: 110ml
- [data...]
Average: 95ml/night

> **Not comparable to model output.** Collector area was never recorded, so
> ml/night cannot be converted to the mm/day the simulations report. No unpowered
> control ran alongside, so this figure is total collection, not collection
> attributable to the system. Two of the seven nights failed (see below), so the
> average covers an unstated subset. This build does **not** validate the models.
> See [method-log.md](method-log.md) M-07 for what a comparable run needs:
> collector area in m², a paired unpowered control, per-night T_day / T_night /
> RH, and every night logged including failures.

## Code
See [`../firmware/esp32_basic/`](../firmware/esp32_basic/) — note that the
firmware carried a GPIO5 pin collision (air sensor vs. SD chip-select) until
2026-08. If this build logged to an SD card, its air temperatures may be
unreliable. See method-log.md M-06.

## Failures
- Condensation froze on night 4 (need insulation)
- Battery died day 6 (need bigger panel)

## Next iteration
- Add heater for frost protection
- Double solar panel size
