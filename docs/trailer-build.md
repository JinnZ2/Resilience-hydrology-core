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

## Code
See [firmware/esp32_basic/](../firmware/esp32_basic/) — `main.py` logs the
ground/air temperature gradient. There is no separate trailer-specific firmware.

## Failures
- Condensation froze on night 4 (need insulation)
- Battery died day 6 (need bigger panel)

## Next iteration
- Add heater for frost protection
- Double solar panel size
