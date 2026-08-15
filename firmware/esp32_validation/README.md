# ESP32 Validation Logger

Logs the four quantities needed to check the dew model against reality.

The basic logger (`../esp32_basic/`) records a ground/air temperature gradient.
That cannot test anything the project claims: the model predicts a **collector
surface temperature** and a nightly **volume**, driven mostly by **humidity**,
and the basic build measures none of the three.

| Logged | Sensor | Why |
|---|---|---|
| air temperature | DS18B20 | reference |
| **collector surface temperature** | DS18B20 | the model predicts this directly — cheapest test of the physics |
| **relative humidity** | SHT31-D | biggest driver of whether dew forms at all |
| **collected volume** | tipping bucket | turns anecdotes into a nightly series |

Plus two constants you write down once: **collector area** and **tilt angle**.
Their absence is why the existing field data cannot be interpreted — see
[`docs/research-log.md`](../../docs/research-log.md), H9.

## Parts

| Part | Cost |
|---|---|
| DS18B20 (second one, for the surface) | $5 |
| SHT31-D humidity sensor | $6 |
| Tipping-bucket gauge with reed switch | $12 |
| 4.7k resistors, wire | ~$1 |

About **$23** on top of an existing basic node.

## Wiring

```
GPIO4  -> DS18B20 air sensor        (4.7k pullup to 3.3V)
GPIO5  -> DS18B20 collector surface (4.7k pullup to 3.3V)
GPIO21 -> SHT31 SDA
GPIO22 -> SHT31 SCL
GPIO27 -> tipping bucket reed switch (other side to GND)
```

Bond the surface sensor to the **underside of the collector plate**, in thermal
contact and shielded from the sky. It must read the plate, not the air.

## Before you flash

Edit these in `main.py`. They are deliberately set to impossible values so an
unedited node is obvious in its own log:

```python
COLLECTOR_AREA_M2 = -1.0     # measure it
COLLECTOR_TILT_DEG = -1.0    # measure it
ML_PER_TIP = 5.0             # calibrate: pour a known volume, count tips
```

Calibrate the bucket by pouring a measured volume through it and dividing by the
tip count. An uncalibrated gauge produces confident, wrong numbers.

## Flash

```bash
pip install esptool adafruit-ampy
esptool.py --port /dev/ttyUSB0 erase_flash
esptool.py --port /dev/ttyUSB0 write_flash -z 0x1000 firmware.bin
ampy --port /dev/ttyUSB0 put main.py main.py
screen /dev/ttyUSB0 115200
```

See [`../esp32_basic/FLASH_INSTRUCTIONS.md`](../esp32_basic/FLASH_INSTRUCTIONS.md)
for detailed MicroPython setup.

## Output

`dew_log.csv` on the device flash, with the site constants in a header comment:

```
# area_m2=0.240 tilt_deg=30.0 ml_per_tip=5.00
uptime_s,t_air_c,t_surface_c,rh,tips,volume_ml
1699142400,3.20,-1.40,0.910,0,0.0
1699142700,3.10,-1.60,0.915,1,5.0
```

## What to do with the data

The comparison that closes O1:

1. Take a night's `t_air_c`, `rh`, and duration.
2. Run `DewEnergyBalance` from `simulations/04_variable_search.py` with those
   values and your recorded area and tilt.
3. Compare its predicted **surface temperature** against your logged
   `t_surface_c`, and its predicted **volume** against your logged `volume_ml`.

Surface temperature is the more informative of the two: it tests the energy
balance directly, without the collection-efficiency assumption in between.

If they disagree, the model is wrong and that is a result worth writing up. Add
it to [`docs/research-log.md`](../../docs/research-log.md) as a new hypothesis
entry — a falsification from real data would be the most valuable thing this
project has produced.
