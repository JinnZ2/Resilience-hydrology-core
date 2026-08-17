# Firmware

MicroPython code for ESP32 microcontrollers.

## ESP32 Basic Logger

- **Language**: MicroPython
- **Features**: Temperature gradient logging (ground vs air)
- **Hardware**: ESP32 + 2x DS18B20 sensors
- **Cost**: ~$18
- **Code**: `esp32_basic/main.py`

## Pin Map

| GPIO | Function |
|---|---|
| 4 | DS18B20 #1 (ground sensor), 4.7k pullup |
| 5 | DS18B20 #2 (air sensor), 4.7k pullup |
| 18 / 23 / 19 | SD card SCK / MOSI / MISO (optional) |
| 15 | SD card CS (optional) |
| 2 | Status LED |

> SD chip-select was on GPIO5 until 2026-08, colliding with the air sensor's
> OneWire bus. With a card fitted, air readings — and therefore `delta_t`, the
> only quantity this node exists to measure — were silently logged as `NA`.
> Moved to GPIO15; `TemperatureLogger.__init__` now raises on any future
> collision. **Unverified on hardware** (GPIO15 is an ESP32 strapping pin, must
> be high at boot). Confirm and record in `docs/method-log.md` M-06.

## Quick Start

```bash
# 1. Install tools
pip install esptool adafruit-ampy

# 2. Flash MicroPython (see esp32_basic/FLASH_INSTRUCTIONS.md)
esptool.py --port /dev/ttyUSB0 erase_flash
esptool.py --port /dev/ttyUSB0 write_flash -z 0x1000 firmware.bin

# 3. Upload code
ampy --port /dev/ttyUSB0 put esp32_basic/main.py main.py

# 4. Connect and run
screen /dev/ttyUSB0 115200
```

See `esp32_basic/FLASH_INSTRUCTIONS.md` for detailed setup steps.
