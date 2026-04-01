# Firmware

MicroPython code for ESP32 microcontrollers.

## ESP32 Basic Logger

- **Language**: MicroPython
- **Features**: Temperature gradient logging (ground vs air)
- **Hardware**: ESP32 + 2x DS18B20 sensors
- **Cost**: ~$18
- **Code**: `esp32_basic/main.py`

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
