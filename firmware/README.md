# Firmware

Code for running on microcontrollers.

## Options

### ESP32 (Recommended)
- **Language**: MicroPython
- **Features**: Full system (sensors, actuators, LoRa)
- **Cost**: $8
- **Folder**: `esp32_basic/`

### Arduino (Simple)
- **Language**: C++/Arduino
- **Features**: Basic logging only
- **Cost**: $5
- **Folder**: `arduino_simple/`

## Quick Start (ESP32)

```bash
# 1. Install esptool
pip install esptool

# 2. Flash MicroPython
esptool.py --port /dev/ttyUSB0 erase_flash
esptool.py --port /dev/ttyUSB0 write_flash -z 0x1000 firmware.bin

# 3. Upload code
ampy --port /dev/ttyUSB0 put esp32_basic/main.py

# 4. Connect and run
screen /dev/ttyUSB0 115200
