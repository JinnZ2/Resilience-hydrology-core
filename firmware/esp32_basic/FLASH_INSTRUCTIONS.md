# Flashing ESP32 with MicroPython

## Prerequisites

```bash
pip install esptool adafruit-ampy
```

Download MicroPython firmware from micropython.org for ESP32.

## Step 1: Erase Flash

Connect ESP32 via USB, then:

```bash
# Find your port (usually /dev/ttyUSB0 on Linux, COM3 on Windows)
ls /dev/ttyUSB*

# Erase
esptool.py --port /dev/ttyUSB0 erase_flash
```

## Step 2: Flash MicroPython

```bash
esptool.py --chip esp32 --port /dev/ttyUSB0 write_flash -z 0x1000 esp32-firmware.bin
```

Wait for "Hard resetting via RTS pin..."

## Step 3: Test Connection

```bash
screen /dev/ttyUSB0 115200
```

You should see the Python REPL (`>>>`). Press Ctrl-A then K to exit screen.

## Step 4: Upload Code

```bash
ampy --port /dev/ttyUSB0 put main.py
```

## Step 5: Run

```bash
screen /dev/ttyUSB0 115200
```

Press Ctrl-D to soft reset and run main.py.

## Troubleshooting

**"Could not open port"**
- Check USB cable (must be data cable, not charge-only)
- Install CH340 drivers for cheap ESP32 clones
- Try different USB port

**"Failed to connect"**
- Hold BOOT button while running esptool
- Try lower baud rate: `--baud 115200`

**"No module named 'machine'"**
- MicroPython not installed correctly
- Re-flash from Step 1
