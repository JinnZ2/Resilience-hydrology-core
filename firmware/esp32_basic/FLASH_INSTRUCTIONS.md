# Flashing ESP32 with MicroPython

## Prerequisites

Install tools:
```bash
pip install esptool adafruit-ampy



Download MicroPython firmware:


wget https://micropython.org/resources/firmware/esp32-20231005-v1.21.0.bin


Step 1: Erase Flash
Connect ESP32 via USB, then:

# Find your port (usually /dev/ttyUSB0 on Linux, COM3 on Windows)
ls /dev/ttyUSB*

# Erase
esptool.py --port /dev/ttyUSB0 erase_flash


Step 2: Flash MicroPython

esptool.py --chip esp32 --port /dev/ttyUSB0 write_flash -z 0x1000 esp32-20231005-v1.21.0.bin


Wait for “Hard resetting via RTS pin…”
Step 3: Test Connection

screen /dev/ttyUSB0 115200


You should see Python REPL:


>>>


Type:
>>> print("Hello from ESP32")


Press Ctrl-A then K to exit screen.
Step 4: Upload Code

# Upload main script
ampy --port /dev/ttyUSB0 put main.py

# Upload config (if you have one)
ampy --port /dev/ttyUSB0 put config.py


Step 5: Run

screen /dev/ttyUSB0 115200


Press Ctrl-D to soft reset and run main.py
Troubleshooting
“Could not open port”
	•	Check USB cable
	•	Install CH340 drivers (for cheap ESP32 clones)
	•	Try different USB port
“Failed to connect”
	•	Hold BOOT button while running esptool
	•	Try lower baud rate: --baud 115200
“No module named ‘machine’”
	•	MicroPython not installed correctly
	•	Re-flash from Step 1
Next Steps
See main.py for the logging code.
Modify config.py to change sensor pins or logging interval.

( more instructions if necessary )


firmware/esp32_basic/main.py
The cleaned-up logger version (already created above - it’s good as-is)
 Create the File Tree
Create this structure on your system:

mkdir -p resilience-hydrology/{simulations,firmware/esp32_basic,hardware,data,docs}
cd resilience-hydrology



Then copy the code blocks above into:
	•	simulations/01_basic_dew_simulation.py
	•	simulations/02_crop_response.py
	•	simulations/03_seed_optimization.py
	•	firmware/esp32_basic/main.py


# Resilience Hydrology

Open-source atmospheric water harvesting using natural environmental gradients.

## What This Does

Collects water from air during drought by amplifying natural dew formation.
- **Zero energy**: 0.034 mm/day using temperature/pH/light gradients
- **Low energy**: 0.14 mm/day with <1W solar power
- **Cost**: $30-180 depending on scale

## Quick Start

```bash
# Clone and test
git clone https://github.com/[you]/resilience-hydrology.git
cd resilience-hydrology
pip install -r requirements.txt

# Run simulation (30 seconds)
python simulations/01_basic_dew_simulation.py


(find best seed for your area)

example:

---

## Step 5: Add Your Trailer Data

Create `/hardware/trailer_dew_collector/README.md`:

```markdown
# Trailer Dew Collector

My actual working setup, northern Minnesota, November 2025.

## Photos

[You add photos here]

## Parts Used

| Item | Cost | Source |
|------|------|--------|
| ESP32 dev board | $8 | Amazon |
| DS18B20 (2×) | $10 | Amazon |
| Peltier TEC1-12706 | $15 | Amazon |
| 18650 battery + holder | $5 | Local |
| 5W solar panel | $7 | Amazon |
| Heat sink | $3 | Had it |
| Collection container | $0 | Plastic tub |
| **Total** | **$48** | |

## Results (Week 1)

| Night | Temp Range | Water Collected |
|-------|------------|-----------------|
| Nov 12 | 2°C to -3°C | 85 ml |
| Nov 13 | 3°C to -1°C | 110 ml |
| Nov 14 | 1°C to -4°C | 95 ml |
| Nov 15 | 4°C to 0°C | 120 ml |
| Nov 16 | 0°C to -5°C | 65 ml (frost) |
| Nov 17 | 3°C to -2°C | 105 ml |
| Nov 18 | 2°C to -3°C | 90 ml |

**Average**: 95 ml/night
**Collection area**: ~0.5 m²

## Issues

1. Frost on night 5 reduced output
2. Battery died day 6 (need bigger solar panel)
3. Condensation ran off sides (need better collection)

## Next Version

- Add heater element for frost prevention
- 10W solar panel instead of 5W
- Larger collection funnel

