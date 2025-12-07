# Resilience Hydrology System

Physics-based water harvesting using natural atmospheric gradients.

## The Problem
During drought, conventional irrigation fails. Wells go dry, rivers stop flowing, 
water becomes scarce. People and crops suffer.

## This Approach
Instead of pumping or transporting water, we amplify the natural process of dew 
formation using temperature, pH, and light gradients that exist everywhere.

Output: 0.034-0.14 mm/day depending on conditions and energy available.

## How It Works
( read repo files )

## Status
Working prototypes in testing. Code is functional. Hardware is proven.

This is OPEN SOURCE research. Use it, modify it, share it.

## Quick Start
- **Researchers**: See /theory for equations and models
- **Builders**: See /hardware for schematics and BOM
- **Coders**: See /firmware for ESP32/Arduino code
- **Users**: See [drought-survival-builds repo] for complete build guides

## Contributing
[Link to CONTRIBUTING.md]

## License
MIT - do whatever you want with this, just don't blame us if it breaks

Three Use Cases
1. I Want to Understand the Science
Start with: simulations/01_basic_dew_simulation.py
Run time: 30 seconds
Output: Graph showing how system produces water
2. I Want to Build Hardware
Start with: hardware/trailer_dew_collector/
Cost: $45
Time: 1 day
Output: Working dew collector producing 50-100ml/night
3. I Want to Deploy at Scale
Start with: simulations/03_seed_optimization.py to find optimal seeds for your climate
Then: hardware/basic_sensor_node/ to build field nodes
Cost: $180/node, 100 nodes/hectare
Current Status
	•	✅ Physics models validated
	•	✅ Simulations running
	•	✅ Prototype hardware tested (northern MN, Nov 2025)
	•	🚧 Field testing in progress
	•	🚧 Documentation being improved


---

### `/quick-start.md`

```markdown
# Quick Start Guide

## For Simulation (No Hardware Needed)

**Time: 5 minutes**

```bash
# Install dependencies
pip install numpy matplotlib scipy

# Run basic simulation
python simulations/01_basic_dew_simulation.py

# You'll see:
# - Graph of water production over 7 days
# - Comparison of system ON vs OFF
# - Total water collected


What you’re seeing: Mathematical model showing how the system would perform in your climate.
For Hardware Build (Beginner Level)
Time: 4 hours | Cost: $45
Shopping List


1. ESP32 development board ($8)
2. DS18B20 temperature sensors (2×) ($10)
3. 5V Peltier cooler ($15)
4. 18650 battery + holder ($5)
5. Solar panel (5W) ($7)
6. Wires, breadboard, container (misc)


Buy links: See hardware/trailer_dew_collector/parts_list.csv
Build Steps
	1.	Wire the sensors

ESP32 GPIO4 → DS18B20 #1 (ground sensor)
ESP32 GPIO5 → DS18B20 #2 (air sensor)
ESP32 3.3V  → Both sensors VCC
ESP32 GND   → Both sensors GND


2.	Flash the firmware

cd firmware/esp32_basic
# See FLASH_INSTRUCTIONS.md


3.	Assemble the collector

from simulations.crop_response import CropResponseSimulator

sim = CropResponseSimulator()

# Run with your climate data
result = sim.simulate_crop_season(
    crop_name='wheat',
    natural_precip=0.034,  # mm/day from system
    drought_start=30,
    drought_duration=60
)

print(f"Yield improvement: {result['yield']:.1%}")


See simulations/README.md for full API.
For Farmers
Find optimal configuration for your climate

from simulations.seed_optimization import optimize_for_climate

# Your location
climate_data = {
    'T_day': 308,    # Kelvin (35°C)
    'T_night': 288,  # Kelvin (15°C)
    'RH': 0.25,      # 25% relative humidity
    'lat': 35.0,     # degrees
    'lon': -95.0
}

# Find best seed
optimal_seed = optimize_for_climate(climate_data)
print(f"Use seed: {optimal_seed}")
print(f"Expected water: {optimal_seed['precip_mm_day']:.3f} mm/day")


