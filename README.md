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
