#!/usr/bin/env python3
"""
Crop response to atmospheric water during drought.
Shows yield impact for different crops.

Usage:
    python 02_crop_response.py
    python 02_crop_response.py --water 0.27 --drought-days 60

The default water input (0.034 mm/day) is inherited from a retired ion-coupling
model, NOT from 01_basic_dew.py, which produces 0.16-0.30 mm/day. See
docs/method-log.md (M-01) before treating the default as this system's output.
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt

# Legacy default. Provenance: legacy/2025-original/firmware__02_crop_response.md
# ("natural gradient coupling", ~0 kWh/day). Kept as the default so historical
# runs stay reproducible; override with --water to use the current dew model.
LEGACY_WATER_MM_DAY = 0.034

class CropWaterModel:
    """Simplified crop water stress model."""
    
    # Crop water needs by growth stage (mm/day)
    CROPS = {
        'wheat': {
            'needs': [1.5, 3.0, 4.0, 3.5, 1.0],
            'stage_days': [10, 25, 30, 25, 30],
            'stress_tolerance': 1.5,  # exponent (higher = more tolerant)
        },
        'olive': {
            'needs': [0.8, 1.2, 1.5, 1.0, 0.5],
            'stage_days': [120, 30, 20, 60, 135],
            'stress_tolerance': 2.0,  # Very tolerant
        },
        'tomato': {
            'needs': [1.0, 2.5, 4.5, 3.0, 1.5],
            'stage_days': [20, 30, 20, 40, 20],
            'stress_tolerance': 0.7,  # Sensitive
        },
    }
    
    def __init__(self, crop='wheat'):
        self.crop = crop
        self.params = self.CROPS[crop]
        
    def simulate_season(self, system_water_mm_day=LEGACY_WATER_MM_DAY, drought_days=60):
        """
        Simulate full crop season with drought period.
        
        Args:
            system_water_mm_day: Water from atmospheric system
            drought_days: Length of drought period
            
        Returns:
            Final yield as fraction of optimal (0-1)
        """
        total_days = sum(self.params['stage_days'])
        
        # Soil moisture tracking (simplified)
        soil_capacity = 150  # mm available water
        soil_moisture = soil_capacity * 0.8  # Start at 80%
        
        daily_stress = []
        
        for day in range(total_days):
            # Determine growth stage
            stage = 0
            days_so_far = 0
            for i, stage_length in enumerate(self.params['stage_days']):
                if day < days_so_far + stage_length:
                    stage = i
                    break
                days_so_far += stage_length
            
            # Crop water demand
            demand = self.params['needs'][stage]
            
            # Water supply
            if 30 <= day < 30 + drought_days:
                # Drought period - only system water
                supply = system_water_mm_day
            else:
                # Normal - adequate rain
                supply = demand * 1.2  # 120% of need
            
            # Update soil moisture
            soil_moisture += supply - demand
            soil_moisture = max(0, min(soil_moisture, soil_capacity))
            
            # Calculate stress (0=no stress, 1=maximum stress)
            stress_fraction = 1.0 - (soil_moisture / soil_capacity)
            stress_fraction = max(0, min(1, stress_fraction))
            
            daily_stress.append(stress_fraction)
        
        # Calculate yield reduction
        # Apply crop-specific tolerance.
        # stress is in [0,1], so a LARGER exponent gives a SMALLER reduction --
        # i.e. exponent must be the tolerance itself, not its reciprocal.
        # The reciprocal form inverted the ranking (M-04).
        tolerance = self.params['stress_tolerance']
        avg_stress = np.mean(daily_stress)
        yield_reduction = avg_stress ** tolerance
        
        final_yield = 1.0 - yield_reduction
        
        return {
            'yield': max(0, final_yield),
            'avg_stress': avg_stress,
            'max_stress': max(daily_stress)
        }


def compare_crops(water_mm_day=LEGACY_WATER_MM_DAY, drought_days=60):
    """Compare all crops with/without system."""
    print("="*60)
    print(f"Crop Response During {drought_days}-Day Drought")
    print(f"System water: {water_mm_day} mm/day")
    print("="*60)

    crops = ['wheat', 'olive', 'tomato']
    results = {}

    for crop in crops:
        model = CropWaterModel(crop)

        # Without system
        result_off = model.simulate_season(system_water_mm_day=0.0,
                                           drought_days=drought_days)

        # With system
        result_on = model.simulate_season(system_water_mm_day=water_mm_day,
                                          drought_days=drought_days)

        results[crop] = {'off': result_off, 'on': result_on}
        
        print(f"\n{crop.upper()}:")
        print(f"  Without system: {result_off['yield']:.1%} yield")
        print(f"  With system:    {result_on['yield']:.1%} yield")
        if result_off['yield'] > 0:
            improvement = (result_on['yield'] - result_off['yield']) / result_off['yield'] * 100
            print(f"  Improvement:    {improvement:+.1f}% relative "
                  f"({(result_on['yield'] - result_off['yield']) * 100:+.1f} points)")
        else:
            print(f"  Improvement:    n/a (baseline yield is zero)")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(crops))
    width = 0.35
    
    yields_off = [results[c]['off']['yield'] * 100 for c in crops]
    yields_on = [results[c]['on']['yield'] * 100 for c in crops]
    
    ax.bar(x - width/2, yields_off, width, label='System OFF', color='gray', alpha=0.7)
    ax.bar(x + width/2, yields_on, width, label='System ON', color='blue', alpha=0.8)
    
    ax.set_ylabel('Yield (% of optimal)')
    ax.set_title(f'Crop Yield During {drought_days}-Day Drought\n'
                 f'With {water_mm_day} mm/day Atmospheric Water')
    ax.set_xticks(x)
    ax.set_xticklabels([c.title() for c in crops])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig('crop_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nGraph saved to: crop_comparison.png")


def main():
    parser = argparse.ArgumentParser(description='Crop response to atmospheric water')
    parser.add_argument('--water', type=float, default=LEGACY_WATER_MM_DAY,
                        help='System water input in mm/day (default: %(default)s, '
                             'the retired ion-coupling figure -- see docs/method-log.md)')
    parser.add_argument('--drought-days', type=int, default=60)
    args = parser.parse_args()

    compare_crops(water_mm_day=args.water, drought_days=args.drought_days)


if __name__ == '__main__':
    main()
