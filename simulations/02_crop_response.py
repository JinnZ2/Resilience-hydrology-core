#!/usr/bin/env python3
"""
Crop response to atmospheric water during drought.
Shows yield impact for different crops.
"""

import numpy as np
import matplotlib.pyplot as plt

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
        
    def simulate_season(self, system_water_mm_day=0.034, drought_days=60):
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
        # Apply crop-specific tolerance
        tolerance = self.params['stress_tolerance']
        avg_stress = np.mean(daily_stress)
        yield_reduction = avg_stress ** (1.0 / tolerance)
        
        final_yield = 1.0 - yield_reduction
        
        return {
            'yield': max(0, final_yield),
            'avg_stress': avg_stress,
            'max_stress': max(daily_stress)
        }


def compare_crops():
    """Compare all crops with/without system."""
    print("="*60)
    print("Crop Response During 60-Day Drought")
    print("="*60)
    
    crops = ['wheat', 'olive', 'tomato']
    results = {}
    
    for crop in crops:
        model = CropWaterModel(crop)
        
        # Without system
        result_off = model.simulate_season(system_water_mm_day=0.0)
        
        # With system (0.034 mm/day)
        result_on = model.simulate_season(system_water_mm_day=0.034)
        
        results[crop] = {'off': result_off, 'on': result_on}
        
        print(f"\n{crop.upper()}:")
        print(f"  Without system: {result_off['yield']:.1%} yield")
        print(f"  With system:    {result_on['yield']:.1%} yield")
        improvement = (result_on['yield'] - result_off['yield']) / result_off['yield'] * 100
        print(f"  Improvement:    +{improvement:.1f}%")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(crops))
    width = 0.35
    
    yields_off = [results[c]['off']['yield'] * 100 for c in crops]
    yields_on = [results[c]['on']['yield'] * 100 for c in crops]
    
    ax.bar(x - width/2, yields_off, width, label='System OFF', color='gray', alpha=0.7)
    ax.bar(x + width/2, yields_on, width, label='System ON', color='blue', alpha=0.8)
    
    ax.set_ylabel('Yield (% of optimal)')
    ax.set_title('Crop Yield During 60-Day Drought\nWith 0.034 mm/day Atmospheric Water')
    ax.set_xticks(x)
    ax.set_xticklabels([c.title() for c in crops])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig('crop_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nGraph saved to: crop_comparison.png")


if __name__ == '__main__':
    compare_crops()
