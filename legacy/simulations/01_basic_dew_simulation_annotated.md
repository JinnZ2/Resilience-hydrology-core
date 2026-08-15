---

### `/simulations/01_basic_dew_simulation.py`

```python
#!/usr/bin/env python3
"""
Basic Dew Simulation - Quick Start Version

Simulates atmospheric water collection using natural gradients.
Run time: ~30 seconds
Output: Graph showing water production over 7 days

Usage:
    python 01_basic_dew_simulation.py
    
    # Or customize:
    python 01_basic_dew_simulation.py --climate arid --days 14
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class DewSimulator:
    """
    Simplified dew formation simulator.
    
    Physics:
    - Temperature inversion at night drives condensation
    - System amplifies natural process 2-4×
    - Energy input: zero (natural mode) or <1W (boosted)
    """
    
    def __init__(self, T_day=305, T_night=288, RH=0.30, lat=42.0):
        """
        Initialize simulator.
        
        Args:
            T_day: Daytime temperature (Kelvin)
            T_night: Nighttime temperature (Kelvin)
            RH: Relative humidity (0-1)
            lat: Latitude (degrees)
        """
        self.T_day = T_day
        self.T_night = T_night
        self.RH = RH
        self.lat = lat
        
        # Physical constants
        self.dew_point_offset = 5.0  # K below air temp
        self.collection_efficiency = 0.7  # 70% of condensation collected
        
    def simulate_day(self, day_num, system_active=False):
        """
        Simulate one 24-hour period.
        
        Returns:
            water_ml: milliliters of water collected
        """
        # Temperature cycle (simplified sinusoid)
        hours = np.linspace(0, 24, 25)
        T_cycle = self.T_night + (self.T_day - self.T_night) * np.sin(np.pi * hours / 24) ** 2
        
        # Dew formation window (when T < dew point)
        T_dew = self.T_night - self.dew_point_offset
        dew_window = T_cycle < T_dew
        
        # Natural dew formation rate (g/m²/hour)
        natural_rate = self.RH * (T_dew - T_cycle[dew_window].mean()) * 2.0 if np.any(dew_window) else 0
        
        # System amplification
        if system_active:
            amplification = 3.0  # 3× natural rate
        else:
            amplification = 1.0
        
        # Total water collected (assuming 1m² collection area)
        dew_hours = np.sum(dew_window)
        water_grams = natural_rate * amplification * dew_hours
        water_ml = water_grams * self.collection_efficiency
        
        return max(0, water_ml)
    
    def run(self, days=7, system_active=True):
        """
        Run multi-day simulation.
        
        Returns:
            dict with results
        """
        results = {
            'days': [],
            'water_daily_ml': [],
            'water_cumulative_ml': [],
        }
        
        cumulative = 0
        
        for day in range(days):
            daily_water = self.simulate_day(day, system_active)
            cumulative += daily_water
            
            results['days'].append(day + 1)
            results['water_daily_ml'].append(daily_water)
            results['water_cumulative_ml'].append(cumulative)
        
        results['total_water_ml'] = cumulative
        results['avg_daily_ml'] = cumulative / days
        
        return results
    
    def plot_results(self, results_on, results_off):
        """Create comparison plot."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Daily water
        ax1.bar(results_off['days'], results_off['water_daily_ml'], 
                alpha=0.6, label='System OFF', color='gray')
        ax1.bar(results_on['days'], results_on['water_daily_ml'], 
                alpha=0.8, label='System ON', color='blue')
        ax1.set_xlabel('Day')
        ax1.set_ylabel('Water Collected (ml/day)')
        ax1.set_title('Daily Water Production')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Cumulative water
        ax2.plot(results_off['days'], results_off['water_cumulative_ml'], 
                'o-', label='System OFF', color='gray', linewidth=2)
        ax2.plot(results_on['days'], results_on['water_cumulative_ml'], 
                'o-', label='System ON', color='blue', linewidth=2)
        ax2.set_xlabel('Day')
        ax2.set_ylabel('Cumulative Water (ml)')
        ax2.set_title('Total Water Collected')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


def main():
    """Run basic simulation and display results."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simulate dew collection')
    parser.add_argument('--climate', default='semi_arid', 
                       choices=['arid', 'semi_arid', 'mediterranean', 'tropical_dry'])
    parser.add_argument('--days', type=int, default=7)
    parser.add_argument('--output', default='dew_simulation.png')
    
    args = parser.parse_args()
    
    # Climate presets
    climates = {
        'arid': {'T_day': 308, 'T_night': 288, 'RH': 0.25},
        'semi_arid': {'T_day': 303, 'T_night': 290, 'RH': 0.35},
        'mediterranean': {'T_day': 298, 'T_night': 292, 'RH': 0.45},
        'tropical_dry': {'T_day': 305, 'T_night': 295, 'RH': 0.40},
    }
    
    climate_params = climates[args.climate]
    
    print(f"Simulating {args.days} days in {args.climate} climate...")
    print(f"  Day temp: {climate_params['T_day']-273:.1f}°C")
    print(f"  Night temp: {climate_params['T_night']-273:.1f}°C")
    print(f"  Humidity: {climate_params['RH']*100:.0f}%")
    print()
    
    # Create simulator
    sim = DewSimulator(**climate_params)
    
    # Run with system OFF
    print("Running simulation: System OFF...")
    results_off = sim.run(days=args.days, system_active=False)
    
    # Run with system ON
    print("Running simulation: System ON...")
    results_on = sim.run(days=args.days, system_active=True)
    
    # Display results
    print("\nRESULTS:")
    print(f"  System OFF: {results_off['total_water_ml']:.0f} ml total "
          f"({results_off['avg_daily_ml']:.1f} ml/day)")
    print(f"  System ON:  {results_on['total_water_ml']:.0f} ml total "
          f"({results_on['avg_daily_ml']:.1f} ml/day)")
    
    improvement = (results_on['total_water_ml'] / results_off['total_water_ml'] - 1) * 100
    print(f"  Improvement: +{improvement:.0f}%")
    print()
    
    # For perspective
    print("For context:")
    print(f"  {results_on['total_water_ml']:.0f} ml = ~{results_on['total_water_ml']/250:.1f} cups of water")
    print(f"  Collection area: 1 m² (adjust linearly for larger areas)")
    print()
    
    # Generate plot
    print(f"Generating plot: {args.output}")
    fig = sim.plot_results(results_on, results_off)
    fig.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Done!")


if __name__ == '__main__':
    main()
