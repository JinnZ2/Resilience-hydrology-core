#!/usr/bin/env python3
"""
Basic atmospheric water collection simulation.
Shows system ON vs OFF comparison over 7 days.
"""

import numpy as np
import matplotlib.pyplot as plt

class SimpleDewSimulator:
    """Minimal dew formation model for quick understanding."""
    
    def __init__(self, T_day=305, T_night=288, RH=0.30):
        self.T_day = T_day
        self.T_night = T_night  
        self.RH = RH
        
    def simulate_night(self, system_on=False):
        """
        Simulate one night of dew formation.
        
        Returns:
            water_mm: millimeters of water collected
        """
        # Natural dew formation (simplified)
        delta_T = self.T_day - self.T_night
        natural_dew = self.RH * delta_T * 0.02  # mm/night
        
        if system_on:
            # System amplifies by 3-4x through gradient enhancement
            amplification = 3.0
        else:
            amplification = 1.0
            
        return natural_dew * amplification
    
    def run(self, days=7, system_on=True):
        """Run multi-day simulation."""
        daily_water = []
        
        for day in range(days):
            # Night dew collection
            water = self.simulate_night(system_on)
            daily_water.append(water)
        
        return {
            'daily_mm': daily_water,
            'total_mm': sum(daily_water),
            'avg_mm_day': sum(daily_water) / days
        }
    
    def plot_comparison(self, days=7):
        """Plot system ON vs OFF."""
        results_off = self.run(days, system_on=False)
        results_on = self.run(days, system_on=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Daily comparison
        x = range(1, days+1)
        ax1.bar(x, results_off['daily_mm'], alpha=0.6, label='System OFF', color='gray')
        ax1.bar(x, results_on['daily_mm'], alpha=0.8, label='System ON', color='blue')
        ax1.set_xlabel('Day')
        ax1.set_ylabel('Water (mm/day)')
        ax1.set_title('Daily Water Production')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Cumulative
        cumul_off = np.cumsum(results_off['daily_mm'])
        cumul_on = np.cumsum(results_on['daily_mm'])
        ax2.plot(x, cumul_off, 'o-', label='System OFF', linewidth=2, color='gray')
        ax2.plot(x, cumul_on, 'o-', label='System ON', linewidth=2, color='blue')
        ax2.set_xlabel('Day')
        ax2.set_ylabel('Cumulative Water (mm)')
        ax2.set_title('Total Water Collected')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Print summary
        print(f"\n{days}-Day Results:")
        print(f"  System OFF: {results_off['total_mm']:.3f} mm total ({results_off['avg_mm_day']:.4f} mm/day)")
        print(f"  System ON:  {results_on['total_mm']:.3f} mm total ({results_on['avg_mm_day']:.4f} mm/day)")
        improvement = (results_on['total_mm'] / results_off['total_mm'] - 1) * 100
        print(f"  Improvement: +{improvement:.0f}%\n")
        
        return fig


def main():
    """Quick demo."""
    print("="*50)
    print("Basic Dew Collection Simulation")
    print("="*50)
    
    # Create simulator with semi-arid climate
    sim = SimpleDewSimulator(
        T_day=303,    # 30°C day
        T_night=290,  # 17°C night
        RH=0.35       # 35% humidity
    )
    
    # Run and plot
    fig = sim.plot_comparison(days=7)
    plt.savefig('dew_simulation.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Graph saved to: dew_simulation.png")


if __name__ == '__main__':
    main()
