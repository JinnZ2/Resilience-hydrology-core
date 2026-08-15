#!/usr/bin/env python3
"""
Basic Dew Simulation

Simulates atmospheric water collection using natural gradients.
Shows system ON vs OFF comparison over configurable number of days.

Usage:
    python 01_basic_dew.py
    python 01_basic_dew.py --climate arid --days 14
"""

import numpy as np
import matplotlib.pyplot as plt


# Climate presets: temperature in Kelvin, relative humidity 0-1
CLIMATES = {
    'arid': {'T_day': 308, 'T_night': 288, 'RH': 0.25},
    'semi_arid': {'T_day': 303, 'T_night': 290, 'RH': 0.35},
    'mediterranean': {'T_day': 298, 'T_night': 292, 'RH': 0.45},
    'tropical_dry': {'T_day': 305, 'T_night': 295, 'RH': 0.40},
}


class DewSimulator:
    """
    Dew formation simulator.

    Physics:
    - Temperature inversion at night drives condensation
    - Energy input: zero (natural mode) or <1W (boosted)

    ASSUMPTION, not a result: the system amplifies the natural process by a
    hard-coded factor of 3.0 (see simulate_night). That factor is an input, so
    the ON vs OFF comparison illustrates the assumption rather than testing it,
    and every climate reports exactly +200%. Nothing in this repository derives
    or measures it. See docs/research-log.md, H2 and O3.
    """

    def __init__(self, T_day=305, T_night=288, RH=0.30):
        self.T_day = T_day
        self.T_night = T_night
        self.RH = RH
        self.dew_point_offset = 5.0  # K below air temp
        self.collection_efficiency = 0.7

    def simulate_night(self, system_on=False):
        """
        Simulate one night of dew formation.

        Returns:
            water_mm: millimeters of water collected per m^2
        """
        delta_T = self.T_day - self.T_night
        natural_dew = self.RH * delta_T * 0.02  # mm/night

        # Assumed, not derived — see class docstring and research log H2.
        amplification = 3.0 if system_on else 1.0
        return natural_dew * amplification

    def run(self, days=7, system_on=True):
        """Run multi-day simulation."""
        daily_water = [self.simulate_night(system_on) for _ in range(days)]

        return {
            'daily_mm': daily_water,
            'total_mm': sum(daily_water),
            'avg_mm_day': sum(daily_water) / days,
        }

    def plot_comparison(self, days=7):
        """Plot system ON vs OFF and return the figure."""
        results_off = self.run(days, system_on=False)
        results_on = self.run(days, system_on=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        x = range(1, days + 1)
        ax1.bar(x, results_off['daily_mm'], alpha=0.6, label='System OFF', color='gray')
        ax1.bar(x, results_on['daily_mm'], alpha=0.8, label='System ON', color='blue')
        ax1.set_xlabel('Day')
        ax1.set_ylabel('Water (mm/day)')
        ax1.set_title('Daily Water Production')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

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

        print(f"\n{days}-Day Results:")
        print(f"  System OFF: {results_off['total_mm']:.3f} mm total ({results_off['avg_mm_day']:.4f} mm/day)")
        print(f"  System ON:  {results_on['total_mm']:.3f} mm total ({results_on['avg_mm_day']:.4f} mm/day)")
        improvement = (results_on['total_mm'] / results_off['total_mm'] - 1) * 100
        print(f"  Improvement: +{improvement:.0f}%\n")

        return fig


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Simulate dew collection')
    parser.add_argument('--climate', default='semi_arid',
                        choices=list(CLIMATES.keys()))
    parser.add_argument('--days', type=int, default=7)
    parser.add_argument('--output', default='dew_simulation.png')
    args = parser.parse_args()

    params = CLIMATES[args.climate]

    print("=" * 50)
    print("Basic Dew Collection Simulation")
    print("=" * 50)
    print(f"Climate: {args.climate}")
    print(f"  Day temp:  {params['T_day'] - 273:.1f} C")
    print(f"  Night temp: {params['T_night'] - 273:.1f} C")
    print(f"  Humidity:  {params['RH'] * 100:.0f}%")

    sim = DewSimulator(**params)
    fig = sim.plot_comparison(days=args.days)
    fig.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Graph saved to: {args.output}")


if __name__ == '__main__':
    main()
