#!/usr/bin/env python3
"""
Optimize 40-bit seed for local climate conditions.
Uses scipy differential evolution algorithm.
"""

import numpy as np
from scipy.optimize import differential_evolution


CLIMATES = {
    'arid': {'T_day': 308, 'T_night': 288, 'RH': 0.25, 'wind': 2.0},
    'semi_arid': {'T_day': 303, 'T_night': 290, 'RH': 0.35, 'wind': 2.5},
    'mediterranean': {'T_day': 298, 'T_night': 292, 'RH': 0.45, 'wind': 3.0},
    'tropical_dry': {'T_day': 305, 'T_night': 295, 'RH': 0.40, 'wind': 1.5},
}


class SeedOptimizer:
    """Find optimal 40-bit seed for a given climate."""

    def __init__(self, climate='arid'):
        self.climate_name = climate
        self.params = CLIMATES[climate]

    def decode_seed(self, seed_bytes):
        """Convert 5 bytes to control parameters."""
        return {
            'amp_T': 1.0 + (seed_bytes[0] / 255.0) * 4.0,
            'amp_pH': 1.0 + (seed_bytes[1] / 255.0) * 4.0,
            'amp_light': 1.0 + (seed_bytes[2] / 255.0) * 4.0,
            'wavelength': 500 + (seed_bytes[3] / 255.0) * 4500,
            'crop_bias': seed_bytes[4] / 255.0,
        }

    def evaluate_seed(self, seed_bytes):
        """
        Score a seed (higher is better).
        Returns negative for minimization.
        """
        params = self.decode_seed(seed_bytes)

        delta_T = self.params['T_day'] - self.params['T_night']

        # Precipitation estimate (mm/day)
        precip = (0.3 * params['amp_T'] +
                  0.4 * params['amp_pH'] +
                  0.3 * params['amp_light']) * self.params['RH'] * delta_T * 0.01

        # Energy consumption (kWh/day)
        energy = 0.5 * (params['amp_T'] + params['amp_pH'] + params['amp_light'])

        # Safety penalty for high amplification
        max_amp = max(params['amp_T'], params['amp_pH'], params['amp_light'])
        safety = 1.0 - (max_amp / 10.0)

        score = precip * 1.0 - energy * 0.1 + safety * 0.5
        return -score

    def optimize(self):
        """Find optimal seed using differential evolution."""
        print(f"Optimizing for {self.climate_name} climate...")
        print(f"  T_day:  {self.params['T_day'] - 273:.1f} C")
        print(f"  T_night: {self.params['T_night'] - 273:.1f} C")
        print(f"  RH:     {self.params['RH'] * 100:.0f}%")
        print()

        bounds = [(0, 255)] * 5

        result = differential_evolution(
            self.evaluate_seed,
            bounds,
            maxiter=50,
            popsize=15,
            disp=False,
            workers=1,
        )

        optimal_bytes = [int(round(x)) for x in result.x]
        params = self.decode_seed(optimal_bytes)

        return {
            'seed': optimal_bytes,
            'params': params,
            'score': -result.fun,
        }


def main():
    """Optimize seeds for all climates."""
    print("=" * 60)
    print("Seed Optimization for Different Climates")
    print("=" * 60)
    print()

    for climate_name in CLIMATES:
        optimizer = SeedOptimizer(climate_name)
        result = optimizer.optimize()

        print(f"  Optimal seed: {result['seed']}")
        print(f"  Score:        {result['score']:.4f}")
        print(f"  Params:       amp_T={result['params']['amp_T']:.2f}, "
              f"amp_pH={result['params']['amp_pH']:.2f}, "
              f"amp_light={result['params']['amp_light']:.2f}")
        print(f"  Wavelength:   {result['params']['wavelength']:.0f} nm")
        print(f"  Crop bias:    {result['params']['crop_bias']:.2f}")
        print()


if __name__ == '__main__':
    main()
