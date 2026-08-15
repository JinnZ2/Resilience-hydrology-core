#!/usr/bin/env python3
"""
Search for a 40-bit seed suited to local climate conditions.
Uses scipy differential evolution algorithm.

KNOWN DEGENERATE — do not deploy the seeds this prints.

The objective in evaluate_seed() decreases monotonically in amplification: the
precipitation gain per unit is at most 0.020 under any climate preset here,
against an energy penalty of 0.05 and a further 0.05 safety penalty on the
largest amplification. So the optimum is always the corner at minimum
amplification, and every climate returns [0, 0, 0, ...] — "optimal" meaning the
system turned all the way down.

Bytes 3 and 4 (wavelength, crop_bias) are decoded but never read by
evaluate_seed, so they are unconstrained: repeated runs return different values
for them at an identical score.

This file is kept as a search scaffold. The weights were deliberately NOT
retuned to produce an interior optimum — that would manufacture the desired
answer. Fixing it needs a defensible cost model expressing amplification gain
and energy cost in comparable units, plus a decision on whether bytes 3-4 get a
role or leave the seed format. See docs/research-log.md, H5 and O6.
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
    print("WARNING: this objective is known to be degenerate. It always")
    print("prefers minimum amplification, so every climate returns a corner")
    print("solution, and seed bytes 3-4 are unused and therefore random.")
    print("Do not deploy these seeds. See docs/research-log.md, H5.")
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
