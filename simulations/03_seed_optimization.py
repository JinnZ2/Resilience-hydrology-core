#!/usr/bin/env python3
"""
Optimize 40-bit seed for local climate conditions.
Uses scipy differential evolution algorithm.

WARNING: with the objective as written, the optimum is degenerate -- see
docs/method-log.md (M-03). The energy and safety penalties outweigh the
precipitation term at every amplification level, so the optimiser always
returns minimum amplification (system effectively off), and seed bytes 3-4
do not enter the objective at all. The diagnostics below report this rather
than hiding it. Do not quote these seeds as tuned configurations.
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

    def score_terms(self, seed_bytes):
        """Break the objective into its terms, for diagnostics."""
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

        return {
            'precip': precip * 1.0,
            'energy': -energy * 0.1,
            'safety': safety * 0.5,
        }

    def evaluate_seed(self, seed_bytes):
        """
        Score a seed (higher is better).
        Returns negative for minimization.
        """
        return -sum(self.score_terms(seed_bytes).values())

    def byte_sensitivity(self, reference=None):
        """
        How much does each seed byte move the score?

        A byte with zero range is unconstrained: the optimiser will return an
        arbitrary value for it, and that value carries no information.
        """
        reference = list(reference or [128] * 5)
        ranges = []
        for i in range(5):
            scores = []
            for value in (0, 64, 128, 192, 255):
                probe = list(reference)
                probe[i] = value
                scores.append(-self.evaluate_seed(probe))
            ranges.append(max(scores) - min(scores))
        return ranges

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
        sensitivity = self.byte_sensitivity(optimal_bytes)

        return {
            'seed': optimal_bytes,
            'params': params,
            'score': -result.fun,
            'terms': self.score_terms(optimal_bytes),
            'sensitivity': sensitivity,
            'dead_bytes': [i for i, r in enumerate(sensitivity) if r < 1e-9],
            'pinned_bytes': [i for i, b in enumerate(optimal_bytes) if b in (0, 255)],
        }


BYTE_NAMES = ['amp_T', 'amp_pH', 'amp_light', 'wavelength', 'crop_bias']


def main():
    """Optimize seeds for all climates."""
    print("=" * 60)
    print("Seed Optimization for Different Climates")
    print("=" * 60)
    print()

    any_degenerate = False

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

        terms = result['terms']
        print(f"  Score terms:  precip={terms['precip']:+.4f} "
              f"energy={terms['energy']:+.4f} safety={terms['safety']:+.4f}")
        print(f"  Byte sensitivity: " + ", ".join(
            f"{BYTE_NAMES[i]}={r:.4f}" for i, r in enumerate(result['sensitivity'])))

        if result['dead_bytes']:
            any_degenerate = True
            names = ", ".join(BYTE_NAMES[i] for i in result['dead_bytes'])
            print(f"  [!] UNCONSTRAINED bytes (no effect on score): {names}")
            print(f"      Their values above are arbitrary. Do not report them.")
        if result['pinned_bytes']:
            any_degenerate = True
            names = ", ".join(BYTE_NAMES[i] for i in result['pinned_bytes'])
            print(f"  [!] Optimum sits ON A BOUND for: {names}")
            print(f"      The objective wants to leave the search space; the")
            print(f"      'optimum' is the edge of the box, not a real peak.")
        print()

    if any_degenerate:
        print("-" * 60)
        print("RESULT: degenerate. This objective does not select a seed.")
        print("See docs/method-log.md (M-03) for the falsification record.")
        print("-" * 60)


if __name__ == '__main__':
    main()
