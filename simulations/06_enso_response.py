#!/usr/bin/env python3
"""
ENSO response: what a strong El Nino does to modelled dew yield.

WHY THIS FILE EXISTS

Round 3 found that *when* you run a collector dominates everything you bolt onto
it. ENSO is the largest interannual control on "when" — it shifts humidity,
cloud cover and temperature together, on a schedule that is forecast months
ahead. A drought-resilience project that ignores it is choosing to be surprised.

As of August 2026 a very strong El Nino is forecast to peak between October 2026
and January 2027 (see docs/enso-context.md for sources and their limits). The
regions where it raises drought risk — Australia, South-East Asia, southern
Africa, Central America, the northern Amazon — are exactly the regions this
project claims to serve.

THE QUESTION THIS ANSWERS

El Nino drought does two opposite things to radiative dew:

    drier air     -> less vapour to condense        -> LESS dew
    clearer skies -> stronger radiative cooling     -> MORE dew

Both are real and they fight. The model settles which wins, because it contains
both channels explicitly. That is the whole reason for having built an energy
balance rather than a scaling relation.

WHAT THIS IS NOT

There is published work on ENSO and *fog* — El Nino intensifies fog in the Namib
(Li et al. 2025) and raises fog-water yield in the Atacama. That work is about
advection fog: marine air driven onshore over cold water, where sea-surface
temperature sets the outcome. This model is about radiative dew: a surface
cooling below the dew point of the air already above it. Different mechanism,
different driver, different sign in principle.

Carrying the fog result across to dew would be M-01 happening again — a number
detached from the model that produced it. So those studies are cited as context
in docs/enso-context.md and deliberately NOT used as evidence here.

Usage:
    python 06_enso_response.py
    python 06_enso_response.py --region australia --samples 3000
    python 06_enso_response.py --list-regions
    python 06_enso_response.py --decompose
"""

import argparse
import importlib.util
import os

import numpy as np
import matplotlib.pyplot as plt


def _load(name):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    spec = importlib.util.spec_from_file_location(name.replace('.py', ''), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vs = _load('04_variable_search.py')
DewEnergyBalance = vs.DewEnergyBalance


# ---------------------------------------------------------------------------
# ENSO perturbations
# ---------------------------------------------------------------------------

# How a strong El Nino shifts pre-dawn conditions, per region class.
#
# ALL OF THESE NUMBERS ARE [ASSUMED]. The SIGNS come from the documented
# teleconnection pattern (El Nino raises drought risk in the first group and
# wets the second); the MAGNITUDES are this project's guesses, because per-region
# pre-dawn RH and cloud anomalies were not obtainable from the sources available
# here (docs/enso-context.md records that limitation).
#
# So: read the DIRECTION and the RANKING of the channels. Do not quote the
# absolute mL. The decomposition below is the part that does not depend on
# getting the magnitudes right.
REGIONS = {
    'australia': {
        'label': 'Australia — sharpest El Nino drought signal in the S. hemisphere',
        'enso_drought': True,
        'd_rh': -0.10, 'd_cloud': -0.15, 'd_t_air': +1.5,
    },
    'southern_africa': {
        'label': 'Southern Africa — El Nino drought, maize belt',
        'enso_drought': True,
        'd_rh': -0.08, 'd_cloud': -0.12, 'd_t_air': +1.5,
    },
    'se_asia': {
        'label': 'South-East Asia / Indonesia — El Nino drought',
        'enso_drought': True,
        'd_rh': -0.07, 'd_cloud': -0.15, 'd_t_air': +1.0,
    },
    'central_america': {
        'label': 'Central America — El Nino drought corridor',
        'enso_drought': True,
        'd_rh': -0.08, 'd_cloud': -0.10, 'd_t_air': +1.0,
    },
    'southern_us': {
        'label': 'Southern US — El Nino typically WETTER',
        'enso_drought': False,
        'd_rh': +0.06, 'd_cloud': +0.15, 'd_t_air': -0.5,
    },
}

# Baseline pre-dawn conditions, neutral ENSO. Semi-arid drought-prone defaults,
# consistent with 04_variable_search.py's semi_arid preset.  [ASSUMED]
BASELINE_WINDOW = {
    't_air_night': (287.0, 296.0),
    'rh': (0.45, 0.90),
    'wind_speed': (0.0, 5.0),
    'cloud_cover': (0.05, 0.55),
    'night_hours': (10.0, 13.0),
}

# A collector built to the current guidance: season, siting, tilt, insulation,
# no Peltier (docs/build-dew.md).
CONFIG = {
    'sky_view_factor': 0.95,
    'local_vapor_boost': 0.0,
    'surface_emissivity': 0.95,
    'tilt_deg': 30.0,
    'insulation_r': 1.0,
    'electrical_w_m2': 0.0,
    'cop_cooling': 0.5,
}

AREA_M2 = 1.0        # report per square metre; scale linearly for real builds
WEATHER_KEYS = ('t_air_night', 'rh', 'wind_speed', 'cloud_cover', 'night_hours')


def shift_window(window, region, channels=('rh', 'cloud', 't_air')):
    """Apply an ENSO perturbation to a weather window, channel by channel."""
    out = dict(window)
    if 'rh' in channels:
        lo, hi = window['rh']
        out['rh'] = (max(0.05, lo + region['d_rh']), min(1.0, hi + region['d_rh']))
    if 'cloud' in channels:
        lo, hi = window['cloud_cover']
        out['cloud_cover'] = (max(0.0, lo + region['d_cloud']),
                              min(1.0, hi + region['d_cloud']))
    if 't_air' in channels:
        lo, hi = window['t_air_night']
        out['t_air_night'] = (lo + region['d_t_air'], hi + region['d_t_air'])
    return out


class EnsoComparison:
    """Paired-quantile comparison of dew yield across ENSO states."""

    def __init__(self, samples=2000, rng_seed=0):
        rng = np.random.default_rng(rng_seed)
        self.n = samples
        # Same quantiles for every scenario: the same night, under two climates.
        self.u = rng.random((samples, len(WEATHER_KEYS)))
        self.coeffs = []
        for i in range(samples):
            self.coeffs.append({k: lo + rng.random() * (hi - lo)
                                for k, (lo, hi) in vs_uncertainty().items()})

    def evaluate(self, window):
        out = np.zeros(self.n)
        for i in range(self.n):
            v = dict(CONFIG)
            for j, key in enumerate(WEATHER_KEYS):
                lo, hi = window[key]
                v[key] = lo + self.u[i, j] * (hi - lo)
            result = DewEnergyBalance(coeffs=self.coeffs[i], **v).solve()
            out[i] = 0.0 if result['frozen'] else result['yield_mm'] * AREA_M2 * 1000
        return out


def vs_uncertainty():
    """Coefficient uncertainty, shared with 05_transition_paths.py (O10)."""
    return {
        'h_c_still': (1.5, 4.0),
        'h_c_wind': (2.0, 4.5),
        'eff_max': (0.80, 1.00),
        'tilt_char': (10.0, 30.0),
    }


def summarise(base, cand):
    delta = cand - base
    active = np.maximum(base, cand) > 1e-9
    n = int(active.sum())
    return {
        'mean_base': float(base.mean()),
        'mean_cand': float(cand.mean()),
        'delta': float(delta.mean()),
        'pct': float(100 * delta.mean() / base.mean()) if base.mean() > 1e-9 else float('nan'),
        'p_worse': float((delta[active] < 0).mean()) if n else 0.0,
        'productive_base': float((base > 0).mean()),
        'productive_cand': float((cand > 0).mean()),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Dew yield under a strong El Nino, by region.')
    parser.add_argument('--region', default='australia', choices=list(REGIONS))
    parser.add_argument('--samples', type=int, default=2000)
    parser.add_argument('--decompose', action='store_true',
                        help='separate the drying and clearing channels')
    parser.add_argument('--all-regions', action='store_true')
    parser.add_argument('--list-regions', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', default='enso_response.png')
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()

    if args.list_regions:
        print("=" * 72)
        print("ENSO region classes  (all perturbation magnitudes are [ASSUMED])")
        print("=" * 72)
        for key, r in REGIONS.items():
            print(f"\n  {key}")
            print(f"      {r['label']}")
            print(f"      El Nino drought risk: {'yes' if r['enso_drought'] else 'no'}")
            print(f"      d_RH {r['d_rh']:+.2f}   d_cloud {r['d_cloud']:+.2f}"
                  f"   d_T_air {r['d_t_air']:+.1f} K")
        return

    comp = EnsoComparison(samples=args.samples, rng_seed=args.seed)
    neutral = comp.evaluate(BASELINE_WINDOW)

    print("=" * 72)
    print("ENSO Response — modelled dew yield under a strong El Nino")
    print("=" * 72)
    print(f"Samples   : {args.samples} paired draws (same night, two climates)")
    print(f"Collector : built to current guidance, 1 m^2, passive")
    print(f"Neutral   : {neutral.mean():.1f} mL/night mean, "
          f"{100 * (neutral > 0).mean():.0f}% of nights productive")
    print()
    print("Magnitudes below are [ASSUMED] — read the sign and the ranking, not")
    print("the absolute mL. See docs/enso-context.md for what is sourced and")
    print("what is not.")

    regions = list(REGIONS) if args.all_regions else [args.region]

    print("\n" + "-" * 72)
    print("1. YIELD UNDER EL NINO")
    print("-" * 72)
    print(f"{'region':<18}{'neutral':>9}{'el nino':>9}{'change':>9}"
          f"{'':>3}{'productive nights':>20}")
    rows = {}
    for key in regions:
        r = REGIONS[key]
        cand = comp.evaluate(shift_window(BASELINE_WINDOW, r))
        s = summarise(neutral, cand)
        rows[key] = s
        print(f"{key:<18}{s['mean_base']:>9.1f}{s['mean_cand']:>9.1f}"
              f"{s['pct']:>+8.0f}%   {s['productive_base']:>8.0%} -> "
              f"{s['productive_cand']:.0%}")

    if args.decompose:
        print("\n" + "-" * 72)
        print("2. WHICH CHANNEL WINS — drying vs clearing")
        print("-" * 72)
        print("El Nino drought dries the air AND clears the sky. The first")
        print("removes vapour to condense; the second strengthens the radiative")
        print("cooling that drives condensation. Run separately:\n")
        print(f"{'region':<17}{'drying':>9}{'clearing':>10}{'warming':>9}"
              f"{'sum':>8}{'combined':>10}{'':>2}dominant")
        for key in regions:
            r = REGIONS[key]
            d_dry = comp.evaluate(shift_window(BASELINE_WINDOW, r, ('rh',))).mean() - neutral.mean()
            d_clear = comp.evaluate(shift_window(BASELINE_WINDOW, r, ('cloud',))).mean() - neutral.mean()
            d_warm = comp.evaluate(shift_window(BASELINE_WINDOW, r, ('t_air',))).mean() - neutral.mean()
            d_both = comp.evaluate(shift_window(BASELINE_WINDOW, r)).mean() - neutral.mean()
            parts = {'drying': d_dry, 'clearing': d_clear, 'warming': d_warm}
            dominant = max(parts, key=lambda k: abs(parts[k]))
            print(f"{key:<17}{d_dry:>+9.1f}{d_clear:>+10.1f}{d_warm:>+9.1f}"
                  f"{d_dry + d_clear + d_warm:>+8.1f}{d_both:>+10.1f}"
                  f"  {dominant}")
        print("\nThree channels move at once, so 'combined' is shown against the")
        print("naive 'sum'. Where they differ the channels interact: the")
        print("condensation term is nonlinear, and a night that no longer")
        print("reaches the dew point cannot be improved by a clearer sky.")
        print("\nIf drying dominates in the drought regions, the system produces")
        print("least exactly when and where it is most needed. That is a")
        print("property of the physics, not a fixable design flaw.")

    print("\n" + "-" * 72)
    print("CAVEAT")
    print("-" * 72)
    print("  Perturbation magnitudes are invented; only their signs are sourced.")
    print("  The model has never been validated against field data. This says")
    print("  what the model implies about El Nino, not what El Nino will do.")
    print("  Published ENSO/fog results (Namib, Atacama) describe advection fog,")
    print("  a different mechanism, and are deliberately not used as evidence")
    print("  here. See docs/enso-context.md.")
    print()

    if not args.no_plot and rows:
        fig, ax = plt.subplots(figsize=(9, 5))
        keys = list(rows)
        ax.bar(range(len(keys)), [rows[k]['pct'] for k in keys],
               color=['#c62828' if rows[k]['pct'] < 0 else '#2e7d32' for k in keys],
               alpha=0.85)
        ax.axhline(0, color='#333333', linewidth=1)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(keys, rotation=20, ha='right', fontsize=9)
        ax.set_ylabel('change in mean dew yield under strong El Nino (%)')
        ax.set_title('Modelled dew response to a strong El Nino\n'
                     'assumed magnitudes, unvalidated model', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        fig.tight_layout()
        fig.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"Graph saved to: {args.output}")
        plt.close(fig)


if __name__ == '__main__':
    main()
