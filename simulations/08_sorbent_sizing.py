#!/usr/bin/env python3
"""
Size a sorbent water harvester for a target daily yield.

WHY THIS FILE EXISTS

H12 found that passive dew is a wall, not a slope: below roughly 70% pre-dawn RH
at 15 C — or 90% at 32 C — the yield is exactly zero. H13 found that sorption
clears that wall, harvesting down to 11-20% RH, because a sorbent does not need
the air to reach saturation.

That gives the project two builds rather than one, and they serve different
sites. This file sizes the second: how much salt, how much tray area, how much
solar aperture, and roughly what it costs, for a target litres per day.

THE CYCLE

    night   bed open to air, salt adsorbs water vapour     (free)
    day     bed sealed in a glazed box, sun drives vapour  (solar heat)
            off; vapour condenses on a cooler surface and
            runs to a collector

No pump, no compressor, no electricity. The energy is sunlight, as heat, which
is what the mechanism wants and what drought regions have.

THE SALT CHOICE IS THE DESIGN DECISION

A hygroscopic salt only takes up water in bulk above its deliquescence point:

    CaCl2   DRH ~29-30% at 25 C   cheap, food-grade, safe
    LiCl    DRH ~11% at 25 C      works in far drier air, costs more,
                                  and lithium is pharmacologically active

Below its DRH a salt still adsorbs, but far less, and the yield falls off a
cliff much like dew does. **So the site's pre-dawn RH picks the salt, and the
salt decides whether the build is a drinking-water build.** That is the single
most important thing this script is for.

STATUS

Material properties are taken from published work (docs/build-sorbent.md cites
them). The cycle model is this project's own and is [ASSUMED] wherever it is not
a physical constant. **Nobody in this project has built one of these.** Treat the
output as a starting bill of materials to iterate from, not a specification.

Usage:
    python 08_sorbent_sizing.py
    python 08_sorbent_sizing.py --target-l 2.0 --rh 0.35
    python 08_sorbent_sizing.py --salt licl --rh 0.15
    python 08_sorbent_sizing.py --compare-salts
"""

import argparse
import math

# Physical constants  [STANDARD]
L_V = 2.45e6              # J/kg, latent heat of vaporisation
KWH_PER_J = 1.0 / 3.6e6
WATER_DENSITY = 1.0       # kg/L

# Sorbent properties. Uptake ranges and DRH are from published measurements;
# the curve shape between them is [ASSUMED].
SALTS = {
    'cacl2': {
        'name': 'Calcium chloride (CaCl2)',
        'drh': 0.30,              # deliquescence RH at ~25 C
        'uptake_max': 2.44,       # g water / g dry composite, high-RH limit
        'uptake_ref': (0.80, 1.50),   # measured: ~1.5 g/g at 80% RH
        'binding_mj_per_kg': 0.35,    # above latent heat  [ASSUMED]
        'isotherm_n': 1.0,            # deliquescent: sharp bulk uptake above DRH
        'cost_per_kg': 2.0,
        'potable': True,
        'note': 'FDA GRAS food additive. The default choice wherever the site '
                'allows it.',
    },
    'licl': {
        'name': 'Lithium chloride (LiCl)',
        'drh': 0.11,
        'uptake_max': 3.50,
        'uptake_ref': (0.80, 2.00),
        'isotherm_n': 1.0,
        'binding_mj_per_kg': 0.45,
        'cost_per_kg': 45.0,
        'potable': False,
        'note': 'Works far into dry air, and lithium is pharmacologically '
                'active. Do not use for drinking water without a verified '
                'no-carryover design and testing.',
    },
    'silica_gel': {
        'name': 'Silica gel (no salt)',
        'drh': 0.0,               # physisorbent, no deliquescence step
        'uptake_max': 0.40,
        'uptake_ref': (0.80, 0.35),
        # Silica gel's isotherm is S-shaped (Type IV/V), not Langmuir: uptake
        # stays low until mid humidity, then rises. An earlier version used a
        # Langmuir form here and put silica gel at 0.23 g/g at 15% RH, roughly
        # 5x the real value, which made it look like a viable dry-air option.
        # It is not one.  [ASSUMED shape, anchored to the reference point]
        'isotherm_n': 2.5,
        'binding_mj_per_kg': 0.30,
        'cost_per_kg': 5.0,
        'potable': True,
        'note': 'Low capacity but no brine, no creep, no carryover risk. The '
                'safe starting point for a first build.',
    },
}

# Cycle losses: incomplete regeneration, condenser inefficiency, leakage.  [ASSUMED]
CYCLE_EFFICIENCY = 0.65

# Solar collector: glazed box with a black absorber.  [ASSUMED, conventional]
SOLAR_THERMAL_EFF = 0.35          # a simple glazed hot box beats a PV+heater path
INSOLATION = {'desert': 6.5, 'sunny': 5.0, 'moderate': 4.0, 'poor': 2.5}

# Bed loading: how much composite a tray carries per m^2 of exposed surface.
# Deeper beds hold more salt but adsorb more slowly — the air only reaches the
# top few millimetres overnight.  [ASSUMED]
BED_KG_PER_M2 = 8.0

COSTS = {'tray_per_m2': 6.0, 'glazing_per_m2': 12.0, 'substrate_per_kg': 0.5}


def uptake(salt, rh):
    """
    Water uptake, g per g of dry composite, at a given pre-dawn RH.

    Langmuir-like rise with humidity, anchored to the measured reference point,
    with a sharp drop below the deliquescence RH — above DRH the salt forms
    brine and takes up water in bulk; below it, only surface adsorption remains.
    The cliff is the physically important feature.  [ASSUMED shape]
    """
    props = SALTS[salt]
    rh_ref, u_ref = props['uptake_ref']
    u_max = props['uptake_max']
    n = props.get('isotherm_n', 1.0)
    # Solve the constant that reproduces the measured reference point, for
    # whichever isotherm shape this sorbent has.
    k = (rh_ref ** n * (u_max / u_ref - 1.0)) ** (1.0 / n)
    u = u_max * rh ** n / (rh ** n + k ** n)
    drh = props['drh']
    if rh < drh:
        # Below deliquescence: a residual fraction, falling toward zero.
        residual = 0.25 * (rh / drh) ** 2
        u *= residual
    return max(u, 0.0)


def size_build(salt, rh, target_l, insolation='sunny'):
    """Bill of materials for a target daily yield."""
    props = SALTS[salt]
    u = uptake(salt, rh)
    water_per_kg = u * CYCLE_EFFICIENCY          # kg water per kg composite/cycle
    if water_per_kg <= 1e-4:
        return {'feasible': False, 'uptake': u, 'salt': salt, 'rh': rh}

    target_kg = target_l * WATER_DENSITY
    sorbent_kg = target_kg / water_per_kg
    bed_area = sorbent_kg / BED_KG_PER_M2

    energy_mj = target_kg * (L_V / 1e6 + props['binding_mj_per_kg'])
    energy_kwh = energy_mj * 1e6 * KWH_PER_J
    sun = INSOLATION[insolation]
    aperture = energy_kwh / (sun * SOLAR_THERMAL_EFF)

    salt_fraction = 0.35          # composite is ~35% salt by dry mass  [ASSUMED]
    salt_kg = sorbent_kg * salt_fraction
    substrate_kg = sorbent_kg - salt_kg
    cost = (salt_kg * props['cost_per_kg']
            + substrate_kg * COSTS['substrate_per_kg']
            + bed_area * COSTS['tray_per_m2']
            + aperture * COSTS['glazing_per_m2'])

    return {
        'feasible': True, 'salt': salt, 'rh': rh, 'uptake': u,
        'water_per_kg': water_per_kg, 'sorbent_kg': sorbent_kg,
        'salt_kg': salt_kg, 'bed_area': bed_area,
        'energy_kwh': energy_kwh, 'aperture': aperture,
        'cost': cost, 'target_l': target_l,
        'potable': props['potable'],
    }


def report(b, insolation):
    props = SALTS[b['salt']]
    print(f"\n{props['name']} at {b['rh'] * 100:.0f}% pre-dawn RH")
    print(f"  deliquescence RH: {props['drh'] * 100:.0f}%  "
          f"({'ABOVE — bulk uptake' if b['rh'] >= props['drh'] else 'BELOW — uptake collapses'})")

    if not b['feasible']:
        print(f"  uptake {b['uptake']:.3f} g/g — too low to build around.")
        print(f"  This salt does not work at this humidity. Try a lower-DRH")
        print(f"  salt (--compare-salts) or accept that the site is too dry.")
        return

    print(f"  uptake {b['uptake']:.2f} g/g, {b['water_per_kg']:.2f} L of water "
          f"per kg of composite per cycle")
    print(f"\n  For {b['target_l']:.1f} L/day:")
    print(f"    composite         {b['sorbent_kg']:>7.1f} kg "
          f"({b['salt_kg']:.1f} kg salt + substrate)")
    print(f"    bed area          {b['bed_area']:>7.2f} m2  "
          f"(at {BED_KG_PER_M2:.0f} kg/m2)")
    print(f"    regeneration heat {b['energy_kwh']:>7.2f} kWh/day")
    print(f"    solar aperture    {b['aperture']:>7.2f} m2  "
          f"(glazed box, {SOLAR_THERMAL_EFF:.0%}, {insolation} sun)")
    print(f"    rough cost        {b['cost']:>7.0f} USD")
    if not b['potable']:
        print(f"\n    ! NOT a drinking-water build without further work — "
              f"see docs/build-sorbent.md")


def main():
    parser = argparse.ArgumentParser(description='Size a sorbent water harvester.')
    parser.add_argument('--salt', default='cacl2', choices=list(SALTS))
    parser.add_argument('--rh', type=float, default=0.35,
                        help='pre-dawn relative humidity at the site')
    parser.add_argument('--target-l', type=float, default=1.0)
    parser.add_argument('--insolation', default='sunny', choices=list(INSOLATION))
    parser.add_argument('--compare-salts', action='store_true')
    parser.add_argument('--sweep', action='store_true')
    args = parser.parse_args()

    print("=" * 72)
    print("Sorbent Harvester Sizing")
    print("=" * 72)
    print("Sizes the dry-air build. Nobody in this project has built one —")
    print("this is a starting bill of materials, not a specification.")

    if args.compare_salts:
        print("\n" + "-" * 72)
        print(f"SALT CHOICE at {args.rh * 100:.0f}% pre-dawn RH, "
              f"for {args.target_l:.1f} L/day")
        print("-" * 72)
        print(f"{'salt':<24}{'DRH':>6}{'uptake':>9}{'composite':>11}"
              f"{'aperture':>10}{'cost':>8}  potable")
        for key in SALTS:
            b = size_build(key, args.rh, args.target_l, args.insolation)
            props = SALTS[key]
            if not b['feasible']:
                print(f"{props['name'][:24]:<24}{props['drh']:>5.0%}"
                      f"{b['uptake']:>9.2f}{'—':>11}{'—':>10}{'—':>8}"
                      f"  {'yes' if props['potable'] else 'NO'}")
            else:
                print(f"{props['name'][:24]:<24}{props['drh']:>5.0%}"
                      f"{b['uptake']:>9.2f}{b['sorbent_kg']:>10.1f}kg"
                      f"{b['aperture']:>9.2f}m2{b['cost']:>7.0f}$"
                      f"  {'yes' if props['potable'] else 'NO'}")
        print("\n  The site's humidity picks the salt, and the salt decides")
        print("  whether it is a drinking-water build. Above ~30% RH use CaCl2:")
        print("  cheap, food-grade, no lithium question. Below it, CaCl2's")
        print("  uptake collapses and only LiCl or a MOF still works.")
        return

    if args.sweep:
        print("\n" + "-" * 72)
        print(f"HUMIDITY SWEEP — {SALTS[args.salt]['name']}, "
              f"{args.target_l:.1f} L/day")
        print("-" * 72)
        print(f"{'RH':>6}{'uptake g/g':>13}{'composite kg':>15}"
              f"{'bed m2':>9}{'aperture m2':>13}")
        for rh in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80):
            b = size_build(args.salt, rh, args.target_l, args.insolation)
            if not b['feasible']:
                print(f"{rh:>6.0%}{b['uptake']:>13.3f}{'not viable':>15}"
                      f"{'—':>9}{'—':>13}")
            else:
                print(f"{rh:>6.0%}{b['uptake']:>13.2f}{b['sorbent_kg']:>15.1f}"
                      f"{b['bed_area']:>9.2f}{b['aperture']:>13.2f}")
        print("\n  Note what happens below the deliquescence RH: the mass of")
        print("  salt needed climbs steeply, then the build stops being")
        print("  sensible. It is the same kind of wall dew has, just further")
        print("  into dry air — which is exactly why this build exists.")
        return

    b = size_build(args.salt, args.rh, args.target_l, args.insolation)
    report(b, args.insolation)

    print("\n" + "-" * 72)
    print("BEFORE YOU BUILD")
    print("-" * 72)
    print("  1. Measure your site's PRE-DAWN humidity first. Every number here")
    print("     depends on it, and daytime RH is not a substitute.")
    print("  2. Start with silica gel. Lower yield, but no brine, no creep and")
    print("     no carryover risk while you learn the cycle.")
    print("  3. Never drink water that tastes salty. Salt carryover is the")
    print("     failure mode of this design — see docs/build-sorbent.md.")
    print()


if __name__ == '__main__':
    main()
