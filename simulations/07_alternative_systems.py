#!/usr/bin/env python3
"""
Alternative water-harvesting mechanisms for severe, dry-air drought.

WHY THIS FILE EXISTS

H11 (docs/research-log.md, Round 5) falsified this project's founding premise:
radiative dew needs humid air, El Nino droughts are dry-air droughts, and
modelled yield falls 68-85% in exactly the regions most at risk. Supply and need
move in opposite directions.

That is not a reason to stop. It is a reason to ask what mechanism DOES work
when the air is dry, and to compare candidates on the same physical basis
instead of by enthusiasm.

THE STRUCTURAL DIFFERENCE

Passive dew is free but GATED. It requires the surface to reach the dew point,
and radiative cooling delivers only 3-9 K of depression. Below roughly 60-70% RH
the dew-point depression exceeds that and the yield is not small — it is exactly
zero. No tilt, no coating, no amount of money moves it, because the mechanism is
switched off.

Sorption is not gated. A hygroscopic sorbent pulls vapour from air at 10-20% RH,
where condensation is thermodynamically blocked. It costs energy to regenerate:
roughly 1-3 kWh per litre, as HEAT rather than work.

That difference — free-but-gated versus costly-but-always-available — is the
whole decision, and it is what this file quantifies.

WHAT THIS IS AND IS NOT

This is a feasibility screen, not a design. It answers "which mechanisms are
even possible at this humidity, and what would each cost?" It does not design a
sorbent bed, size a solar collector, or cost a build. Sorption performance
figures are taken from published devices and cited in docs/alternative-systems.md;
they are other people's measurements of other people's hardware, which is a
different and stronger kind of evidence than anything else in this repository —
and also not transferable to a build nobody here has made.

Usage:
    python 07_alternative_systems.py
    python 07_alternative_systems.py --rh 0.25 --t-air 305
    python 07_alternative_systems.py --sweep
    python 07_alternative_systems.py --budget-check
"""

import argparse
import importlib.util
import math
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
sat_vp = vs.saturation_vapor_pressure

# Physical constants  [STANDARD]
R_UNIVERSAL = 8.314      # J/(mol K)
M_WATER = 0.018015       # kg/mol
C_P_AIR = 1005.0         # J/(kg K)
L_V = 2.45e6             # J/kg
P_ATM = 101325.0         # Pa
KWH_PER_J = 1.0 / 3.6e6


def humidity_ratio(t_kelvin, rh):
    """kg water vapour per kg dry air.  [STANDARD]"""
    e = rh * sat_vp(t_kelvin)
    return 0.622 * e / max(P_ATM - e, 1.0)


def dew_point(t_kelvin, rh):
    """Dew-point temperature, K. Inverts the Magnus form.  [STANDARD]"""
    e = max(rh * sat_vp(t_kelvin), 1.0)
    ln = math.log(e / 610.94)
    return 273.15 + 243.04 * ln / (17.625 - ln)


def minimum_work(t_kelvin, rh):
    """
    Thermodynamic floor: reversible isothermal work to extract 1 kg of water
    from air at relative humidity rh.  w = (RT/M) ln(1/rh).  [STANDARD]

    Included because it settles a common confusion. The floor is tiny — a few
    hundredths of a kWh per litre even in desert air. Nothing here is limited by
    thermodynamics. Everything is limited by how badly real devices miss it.
    """
    return (R_UNIVERSAL * t_kelvin / M_WATER) * math.log(1.0 / max(rh, 1e-6))


# ---------------------------------------------------------------------------
# Mechanisms
# ---------------------------------------------------------------------------

# Sorption performance from published devices. These are MEASUREMENTS, from
# other groups' hardware — see docs/alternative-systems.md for sources.
SORPTION = {
    'rh_min': 0.11,               # LiCl-polyacrylamide hydrogel, Atacama field test
    'yield_low_rh': 5.5,          # kg/m^2/day, advanced sorbent, low humidity
    'yield_high_rh': 16.9,        # kg/m^2/day, higher humidity
    'yield_modest': 1.7,          # kg/m^2/day, MIT hydrogel device at 50% RH
    'kwh_per_l_thermal': (1.0, 3.0),
}

# Solar resource. Drought regions are sunny — that is not incidental, it is the
# same clear sky that fails to save dew.  [ASSUMED range, conventional values]
INSOLATION_KWH_M2_DAY = {'drought_clear': 6.0, 'temperate': 4.0, 'cloudy': 2.5}
SOLAR_THERMAL_EFF = 0.12          # measured integrated solar AWH system
SOLAR_PV_EFF = 0.18               # [ASSUMED] conventional panel

FIELD_ELECTRICAL_BUDGET_KWH = 3.0 * 12 / 1000.0   # 3 W/m^2 over a 12 h night


def passive_dew_yield(t_air, rh, samples=400, rng_seed=0):
    """Modelled passive dew, mL/m^2/night, using this project's energy balance."""
    rng = np.random.default_rng(rng_seed)
    config = {'sky_view_factor': 0.95, 'local_vapor_boost': 0.0,
              'surface_emissivity': 0.95, 'tilt_deg': 30.0, 'insulation_r': 1.0,
              'electrical_w_m2': 0.0, 'cop_cooling': 0.5, 'night_hours': 11.0}
    out = np.zeros(samples)
    for i in range(samples):
        v = dict(config, t_air_night=t_air, rh=rh,
                 wind_speed=rng.uniform(0.0, 3.0),
                 cloud_cover=rng.uniform(0.0, 0.25))
        r = DewEnergyBalance(**v).solve()
        out[i] = 0.0 if r['frozen'] else r['yield_mm'] * 1000
    return float(out.mean()), float((out > 0).mean())


# Effective COP calibrated against a published measurement: a dehumidifier-based
# harvester consumed 1.02 kWh/L at 30 C / 62% RH. This model's raw thermal load
# at that condition is 1.762 kWh/L, implying an end-to-end COP of 1.73 — well
# below a nameplate figure, because it absorbs fan work, cycling, heat-exchanger
# approach and everything else a real machine pays for.
#
# This is the only calibration against measured data anywhere in this repository.
# An earlier version assumed COP 2.5, which flattered condensation by about 1.4x.
COP_CALIBRATED = 1.73


def active_condensation_energy(t_air, rh, t_cold=None, cop=COP_CALIBRATED):
    """
    Electrical kWh per litre to condense water by chilling air.  [STANDARD]

    Cool air below its dew point: pay sensible heat for ALL the air processed,
    plus latent heat for the water removed. At low RH the air-to-water ratio
    explodes, so the sensible term dominates and the cost runs away. This is the
    quantitative reason condensation fails in dry air, and it applies to a
    Peltier exactly as it applies to a compressor.
    """
    t_dp = dew_point(t_air, rh)
    if t_cold is None:
        t_cold = t_dp - 3.0
    w_in = humidity_ratio(t_air, rh)
    w_out = humidity_ratio(t_cold, 1.0)
    dw = w_in - w_out
    if dw <= 1e-9:
        return float('inf'), float('inf')
    air_kg_per_kg_water = 1.0 / dw
    sensible = air_kg_per_kg_water * C_P_AIR * (t_air - t_cold)
    latent = L_V
    thermal_j = sensible + latent
    return thermal_j * KWH_PER_J / cop, air_kg_per_kg_water


def evaluate(t_air, rh, insolation='drought_clear'):
    """Everything worth knowing at one condition."""
    t_dp = dew_point(t_air, rh)
    depression = t_air - t_dp
    dew_ml, dew_nights = passive_dew_yield(t_air, rh)
    cond_kwh, air_ratio = active_condensation_energy(t_air, rh)
    floor_kwh = minimum_work(t_air, rh) * KWH_PER_J

    sun = INSOLATION_KWH_M2_DAY[insolation]
    thermal_avail = sun * SOLAR_THERMAL_EFF
    electric_avail = sun * SOLAR_PV_EFF

    lo, hi = SORPTION['kwh_per_l_thermal']
    sorption_feasible = rh >= SORPTION['rh_min']
    sorb_lo = thermal_avail / hi * 1000 if sorption_feasible else 0.0
    sorb_hi = thermal_avail / lo * 1000 if sorption_feasible else 0.0
    # Sorbent capacity caps what a square metre can cycle per day.
    cap = (SORPTION['yield_low_rh'] if rh < 0.4 else SORPTION['yield_high_rh']) * 1000
    sorb_lo, sorb_hi = min(sorb_lo, cap), min(sorb_hi, cap)

    cond_solar_ml = (electric_avail / cond_kwh * 1000) if cond_kwh < float('inf') else 0.0

    return {
        't_air': t_air, 'rh': rh, 'dew_point': t_dp, 'depression': depression,
        'dew_ml': dew_ml, 'dew_nights': dew_nights,
        'cond_kwh_per_l': cond_kwh, 'air_ratio': air_ratio,
        'cond_solar_ml': cond_solar_ml,
        'floor_kwh_per_l': floor_kwh,
        'sorption_feasible': sorption_feasible,
        'sorb_lo': sorb_lo, 'sorb_hi': sorb_hi,
        'thermal_avail': thermal_avail, 'electric_avail': electric_avail,
    }


def report_point(r):
    print(f"\nConditions: {r['t_air'] - 273.15:.0f} C, {r['rh'] * 100:.0f}% RH")
    print(f"  dew point {r['dew_point'] - 273.15:.1f} C — the surface must fall "
          f"{r['depression']:.1f} K below air temperature")
    print(f"  radiative cooling delivers roughly 3-9 K")
    print(f"\n  {'mechanism':<26}{'feasible':>10}{'yield mL/m2/day':>18}"
          f"{'energy kWh/L':>15}")
    gated = r['depression'] > 9.0
    print(f"  {'passive radiative dew':<26}{'NO' if gated else 'yes':>10}"
          f"{r['dew_ml']:>18.0f}{'0 (free)':>15}")
    print(f"  {'active condensation':<26}{'yes':>10}"
          f"{r['cond_solar_ml']:>18.0f}{r['cond_kwh_per_l']:>15.2f}")
    lo, hi = SORPTION['kwh_per_l_thermal']
    sorb_yield = (f"{r['sorb_lo']:.0f}-{r['sorb_hi']:.0f}"
                  if r['sorption_feasible'] else "0")
    sorb_energy = f"{lo:.0f}-{hi:.0f} (heat)"
    print(f"  {'sorption + solar heat':<26}"
          f"{'yes' if r['sorption_feasible'] else 'NO':>10}"
          f"{sorb_yield:>18}{sorb_energy:>15}")
    print(f"\n  thermodynamic floor: {r['floor_kwh_per_l']:.3f} kWh/L — nothing "
          f"here is limited by physics,")
    print(f"  only by how far real devices sit above it "
          f"({r['cond_kwh_per_l'] / max(r['floor_kwh_per_l'], 1e-9):.0f}x for "
          f"condensation)")
    if r['air_ratio'] < float('inf'):
        print(f"  condensation must chill {r['air_ratio']:.0f} kg of air per kg "
              f"of water at this humidity")


def main():
    parser = argparse.ArgumentParser(
        description='Compare water-harvesting mechanisms in dry-air drought.')
    parser.add_argument('--rh', type=float, default=0.25)
    parser.add_argument('--t-air', type=float, default=305.0)
    parser.add_argument('--insolation', default='drought_clear',
                        choices=list(INSOLATION_KWH_M2_DAY))
    parser.add_argument('--sweep', action='store_true')
    parser.add_argument('--budget-check', action='store_true')
    parser.add_argument('--output', default='alternative_systems.png')
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()

    print("=" * 74)
    print("Alternative Systems for Severe Drought")
    print("=" * 74)
    print("H11 found that passive dew fails in dry-air drought. This asks what")
    print("works instead, and what it costs.")

    r = evaluate(args.t_air, args.rh, args.insolation)
    report_point(r)

    if args.sweep:
        print("\n" + "-" * 74)
        print("HUMIDITY SWEEP — where each mechanism switches on")
        print("-" * 74)
        print("Dew depends on temperature as well as humidity: warm air needs a")
        print("bigger absolute depression, and a warm humid sky radiates more")
        print("back. Dew columns are mL/m2/night at three air temperatures.\n")
        print(f"{'RH':>5}{'dew 15C':>9}{'dew 22C':>9}{'dew 32C':>9}"
              f"{'cond kWh/L':>12}{'cond mL':>9}{'sorption mL':>14}")
        for rh in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90):
            row = evaluate(args.t_air, rh, args.insolation)
            d15, _ = passive_dew_yield(288.0, rh, samples=250)
            d22, _ = passive_dew_yield(295.0, rh, samples=250)
            d32, _ = passive_dew_yield(305.0, rh, samples=250)
            cond = (f"{row['cond_kwh_per_l']:.1f}"
                    if row['cond_kwh_per_l'] < 999 else "inf")
            sorb = (f"{row['sorb_lo']:.0f}-{row['sorb_hi']:.0f}"
                    if row['sorption_feasible'] else "blocked")
            print(f"{rh:>5.2f}{d15:>9.0f}{d22:>9.0f}{d32:>9.0f}"
                  f"{cond:>12}{row['cond_solar_ml']:>9.0f}{sorb:>14}")
        print("\nDew does not decline gracefully — it stops. The wall sits near")
        print("70% RH at 15 C and near 90% at 32 C: hot dry air is the worst")
        print("case and it is exactly what severe drought looks like. Below the")
        print("wall the yield is not small, it is zero, and no design change")
        print("crosses it. Sorption has no such wall down to about 11% RH.")

    if args.budget_check:
        print("\n" + "-" * 74)
        print("WHAT THE FIELD BUDGET AFFORDS")
        print("-" * 74)
        sun = INSOLATION_KWH_M2_DAY[args.insolation]
        print(f"  $45 build, electrical           "
              f"{FIELD_ELECTRICAL_BUDGET_KWH:.3f} kWh/m2/day")
        print(f"  1 m2 solar PV at {SOLAR_PV_EFF:.0%}           "
              f"{sun * SOLAR_PV_EFF:.2f} kWh/m2/day")
        print(f"  1 m2 solar THERMAL at {SOLAR_THERMAL_EFF:.0%}      "
              f"{sun * SOLAR_THERMAL_EFF:.2f} kWh/m2/day")
        print()
        print(f"  Sorption needs heat, not work. A solar thermal collector")
        print(f"  delivers {sun * SOLAR_THERMAL_EFF / FIELD_ELECTRICAL_BUDGET_KWH:.0f}x "
              f"the energy the current build has, in the form")
        print(f"  the mechanism actually wants, from the clear skies that")
        print(f"  define the drought.")
        print()
        print(f"  The same clear sky that cannot save dew — because it cannot")
        print(f"  supply vapour — powers sorption, because it supplies sun.")
        print(f"  H11's anti-correlation reverses for this mechanism.")

    print("\n" + "-" * 74)
    print("AN UNRESOLVED DISCREPANCY")
    print("-" * 74)
    lo, hi = SORPTION['kwh_per_l_thermal']
    cap_low = SORPTION['yield_low_rh'] * 1000
    print(f"  Two sourced numbers do not reconcile, and neither is dismissed:")
    print(f"    energy route : 1 m2 solar thermal / {lo:.0f}-{hi:.0f} kWh per L")
    print(f"                   = {r['thermal_avail']/hi*1000:.0f}-"
          f"{r['thermal_avail']/lo*1000:.0f} mL/m2/day")
    print(f"    device route : published sorbents report {cap_low:.0f} mL/m2/day")
    print(f"                   at low humidity")
    print(f"  About a 10x gap. Plausible explanations: those devices use a solar")
    print(f"  aperture larger than their sorbent area, or run multiple cycles a")
    print(f"  day, or the 1-3 kWh/L figure does not describe them. This could")
    print(f"  not be settled from the sources reachable here, so the smaller,")
    print(f"  energy-limited figure is the one reported above. If the device")
    print(f"  figure is right, sorption is far better than this screen suggests.")
    print(f"  See docs/research-log.md, O17.")

    print("\n" + "-" * 74)
    print("CAVEAT")
    print("-" * 74)
    print("  Sorption figures are other groups' measurements of other groups'")
    print("  hardware (docs/alternative-systems.md). They are stronger evidence")
    print("  than anything else here — and not transferable to a build nobody")
    print("  in this project has made. Nothing in this file has been built or")
    print("  measured by this project. It is a screen for what is worth")
    print("  attempting, not a claim that any of it works.")
    print()

    if not args.no_plot:
        rhs = np.linspace(0.10, 0.95, 30)
        dew, sorb, cond = [], [], []
        for rh in rhs:
            s = evaluate(args.t_air, float(rh), args.insolation)
            dew.append(s['dew_ml'])
            sorb.append((s['sorb_lo'] + s['sorb_hi']) / 2)
            cond.append(s['cond_solar_ml'])
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(rhs * 100, dew, 'o-', color='#1565c0', label='passive dew (free)')
        ax.plot(rhs * 100, sorb, 's-', color='#2e7d32',
                label='sorption + solar heat')
        ax.plot(rhs * 100, cond, '^-', color='#c62828',
                label='active condensation + solar PV')
        ax.set_yscale('symlog', linthresh=1)
        ax.set_xlabel('relative humidity (%)')
        ax.set_ylabel('mL per m² per day (log scale)')
        ax.set_title(f'Harvesting mechanisms vs humidity — '
                     f'{args.t_air - 273.15:.0f} C, {args.insolation}\n'
                     f'sorption from published devices; dew from this '
                     f'project\'s unvalidated model', fontsize=11)
        ax.axvspan(10, 40, alpha=0.08, color='red')
        ax.annotate('severe dry-air drought', xy=(25, ax.get_ylim()[1] * 0.4),
                    ha='center', fontsize=9, color='#c62828')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"Graph saved to: {args.output}")
        plt.close(fig)


if __name__ == '__main__':
    main()
