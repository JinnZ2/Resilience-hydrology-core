#!/usr/bin/env python3
"""
Constrained variable search: which variables actually move dew yield, and over
what range.

WHAT THIS DOES

Samples a constrained variable space, evaluates each sample against a dew
energy-balance model, and reports:

  1. Which variables the outcome is actually sensitive to (first-order index)
  2. The RANGE of each variable that the best outcomes occupy — not a single
     "optimal" point
  3. Whether that range sits at a bound (a corner solution, i.e. the search is
     telling you nothing except "more/less is better")
  4. Which variables have an INTERIOR optimum — a real best range, where both
     too little and too much hurt
  5. Which high-leverage variables have never been measured in the field

Point 3 exists because of how 03_seed_optimization.py failed: it reported an
"optimal" seed that was really the corner of its search box, with no signal that
the answer was degenerate (docs/research-log.md, H5). Every range this script
reports is checked for that condition and labelled.

WHY A DIFFERENT MODEL

01_basic_dew.py computes `RH * delta_T * 0.02 * amplification`. Correlating
variables against that is circular — it can only rediscover its own two inputs
and a constant. To ask which ecological variables matter, the model has to
contain channels for them to act through. So this file uses a surface energy
balance instead:

    radiative loss to sky + active cooling
        = convective gain from air + latent heat of condensation + conduction

solved for surface temperature, with condensation from the vapour-pressure
gradient via the heat/mass transfer analogy. Wind, cloud cover, canopy openness,
tilt, emissivity, and insulation all enter this balance physically rather than
by assertion.

STATUS OF THE OUTPUT — READ THIS

Every number here is a property of THIS MODEL, not of the atmosphere. The model
is structurally more defensible than 01_basic_dew.py but it is still
unvalidated: no output of this repository has ever been compared against a field
measurement (docs/research-log.md, O1).

So the correlations below are not findings about dew. They are HYPOTHESES ABOUT
WHERE TO LOOK, and the most useful thing this script produces is the measurement
priority list at the end of the report — the variables whose value would most
change the answer and which nobody has measured yet.

Coefficient provenance is tagged in the source: [STANDARD] for established
physical relations, [ASSUMED] for numbers this project picked and has not
justified. Every [ASSUMED] coefficient is a candidate falsification target.

Usage:
    python 04_variable_search.py
    python 04_variable_search.py --climate arid --samples 8000
    python 04_variable_search.py --list-variables
    python 04_variable_search.py --fix wind_speed=1.2 --fix tilt_deg=30
    python 04_variable_search.py --energy-budget 5.0 --top-frac 0.05
"""

import argparse
import math
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Physical constants  [STANDARD]
# ---------------------------------------------------------------------------

SIGMA = 5.670374419e-8   # Stefan-Boltzmann, W/(m^2 K^4)
L_V = 2.45e6             # latent heat of vaporisation, J/kg
C_P = 1005.0             # specific heat of air, J/(kg K)
P_ATM = 101325.0         # atmospheric pressure, Pa
MW_RATIO = 0.622         # molecular weight ratio water/dry air
LEWIS_FACTOR = 0.91      # Le^(2/3) for water vapour in air, Le ~ 0.87
T_FREEZE = 273.15        # K

# Convective transfer coefficient, h_c = H_C_STILL + H_C_WIND * u  [ASSUMED]
# Linear-in-windspeed form is standard for flat plates; these two coefficients
# are plausible values this project has not justified against measurement.
H_C_STILL = 2.5          # W/(m^2 K)
H_C_WIND = 3.0           # W/(m^2 K) per m/s

# Droplet collection efficiency vs tilt: eff = EFF_MAX * (1 - exp(-tilt/TILT_CHAR))
# Shape (steeper drains better, with diminishing returns) is physically sensible;
# both numbers are [ASSUMED].
EFF_MAX = 0.95
TILT_CHAR = 18.0         # degrees


# ---------------------------------------------------------------------------
# Variable space — this is the "list of variable constraints"
# ---------------------------------------------------------------------------

# kind:   climate  - not controllable, set by where and when you are
#         siting   - controllable by placement, vegetation, and land management
#         design   - controllable by what you build
#         control  - controllable at runtime, costs energy
#
# status: MEASURED   - a field value exists in this repository
#         UNMEASURED - nobody has recorded this in the field yet
#         DESIGN     - you choose it, so it needs no measurement

@dataclass
class Variable:
    name: str
    unit: str
    low: float
    high: float
    kind: str
    status: str
    channel: str          # how it physically acts in the model
    note: str = ''

    @property
    def span(self):
        return self.high - self.low


VARIABLES = [
    Variable('t_air_night', 'K', 283.0, 303.0, 'climate', 'MEASURED',
             'sets both the radiative sink and the saturation vapour pressure',
             'the ESP32 logger records this'),
    Variable('rh', 'fraction', 0.15, 0.95, 'climate', 'UNMEASURED',
             'sets air vapour pressure, the source term for condensation',
             'no humidity sensor in the current build - see O1'),
    Variable('wind_speed', 'm/s', 0.0, 6.0, 'climate', 'UNMEASURED',
             'raises convective heat gain (warms the surface) and mass transfer '
             '(feeds vapour) at the same time',
             'reducible by shelterbelt planting - an ecological lever'),
    Variable('cloud_cover', 'fraction', 0.0, 1.0, 'climate', 'UNMEASURED',
             'raises sky emissivity, cutting the radiative cooling that drives '
             'the whole process',
             'recoverable from local weather records for past field nights'),
    Variable('sky_view_factor', 'fraction', 0.30, 1.00, 'siting', 'UNMEASURED',
             'fraction of the hemisphere that is cold sky rather than warm '
             'canopy or structure',
             'set by canopy openness and obstructions - the main ecological '
             'siting lever'),
    Variable('local_vapor_boost', 'fraction', 0.0, 0.15, 'siting', 'UNMEASURED',
             'fractional rise in near-surface vapour pressure from soil and '
             'plant moisture upwind',
             '[ASSUMED] channel: mulch, wet soil, and vegetation as a local '
             'vapour source. Bounded small on purpose'),
    Variable('surface_emissivity', 'fraction', 0.85, 0.98, 'design', 'DESIGN',
             'scales radiative loss directly',
             'material choice: most paints and plastics sit near 0.9'),
    Variable('tilt_deg', 'degrees', 0.0, 60.0, 'design', 'DESIGN',
             'trades drainage (steeper collects more of what forms) against '
             'sky view (steeper sees less cold sky)',
             'the clearest interior-optimum candidate in the set'),
    Variable('insulation_r', 'm^2K/W', 0.05, 2.00, 'design', 'DESIGN',
             'resists conductive heat leak from the substrate into the cold '
             'surface',
             'the trailer build had none - see docs/trailer-build.md'),
    Variable('night_hours', 'hours', 8.0, 14.0, 'climate', 'MEASURED',
             'how long the condensing window lasts',
             'latitude and season; the lat parameter dropped in the 2026-03 '
             'merge, preserved in legacy/'),
    Variable('electrical_w_m2', 'W/m^2', 0.0, 8.0, 'control', 'DESIGN',
             'electrical power spent on active cooling; delivers '
             'electrical_w_m2 * cop_cooling of heat removal',
             'this is what "system ON" actually costs. Sampled within the '
             '--energy-budget, so the budget shapes the search space instead '
             'of discarding samples afterwards'),
    Variable('cop_cooling', 'ratio', 0.20, 1.20, 'control', 'DESIGN',
             'cooling delivered per watt of electricity; converts the energy '
             'budget into actual heat removal',
             'Peltier modules are poor here, which is why the energy budget '
             'binds so hard'),
]

VAR_BY_NAME = {v.name: v for v in VARIABLES}


# Climate presets narrow the climate variables to a locally plausible window.
# Ranges rather than points: a preset is a place, and a place has weather.
#
# IMPORTANT — these RH values are NOT the ones in 01_basic_dew.py and
# 03_seed_optimization.py. Those files carry a single RH per climate (0.25 for
# arid, 0.35 semi-arid, 0.45 mediterranean, 0.40 tropical dry), which are
# daytime figures. Dew is governed by NEAR-SURFACE RH IN THE HOURS BEFORE DAWN,
# which is far higher than the daytime value at the same site: the air cools
# toward its dew point overnight while absolute humidity changes little.
#
# The distinction is not cosmetic. At RH 0.25-0.45 the dew-point depression is
# 12-18 K, while radiative cooling of a good passive surface delivers only
# 3-9 K. Dew cannot form. Running this model on the repo's daytime presets
# returns exactly zero yield for every sample — see docs/research-log.md, H7.
#
# Windows below are pre-dawn values. [ASSUMED] — plausible for each climate
# class, not measured, and the field build has no humidity sensor at all (O1).
CLIMATES = {
    'arid': {
        't_air_night': (283.0, 293.0), 'rh': (0.35, 0.80),
        'wind_speed': (0.0, 5.0), 'cloud_cover': (0.0, 0.30),
        'night_hours': (10.0, 13.0),
    },
    'semi_arid': {
        't_air_night': (285.0, 295.0), 'rh': (0.45, 0.90),
        'wind_speed': (0.0, 5.0), 'cloud_cover': (0.0, 0.50),
        'night_hours': (9.0, 14.0),
    },
    'mediterranean': {
        't_air_night': (288.0, 297.0), 'rh': (0.55, 0.97),
        'wind_speed': (0.0, 4.0), 'cloud_cover': (0.0, 0.60),
        'night_hours': (9.0, 14.0),
    },
    'tropical_dry': {
        't_air_night': (292.0, 300.0), 'rh': (0.50, 0.95),
        'wind_speed': (0.0, 3.5), 'cloud_cover': (0.0, 0.55),
        'night_hours': (10.0, 12.5),
    },
    # The daytime RH values the rest of the repository uses, for comparison.
    # Expect zero yield: this preset exists to demonstrate H7, not to be used.
    'repo_daytime_rh': {
        't_air_night': (288.0, 295.0), 'rh': (0.25, 0.45),
        'wind_speed': (0.0, 5.0), 'cloud_cover': (0.0, 0.30),
        'night_hours': (10.0, 13.0),
    },
    'all': {},  # full declared bounds, no climate narrowing
}


@dataclass
class Constraints:
    """Feasibility limits applied to every sample."""

    max_electrical_w_m2: float = 3.0   # energy budget: solar-realistic at night
    allow_freeze: bool = False         # frost night = lost collection
    condensing_only: bool = False      # analyse only nights that made water
    pinned: dict = field(default_factory=dict)

    def describe(self):
        lines = [f"energy budget      <= {self.max_electrical_w_m2:.2f} W/m^2 electrical",
                 f"freezing surface   {'allowed' if self.allow_freeze else 'rejected (frost = lost night)'}"]
        if self.condensing_only:
            lines.append("analysis set       condensing nights only "
                         "(isolates design levers from whether dew forms)")
        for name, value in self.pinned.items():
            lines.append(f"pinned             {name} = {value:g} {VAR_BY_NAME[name].unit}")
        return lines


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------

def saturation_vapor_pressure(t_kelvin):
    """Saturation vapour pressure over water, Pa. Magnus form.  [STANDARD]"""
    t_c = t_kelvin - 273.15
    return 610.94 * math.exp(17.625 * t_c / (t_c + 243.04))


def clear_sky_emissivity(vapor_pressure_pa, t_air):
    """Brutsaert clear-sky emissivity.  [STANDARD]"""
    e_hpa = vapor_pressure_pa / 100.0
    eps = 1.24 * (e_hpa / t_air) ** (1.0 / 7.0)
    return min(max(eps, 0.60), 1.0)


class DewEnergyBalance:
    """
    Surface energy balance for a passive or actively cooled condenser.

    At equilibrium, per m^2 of condenser surface:

        radiative loss + active cooling
            = convective gain + latent release + conductive gain

    Solved for surface temperature, which then sets the condensation rate.
    """

    def __init__(self, **v):
        self.v = v

    def _p_cool(self):
        """Heat removal actually delivered, W/m^2, from power spent and COP."""
        return self.v['electrical_w_m2'] * self.v['cop_cooling']

    def _air_vapor_pressure(self):
        v = self.v
        e_sat_air = saturation_vapor_pressure(v['t_air_night'])
        e_air = v['rh'] * e_sat_air * (1.0 + v['local_vapor_boost'])
        return min(e_air, e_sat_air)   # cannot exceed saturation

    def _view_factors(self):
        """Tilting the plate trades sky view for drainage.  [STANDARD]"""
        v = self.v
        tilt_rad = math.radians(v['tilt_deg'])
        # view factor of a tilted plane to the sky hemisphere
        sky_fraction = v['sky_view_factor'] * (1.0 + math.cos(tilt_rad)) / 2.0
        return sky_fraction

    def _mass_flux(self, t_surface, h_c, e_air):
        """
        Condensation rate, kg/(m^2 s), from the heat/mass transfer analogy.
        [STANDARD] relation; zero when the surface is above the dew point.
        """
        delta_e = e_air - saturation_vapor_pressure(t_surface)
        if delta_e <= 0:
            return 0.0
        return (h_c / C_P) * MW_RATIO * delta_e / (P_ATM * LEWIS_FACTOR)

    def _imbalance(self, t_surface, h_c, e_air, sky_fraction):
        """Net loss minus net gain. Zero at the equilibrium surface temperature."""
        v = self.v
        t_air = v['t_air_night']

        eps_sky = clear_sky_emissivity(e_air, t_air)
        eps_sky_eff = eps_sky + (1.0 - eps_sky) * v['cloud_cover']

        # Hemisphere splits into cold sky and warm surroundings (emissivity ~1
        # at air temperature).
        absorbed = sky_fraction * eps_sky_eff + (1.0 - sky_fraction)
        radiative_loss = v['surface_emissivity'] * SIGMA * (
            t_surface ** 4 - absorbed * t_air ** 4)

        convective_gain = h_c * (t_air - t_surface)
        latent_gain = L_V * self._mass_flux(t_surface, h_c, e_air)
        conductive_gain = (1.0 / v['insulation_r']) * (t_air - t_surface)

        return (radiative_loss + self._p_cool()
                - convective_gain - latent_gain - conductive_gain)

    def solve(self):
        """
        Returns dict with surface temperature, yield, energy use, and flags.
        Yield is mm per night per m^2 (1 kg/m^2 == 1 mm).
        """
        v = self.v
        t_air = v['t_air_night']
        h_c = H_C_STILL + H_C_WIND * v['wind_speed']
        e_air = self._air_vapor_pressure()
        sky_fraction = self._view_factors()

        def f(t_s):
            return self._imbalance(t_s, h_c, e_air, sky_fraction)

        lo, hi = t_air - 45.0, t_air + 2.0
        try:
            if f(lo) > 0 or f(hi) < 0:
                # No bracketed root: surface cannot be driven below air
                # temperature under these conditions.
                t_surface = t_air
            else:
                t_surface = brentq(f, lo, hi, xtol=1e-3)
        except (ValueError, OverflowError):
            t_surface = t_air

        m_dot = self._mass_flux(t_surface, h_c, e_air)
        seconds = v['night_hours'] * 3600.0

        tilt_eff = EFF_MAX * (1.0 - math.exp(-v['tilt_deg'] / TILT_CHAR))
        formed_mm = m_dot * seconds
        frozen = t_surface < T_FREEZE

        return {
            't_surface': t_surface,
            'dew_point_depression': t_air - t_surface,
            'formed_mm': formed_mm,
            'yield_mm': formed_mm * tilt_eff,
            'electrical_w_m2': v['electrical_w_m2'],
            'p_cool': self._p_cool(),
            'frozen': frozen,
        }


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

class VariableSearch:
    """Sample a constrained variable space and analyse what drives the outcome."""

    def __init__(self, climate='semi_arid', constraints=None, rng_seed=0):
        self.climate = climate
        self.constraints = constraints or Constraints()
        self.rng = np.random.default_rng(rng_seed)
        self.bounds = self._effective_bounds()
        self.free = [v.name for v in VARIABLES
                     if v.name not in self.constraints.pinned]

    def _effective_bounds(self):
        """Declared bounds, narrowed by climate preset, overridden by pins."""
        preset = CLIMATES[self.climate]
        bounds = {}
        for v in VARIABLES:
            lo, hi = v.low, v.high
            if v.name in preset:
                p_lo, p_hi = preset[v.name]
                lo, hi = max(lo, p_lo), min(hi, p_hi)
            # The energy budget is a bound on the search space, not a filter
            # applied afterwards: sampling power we cannot afford would throw
            # away most of the samples.
            if v.name == 'electrical_w_m2':
                hi = min(hi, self.constraints.max_electrical_w_m2)
                lo = min(lo, hi)
            if v.name in self.constraints.pinned:
                value = self.constraints.pinned[v.name]
                lo = hi = value
            bounds[v.name] = (lo, hi)
        return bounds

    def sample(self, n):
        """Latin hypercube over the free variables; pinned ones held fixed."""
        try:
            from scipy.stats import qmc
            engine = qmc.LatinHypercube(d=max(len(self.free), 1),
                                        seed=int(self.rng.integers(2 ** 31)))
            unit = engine.random(n)
        except Exception:
            unit = self.rng.random((n, max(len(self.free), 1)))

        columns = {}
        for i, name in enumerate(self.free):
            lo, hi = self.bounds[name]
            columns[name] = lo + unit[:, i] * (hi - lo)
        for name, value in self.constraints.pinned.items():
            columns[name] = np.full(n, float(value))
        return columns

    def evaluate(self, columns, n):
        """Run the model on every sample; return outcome arrays."""
        out = {k: np.zeros(n) for k in
               ('yield_mm', 'electrical_w_m2', 't_surface', 'dew_point_depression')}
        out['frozen'] = np.zeros(n, dtype=bool)

        for i in range(n):
            kwargs = {name: float(columns[name][i]) for name in columns}
            result = DewEnergyBalance(**kwargs).solve()
            for key in ('yield_mm', 'electrical_w_m2', 't_surface',
                        'dew_point_depression'):
                out[key][i] = result[key]
            out['frozen'][i] = result['frozen']

        over_budget = out['electrical_w_m2'] > self.constraints.max_electrical_w_m2
        feasible = ~over_budget
        if not self.constraints.allow_freeze:
            feasible &= ~out['frozen']
        out['over_budget'] = over_budget
        out['condensed'] = out['yield_mm'] > 0
        if self.constraints.condensing_only:
            feasible &= out['condensed']
        out['feasible'] = feasible
        return out


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def first_order_index(x, y, bins=10):
    """
    Fraction of outcome variance explained by this variable alone.

    Var(E[Y|X]) / Var(Y), estimated by binning. Captures non-monotonic effects
    that a correlation coefficient misses — which matters here, because the
    interesting variables are the ones with an interior optimum.
    """
    if np.var(y) <= 0 or np.ptp(x) <= 0:
        return 0.0
    edges = np.quantile(x, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
    means, weights = [], []
    for b in range(len(edges) - 1):
        sel = idx == b
        if sel.sum() >= 2:
            means.append(y[sel].mean())
            weights.append(sel.sum())
    if len(means) < 2:
        return 0.0
    means, weights = np.array(means), np.array(weights, dtype=float)
    grand = np.average(means, weights=weights)
    between = np.average((means - grand) ** 2, weights=weights)
    return float(np.clip(between / np.var(y), 0.0, 1.0))


def response_curve(x, y, bins=10):
    """Mean outcome per quantile bin of x. Returns (centres, means)."""
    edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return np.array([]), np.array([])
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
    centres, means = [], []
    for b in range(len(edges) - 1):
        sel = idx == b
        if sel.sum() >= 2:
            centres.append(0.5 * (edges[b] + edges[b + 1]))
            means.append(y[sel].mean())
    return np.array(centres), np.array(means)


def analyse_variable(name, x, y, x_top, bounds, top_frac):
    """Sensitivity, best-performing range, and degeneracy flags for one variable."""
    var = VAR_BY_NAME[name]
    lo, hi = bounds[name]
    span = hi - lo

    if span <= 0:
        return None   # pinned

    s1 = first_order_index(x, y)
    if len(x) > 2 and np.ptp(x) > 0 and np.ptp(y) > 0:
        rho = spearmanr(x, y).statistic
        rho = 0.0 if np.isnan(rho) else float(rho)
    else:
        rho = 0.0

    p10, p90 = np.percentile(x_top, [10, 90])
    # A uniform sample's p10-p90 covers 80% of the range; anything tighter means
    # the good outcomes are genuinely selective about this variable.
    narrowing = float(np.clip(1.0 - (p90 - p10) / (0.8 * span), 0.0, 1.0))

    edge = 0.05 * span
    at_low = p10 <= lo + edge
    at_high = p90 >= hi - edge
    if narrowing < 0.05:
        # The winners span essentially the whole box: touching a bound here
        # means nothing, the variable simply is not selective.
        bound_flag = 'no constraint'
    elif at_low and at_high:
        bound_flag = 'unconstrained'
    elif at_high:
        bound_flag = 'AT UPPER BOUND'
    elif at_low:
        bound_flag = 'AT LOWER BOUND'
    else:
        bound_flag = 'interior'

    centres, means = response_curve(x, y)
    interior_peak = None
    if len(means) >= 3:
        peak = int(np.argmax(means))
        if 0 < peak < len(means) - 1:
            interior_peak = float(centres[peak])

    return {
        'name': name, 'var': var, 'low': lo, 'high': hi,
        's1': s1, 'rho': rho, 'p10': float(p10), 'p90': float(p90),
        'narrowing': narrowing, 'bound_flag': bound_flag,
        'interior_peak': interior_peak, 'centres': centres, 'means': means,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(search, columns, results, analyses, n, top_frac):
    c = search.constraints
    feasible = results['feasible']
    n_feasible = int(feasible.sum())

    print("=" * 72)
    print("Constrained Variable Search")
    print("=" * 72)
    print(f"Climate preset : {search.climate}")
    print(f"Samples        : {n} (Latin hypercube)")
    for line in c.describe():
        print(f"  {line}")
    print()

    print(f"Feasible       : {n_feasible} / {n} ({n_feasible / n:.0%})")
    print(f"  rejected, over energy budget : {int(results['over_budget'].sum())}"
          f"   (budget bounds the search space, so this is 0 unless pinned)")
    print(f"  rejected, surface below 0 C  : "
          f"{int((results['frozen'] & ~results['over_budget']).sum())}"
          f"{'  (counted as feasible)' if c.allow_freeze else ''}")
    if c.condensing_only:
        print(f"  rejected, no condensation    : "
              f"{int((~results['condensed']).sum())}")
    if n_feasible < 30:
        print("\n  Too few feasible samples to analyse. Loosen a constraint or")
        print("  raise --samples.\n")
        return
    if n_feasible < 200:
        print(f"\n  ! Only {n_feasible} feasible samples. Sensitivity estimates")
        print(f"    below are noisy — raise --samples before trusting the order.")

    y = results['yield_mm'][feasible]
    condensing = float((y > 0).mean())
    if not c.condensing_only:
        print(f"\nCondensing nights : {condensing:.0%} of feasible samples reached "
              f"the dew point")
        if 0.0 < condensing < 0.9:
            print(f"  With {1 - condensing:.0%} of samples at exactly zero, the")
            print(f"  ranking below is dominated by WHETHER dew forms, which is")
            print(f"  mostly humidity. Rerun with --condensing-only to see what")
            print(f"  the design variables do once it does form.")
    print(f"Yield across feasible samples (mm/night per m^2):")
    print(f"  median {np.median(y):.3f}   p90 {np.percentile(y, 90):.3f}   "
          f"max {y.max():.3f}")
    print(f"  For reference, 01_basic_dew.py asserts 0.162-0.300 mm/day for the")
    print(f"  system-on case using a hard-coded 3x factor (research log H2).")

    if condensing == 0.0:
        depression = results['dew_point_depression'][feasible]
        print("\n" + "!" * 72)
        print("NULL RESULT — no sample condensed. This is a finding, not a crash.")
        print("!" * 72)
        print(f"  Cooling achieved: {depression.mean():.1f} K below air on")
        print(f"  average (max {depression.max():.1f} K). That was never enough to")
        print("  reach the dew point at the humidity sampled here.")
        print()
        print("  Dew needs the surface driven BELOW the dew point. At low RH the")
        print("  dew-point depression is 12-18 K; radiative cooling delivers")
        print("  3-9 K. No amount of tilt, emissivity, or insulation closes that")
        print("  gap — only higher humidity or far more active cooling does.")
        print()
        print("  Note that 01_basic_dew.py reports a healthy yield under these")
        print("  same conditions, because its formula never checks whether the")
        print("  surface reaches the dew point (docs/research-log.md, H7).")
        print()
        print("  Try: --climate semi_arid (pre-dawn humidity), or raise")
        print("  --energy-budget to allow real active cooling.")
        print()
        return

    ranked = sorted(analyses, key=lambda a: a['s1'], reverse=True)

    print("\n" + "-" * 72)
    print("1. SENSITIVITY — what the outcome actually responds to")
    print("-" * 72)
    print(f"{'variable':<20} {'kind':<9} {'S1':>6}  {'rank rho':>8}  effect")
    for a in ranked:
        if a['rho'] > 0.15:
            effect = 'yield rises with it'
        elif a['rho'] < -0.15:
            effect = 'yield falls with it'
        elif a['s1'] > 0.02:
            effect = 'non-monotonic - has a peak'
        else:
            effect = 'little effect'
        print(f"{a['name']:<20} {a['var'].kind:<9} {a['s1']:>6.3f}  "
              f"{a['rho']:>8.2f}  {effect}")
    print("\nS1 = fraction of yield variance explained by that variable alone.")
    print("Low S1 with high |rho| means a weak but consistent effect; high S1")
    print("with rho near zero means a strong effect with a peak in the middle.")

    print("\n" + "-" * 72)
    print(f"2. OPTIMAL RANGES — where the top {top_frac:.0%} of outcomes live")
    print("-" * 72)
    print(f"{'variable':<20} {'searched':>19}  {'best range (p10-p90)':>21}  {'narrow':>6}  status")
    for a in ranked:
        searched = f"{a['low']:.2f}-{a['high']:.2f}"
        best = f"{a['p10']:.2f}-{a['p90']:.2f}"
        print(f"{a['name']:<20} {searched:>19}  {best:>21}  "
              f"{a['narrowing']:>6.2f}  {a['bound_flag']}")
    print("\nnarrow = how much the winning range tightens vs the searched range")
    print("  (0.00 = the best outcomes use the whole range, so it doesn't")
    print("  matter; near 1.00 = you must hit a specific window).")

    corners = [a for a in ranked if a['bound_flag'].startswith('AT ') and a['s1'] > 0.02]
    if corners:
        print("\n  ! CORNER SOLUTIONS — the search hit a wall rather than an optimum:")
        for a in corners:
            side = 'upper' if 'UPPER' in a['bound_flag'] else 'lower'
            print(f"      {a['name']}: pinned to its {side} bound "
                  f"({a['low']:.2f}-{a['high']:.2f} {a['var'].unit})")
        print("    For these, 'optimal' only means 'as far as the box allowed'.")
        print("    Either the bound is a real physical/budget limit, or it is")
        print("    arbitrary and the box needs widening. This is the failure")
        print("    03_seed_optimization.py shipped undetected (research log H5).")

    interior = [a for a in ranked if a['interior_peak'] is not None and a['s1'] > 0.02]
    print("\n" + "-" * 72)
    print("3. INTERIOR OPTIMA — variables with a genuine best range")
    print("-" * 72)
    if interior:
        for a in interior:
            print(f"  {a['name']:<20} peak bin near {a['interior_peak']:.2f} "
                  f"{a['var'].unit}, good range {a['p10']:.2f}-{a['p90']:.2f} "
                  f"(S1 {a['s1']:.3f})")
            print(f"      {a['var'].channel}")
        print("\n  Trust the range, not the peak point: which bin comes out")
        print("  highest is the noisiest number in this report and moves")
        print("  between runs. The range is stable.")
    else:
        print("  None detected. Every influential variable is monotone over the")
        print("  range searched, so the answer is a bound, not a range.")

    print("\n" + "-" * 72)
    print("4. ECOLOGICAL LEVERS — what siting and land management can change")
    print("-" * 72)
    eco = [a for a in ranked if a['var'].kind == 'siting']
    for a in eco:
        print(f"  {a['name']:<20} S1 {a['s1']:.3f}   best {a['p10']:.2f}-{a['p90']:.2f} "
              f"{a['var'].unit}")
        print(f"      {a['var'].note}")
    wind = next((a for a in ranked if a['name'] == 'wind_speed'), None)
    if wind is not None:
        print(f"  {'wind_speed':<20} S1 {wind['s1']:.3f}   best "
              f"{wind['p10']:.2f}-{wind['p90']:.2f} m/s   (climate, but "
              f"reducible by")
        print(f"      shelterbelt planting - the one climate variable land "
              f"management reaches)")

    print("\n" + "-" * 72)
    print("5. MEASUREMENT PRIORITIES — high leverage, never measured")
    print("-" * 72)
    unmeasured = [a for a in ranked if a['var'].status == 'UNMEASURED']
    if unmeasured:
        for i, a in enumerate(unmeasured, 1):
            print(f"  {i}. {a['name']:<18} S1 {a['s1']:.3f}   {a['var'].note}")
        print("\n  These are model-derived priorities, not facts about dew. They")
        print("  say where a measurement would most change the answer, which is")
        print("  exactly what O1 in docs/research-log.md is asking for.")
    else:
        print("  Every influential variable has a field value. Unlikely — check")
        print("  the status tags in VARIABLES.")

    print("\n" + "-" * 72)
    print("CAVEAT")
    print("-" * 72)
    print("  All of the above describes this model, which has never been")
    print("  compared against a field measurement. Coefficients tagged")
    print("  [ASSUMED] in the source are unjustified numbers this project")
    print("  chose. Treat every ranking here as a hypothesis to test, not a")
    print("  result to build on. See docs/research-log.md.")
    print()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(analyses, top_frac, climate):
    ranked = sorted(analyses, key=lambda a: a['s1'], reverse=True)
    curved = [a for a in ranked if len(a['means']) >= 3][:8]

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.25, 1, 1], hspace=0.55, wspace=0.3)

    # Tornado of first-order sensitivity
    ax = fig.add_subplot(gs[0, :2])
    names = [a['name'] for a in ranked][::-1]
    values = [a['s1'] for a in ranked][::-1]
    colors = {'climate': '#888888', 'siting': '#2e7d32',
              'design': '#1565c0', 'control': '#c62828'}
    ax.barh(names, values,
            color=[colors[VAR_BY_NAME[n].kind] for n in names], alpha=0.85)
    ax.set_xlabel('First-order sensitivity  Var(E[Y|X]) / Var(Y)')
    ax.set_title('What moves yield')
    ax.grid(True, alpha=0.3, axis='x')
    handles = [plt.Rectangle((0, 0), 1, 1, color=v, alpha=0.85)
               for v in colors.values()]
    ax.legend(handles, colors.keys(), fontsize=8, loc='lower right')

    # Optimal-range bars
    ax = fig.add_subplot(gs[0, 2:])
    for i, a in enumerate(ranked[::-1]):
        span = a['high'] - a['low']
        if span <= 0:
            continue
        norm_lo = (a['p10'] - a['low']) / span
        norm_hi = (a['p90'] - a['low']) / span
        ax.barh(i, 1.0, left=0.0, color='#dddddd', height=0.6)
        flag = a['bound_flag']
        color = '#c62828' if flag.startswith('AT ') else '#2e7d32'
        ax.barh(i, norm_hi - norm_lo, left=norm_lo, color=color,
                height=0.6, alpha=0.9)
    ax.set_yticks(range(len(ranked)))
    ax.set_yticklabels([a['name'] for a in ranked[::-1]], fontsize=8)
    ax.yaxis.tick_right()   # keep labels clear of the tornado panel
    ax.set_xlabel('Position within searched range (0 = low bound, 1 = high bound)')
    ax.set_title(f'Where the top {top_frac:.0%} of outcomes sit\n'
                 f'(red = pinned at a bound, not a true optimum)', fontsize=10)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3, axis='x')

    # Response curves for the most influential variables
    for i, a in enumerate(curved):
        ax = fig.add_subplot(gs[1 + i // 4, i % 4])
        ax.plot(a['centres'], a['means'], 'o-', color='#1565c0', linewidth=2,
                markersize=3)
        if a['interior_peak'] is not None:
            ax.axvline(a['interior_peak'], color='#c62828', linestyle='--',
                       linewidth=1, label='interior peak')
            ax.legend(fontsize=7)
        ax.set_title(f"{a['name']}  (S1 {a['s1']:.2f})", fontsize=9)
        ax.set_xlabel(a['var'].unit, fontsize=8)
        if i % 4 == 0:
            ax.set_ylabel('mean yield (mm/night)', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Constrained variable search - {climate}\n'
                 f'model output, unvalidated against field data',
                 fontsize=13)
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def list_variables():
    print("=" * 78)
    print("Variable constraints")
    print("=" * 78)
    for kind in ('climate', 'siting', 'design', 'control'):
        print(f"\n[{kind}]")
        for v in VARIABLES:
            if v.kind != kind:
                continue
            print(f"  {v.name:<20} {v.low:>8.2f} .. {v.high:<8.2f} {v.unit:<10} "
                  f"{v.status}")
            print(f"      channel: {v.channel}")
            if v.note:
                print(f"      note   : {v.note}")
    print("\nClimate presets narrow the climate variables; --fix pins any variable.")


def parse_fix(pairs):
    pinned = {}
    for item in pairs or []:
        if '=' not in item:
            raise SystemExit(f"--fix expects name=value, got: {item}")
        name, value = item.split('=', 1)
        name = name.strip()
        if name not in VAR_BY_NAME:
            raise SystemExit(f"unknown variable: {name} "
                             f"(see --list-variables)")
        try:
            pinned[name] = float(value)
        except ValueError:
            raise SystemExit(f"--fix value must be a number, got: {value}")
        var = VAR_BY_NAME[name]
        if not (var.low <= pinned[name] <= var.high):
            print(f"  note: {name}={pinned[name]:g} is outside its declared "
                  f"range {var.low:g}..{var.high:g} {var.unit}")
    return pinned


def main():
    parser = argparse.ArgumentParser(
        description='Search a constrained variable space for the ranges that '
                    'improve dew yield, and rank variables by leverage.')
    parser.add_argument('--climate', default='semi_arid',
                        choices=list(CLIMATES.keys()))
    parser.add_argument('--samples', type=int, default=4000)
    parser.add_argument('--top-frac', type=float, default=0.10,
                        help='fraction of best outcomes defining "optimal"')
    parser.add_argument('--energy-budget', type=float, default=3.0,
                        help='max electrical W/m^2')
    parser.add_argument('--fix', action='append', metavar='NAME=VALUE',
                        help='pin a variable; repeatable')
    parser.add_argument('--allow-freeze', action='store_true',
                        help='keep samples whose surface drops below 0 C')
    parser.add_argument('--condensing-only', action='store_true',
                        help='analyse only samples that actually made water, '
                             'isolating design levers from whether dew forms')
    parser.add_argument('--list-variables', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', default='variable_search.png')
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()

    if args.list_variables:
        list_variables()
        return

    constraints = Constraints(max_electrical_w_m2=args.energy_budget,
                              allow_freeze=args.allow_freeze,
                              condensing_only=args.condensing_only,
                              pinned=parse_fix(args.fix))

    search = VariableSearch(climate=args.climate, constraints=constraints,
                            rng_seed=args.seed)
    columns = search.sample(args.samples)
    results = search.evaluate(columns, args.samples)

    feasible = results['feasible']
    if feasible.sum() >= 30:
        y_all = results['yield_mm'][feasible]
        cutoff = np.percentile(y_all, 100 * (1 - args.top_frac))
        top = feasible & (results['yield_mm'] >= cutoff)
        analyses = []
        for name in columns:
            a = analyse_variable(name, columns[name][feasible], y_all,
                                 columns[name][top], search.bounds,
                                 args.top_frac)
            if a is not None:
                analyses.append(a)
    else:
        analyses = []

    print_report(search, columns, results, analyses, args.samples, args.top_frac)

    if analyses and not args.no_plot:
        fig = plot_results(analyses, args.top_frac, args.climate)
        fig.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"Graph saved to: {args.output}")
        plt.close(fig)


if __name__ == '__main__':
    main()
