#!/usr/bin/env python3
"""
Transition paths: the smallest, most leveraged steps from the deployed build to
a better one.

THE PROBLEM THIS SOLVES

04_variable_search.py says what matters. It does not say what to DO about the
collector already sitting in a field. Those are different questions, because a
deployed unit has sunk costs, an owner, a budget, and a limited appetite for
disruption. "Tilt matters" is not an instruction. "Spend $3 and half an hour
adding a bracket, this weekend, and you can undo it" is.

So this file takes the current build as it actually exists, a catalogue of
possible modifications with their real costs, and returns:

  1. NO-REGRET MOVES — changes that pay off no matter how the open questions
     resolve, ranked by leverage per dollar
  2. A STAGED PLAN — what to do at $0, then $10, then $25, then $50 per unit,
     with each stage re-evaluated after the previous one is applied so that
     interactions are handled rather than assumed away
  3. WHO MUST ACT — every step tagged builder / site / funder / standards,
     because a transition that needs a funding decision and one that needs a
     screwdriver are not the same kind of step
  4. WHAT NOT TO DO — modifications whose cost is not repaid, stated explicitly,
     including one the project has already spent money on

DECIDING WITH AN UNVALIDATED MODEL

The model underneath this has never been checked against a field measurement,
and several of its coefficients are numbers this project invented
(docs/research-log.md, O10). Ranking modifications by a single predicted yield
would launder that uncertainty into false confidence.

Instead every modification is evaluated across a Monte Carlo that varies BOTH
the weather AND our own assumed coefficients. What gets reported is not "this
gains 0.04 L/night" but "this improved things in 96% of draws, with a 10th
percentile of +0.01". A recommendation that survives our own uncertainty is
worth acting on. One that does not is worth measuring first.

That distinction is the whole point: it separates moves you can make today from
moves that need O1 closed first.

INFORMATION VS PERFORMANCE

Some modifications produce no water at all — a humidity sensor, a tape measure
taken to the collector. They are scored separately, by which open question they
close, because a project whose largest problem is that nothing has been measured
should be able to see that spending on knowledge outranks spending on hardware.

Usage:
    python 05_transition_paths.py
    python 05_transition_paths.py --site field_nov_mn --samples 3000
    python 05_transition_paths.py --list-mods
    python 05_transition_paths.py --budget 0 10 25 50 100
    python 05_transition_paths.py --site arid_summer --no-plot
"""

import argparse
import importlib.util
import os
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt


def _load_variable_search():
    """
    Import 04_variable_search.py.

    The repository numbers its simulation files, so the module name starts with
    a digit and a plain import statement is a syntax error. This is the cost of
    the naming convention; importlib pays it.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '04_variable_search.py')
    spec = importlib.util.spec_from_file_location('variable_search', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vs = _load_variable_search()
DewEnergyBalance = vs.DewEnergyBalance


# ---------------------------------------------------------------------------
# Where the units actually are
# ---------------------------------------------------------------------------

# Weather windows per site. field_nov_mn reproduces the conditions of the one
# real deployment this project has (docs/trailer-build.md): northern Minnesota,
# November, long nights, near-freezing, humid, frequently overcast. The frost
# failure on night 4 of that log is a modelled outcome here, not an anecdote.
SITES = {
    'field_nov_mn': {
        'label': 'northern Minnesota, November (the actual deployment)',
        't_air_night': (269.0, 280.0), 'rh': (0.70, 0.95),
        'wind_speed': (0.0, 5.0), 'cloud_cover': (0.10, 0.85),
        'night_hours': (13.0, 15.0),
    },
    'field_sep_mn': {
        'label': 'northern Minnesota, September (same site, earlier season)',
        't_air_night': (277.0, 288.0), 'rh': (0.75, 0.97),
        'wind_speed': (0.0, 4.5), 'cloud_cover': (0.10, 0.70),
        'night_hours': (11.0, 13.0),
    },
    'semi_arid_summer': {
        'label': 'semi-arid summer, pre-dawn humidity',
        't_air_night': (285.0, 295.0), 'rh': (0.45, 0.90),
        'wind_speed': (0.0, 5.0), 'cloud_cover': (0.0, 0.50),
        'night_hours': (9.0, 12.0),
    },
    'arid_summer': {
        'label': 'arid summer, pre-dawn humidity',
        't_air_night': (283.0, 293.0), 'rh': (0.35, 0.80),
        'wind_speed': (0.0, 5.0), 'cloud_cover': (0.0, 0.30),
        'night_hours': (10.0, 13.0),
    },
}


# The trailer build as it was actually deployed. Where a value was never
# recorded, the honest entry is the unimproved default — an unrecorded tilt is
# an uncontrolled tilt, and flat is what a collector does if nobody angles it.
BASELINE = {
    'sky_view_factor': 0.75,       # parked beside a trailer
    'local_vapor_boost': 0.0,
    'surface_emissivity': 0.90,    # generic painted metal
    'tilt_deg': 0.0,               # never specified anywhere - O9
    'insulation_r': 0.05,          # bare mount, no insulation
    'electrical_w_m2': 3.0,        # Peltier running on the full budget
    'cop_cooling': 0.50,
    'area_m2': 0.25,               # estimated, never measured - O1
}

BASELINE_NOTES = {
    'tilt_deg': 'never recorded; flat is what happens by default (O9)',
    'area_m2': 'estimated from the parts list, never measured (O1)',
    'electrical_w_m2': 'the Peltier, at the full energy budget (H8)',
    'insulation_r': 'bare mount - the night-4 frost failure',
}


# ---------------------------------------------------------------------------
# What can be changed
# ---------------------------------------------------------------------------

# actor: builder    - someone with a screwdriver at the unit
#        site       - a land/siting decision, may need a landowner
#        funder     - needs money released before anything happens
#        standards  - needs a documented spec so every future build inherits it
#
# kind:  performance - changes how much water comes out
#        information - produces no water; closes an open question
#        removal     - takes something away, usually recovering money

@dataclass
class Modification:
    mod_id: str
    name: str
    cost_usd: float
    labor_hours: float
    actor: str
    kind: str
    sets: dict = field(default_factory=dict)       # variable -> new value
    scales: dict = field(default_factory=dict)     # variable -> multiplier
    requires: tuple = ()
    group: str = ''                                # mutually exclusive options
    reversible: bool = True
    prevents_freeze: bool = False
    site_shift: str = ''                           # operate in a different window
    closes: str = ''                               # open question it answers
    evidence: str = ''
    note: str = ''

    def apply(self, config):
        out = dict(config)
        out.update(self.sets)
        for key, factor in self.scales.items():
            out[key] = out[key] * factor
        return out


MODIFICATIONS = [
    # ---- free, today, reversible -----------------------------------------
    Modification(
        'record_area', 'Measure and record the collector area', 0.0, 0.1,
        'builder', 'information',
        closes='half of O1',
        evidence='research log O1',
        note='A tape measure. Without this number every volume ever collected '
             'is uncomparable to every model, so nothing else can be checked.'),
    Modification(
        'record_tilt', 'Measure and record the current tilt angle', 0.0, 0.1,
        'builder', 'information',
        closes='O9',
        evidence='research log O9',
        note='The largest design lever is currently an uncontrolled variable '
             'in every field result.'),
    Modification(
        'shift_season', 'Operate in the dew season, not the frost season',
        0.0, 0.0, 'site', 'performance',
        site_shift='field_sep_mn',
        evidence='research log Round 3',
        note='Costs nothing at all. A site that freezes on half its nights is '
             'not a dew site on those nights, and no hardware fixes that. '
             'Choosing WHEN to run is free and outranks everything you can '
             'bolt on.'),
    Modification(
        'resite_open_sky', 'Move the collector clear of the trailer and canopy',
        0.0, 0.5, 'site', 'performance',
        sets={'sky_view_factor': 0.95},
        evidence='research log Round 2 (S1 0.078)',
        note='Costs nothing but a decision about where the thing sits.'),
    Modification(
        'tilt_30', 'Angle the collector to 30 degrees', 3.0, 0.5,
        'builder', 'performance',
        sets={'tilt_deg': 30.0}, group='tilt',
        evidence='research log Round 2 (S1 0.152, best range 19-53 deg)',
        note='Scrap bracket. The single largest design lever in the model.'),
    Modification(
        'tilt_45', 'Angle the collector to 45 degrees', 3.0, 0.5,
        'builder', 'performance',
        sets={'tilt_deg': 45.0}, group='tilt',
        evidence='research log Round 2',
        note='Alternative to tilt_30; better drainage, less sky view.'),

    # ---- cheap hardware ---------------------------------------------------
    Modification(
        'insulate_mount', 'Foam block between collector and mount', 4.0, 0.5,
        'builder', 'performance',
        sets={'insulation_r': 1.0},
        evidence='research log Round 2; docs/trailer-build.md night-4 failure',
        note='Stops the mount conducting heat back into the cold surface.'),
    Modification(
        'surface_thermometer', 'Third DS18B20 bonded to the collector plate',
        5.0, 0.5, 'builder', 'information',
        closes='the energy-balance validation gap',
        evidence='research log O1, H8',
        note='The model PREDICTS surface temperature. Measuring it is the '
             'cheapest way to find out whether the physics here is right.'),
    Modification(
        'humidity_sensor', 'SHT31 humidity sensor and firmware update', 6.0, 1.5,
        'builder', 'information',
        closes='the largest part of O1',
        evidence='research log Round 2, measurement priority #1',
        note='Humidity is the biggest driver of whether dew forms at all, and '
             'the current build cannot measure it.'),
    Modification(
        'high_emissivity_coat', 'High-emissivity coating on the collector',
        8.0, 1.0, 'builder', 'performance',
        sets={'surface_emissivity': 0.96},
        evidence='research log Round 2 (S1 0.010)',
        note='Small but cheap and permanent.'),
    Modification(
        'mulch_upwind', 'Wet mulch bed upwind of the collector', 5.0, 2.0,
        'site', 'performance',
        sets={'local_vapor_boost': 0.06},
        evidence='research log Round 2 (S1 0.086) - but see O10',
        note='[ASSUMED CHANNEL] The vapour-boost mechanism is unmeasured and '
             'its size was invented by this project. Treat as a trial, not a '
             'recommendation.'),
    Modification(
        'volume_gauge', 'Tipping-bucket gauge and pulse counter', 12.0, 2.0,
        'builder', 'information',
        closes='the rest of O1',
        evidence='research log O1, O5',
        note='Turns "85 ml, then 110 ml, then the notebook stops" into a '
             'nightly series.'),

    # ---- larger spend -----------------------------------------------------
    Modification(
        'double_area', 'Double the collector area', 25.0, 3.0,
        'funder', 'performance',
        scales={'area_m2': 2.0},
        evidence='geometry, not the model',
        note='Does nothing to mm/day; doubles litres. The only modification '
             'here whose effect is arithmetic rather than physics.'),
    Modification(
        'bigger_panel', 'Larger panel and battery (raises the energy budget)',
        18.0, 1.0, 'funder', 'performance',
        sets={'electrical_w_m2': 8.0},
        evidence='research log H8',
        note='Buys more power for the Peltier. H8 says this is the wrong '
             'thing to buy; included so the report can show that.'),
    Modification(
        'frost_heater', 'Small resistive heater for frost nights', 10.0, 2.0,
        'builder', 'performance',
        requires=('bigger_panel',), prevents_freeze=True,
        evidence='docs/trailer-build.md, night 4',
        note='Addresses the observed failure directly: frost meant the night '
             'was lost entirely.'),

    # ---- taking things away ----------------------------------------------
    Modification(
        'remove_peltier', 'Remove the Peltier cooler entirely', -15.0, 1.0,
        'builder', 'removal',
        sets={'electrical_w_m2': 0.0},
        evidence='research log H8',
        note='Recovers the part cost and removes the largest power draw. H8 '
             'found it worth 1.11x at this budget. Negative cost: this '
             'modification pays you.'),
]

MOD_BY_ID = {m.mod_id: m for m in MODIFICATIONS}


# ---------------------------------------------------------------------------
# Evaluation under uncertainty
# ---------------------------------------------------------------------------

# How far the [ASSUMED] coefficients are allowed to wander. These bounds are
# themselves guesses — the honest position is that we do not know these numbers,
# so a recommendation should survive a wide range of them.  [ASSUMED]
COEFF_UNCERTAINTY = {
    'h_c_still': (1.5, 4.0),
    'h_c_wind': (2.0, 4.5),
    'eff_max': (0.80, 1.00),
    'tilt_char': (10.0, 30.0),
}


class TransitionEvaluator:
    """Evaluates configurations across weather and our own uncertainty."""

    WEATHER_KEYS = ('t_air_night', 'rh', 'wind_speed', 'cloud_cover',
                    'night_hours')

    def __init__(self, site='field_nov_mn', samples=2000, rng_seed=0):
        self.site = site
        self.window = SITES[site]
        self.n = samples
        rng = np.random.default_rng(rng_seed)

        # Quantiles, not realised values. Every configuration is evaluated on
        # the same quantile draws, so two configurations are always compared on
        # the same nights — and a modification that changes WHICH weather
        # applies (operating season) stays paired too, because the same
        # quantile realises the 30th-percentile night of either window.
        self.u_weather = rng.random((samples, len(self.WEATHER_KEYS)))
        self.u_coeffs = rng.random((samples, len(COEFF_UNCERTAINTY)))
        self.coeff_keys = list(COEFF_UNCERTAINTY)

    def _weather(self, i, window):
        return {key: window[key][0] + self.u_weather[i, j] *
                (window[key][1] - window[key][0])
                for j, key in enumerate(self.WEATHER_KEYS)}

    def _coeffs(self, i):
        out = {}
        for j, key in enumerate(self.coeff_keys):
            lo, hi = COEFF_UNCERTAINTY[key]
            out[key] = lo + self.u_coeffs[i, j] * (hi - lo)
        return out

    def evaluate(self, config, frost_protected=False, window=None):
        """
        Millilitres per night across all draws, one entry per draw.

        mL because that is the unit the field log is written in (85 ml, 110 ml
        on the two recorded nights), so a model number and a measured number can
        be put side by side without arithmetic.
        """
        window = window or self.window
        area = config['area_m2']
        model_keys = [k for k in config if k != 'area_m2']
        out = np.zeros(self.n)

        for i in range(self.n):
            v = {k: config[k] for k in model_keys}
            v.update(self._weather(i, window))
            result = DewEnergyBalance(coeffs=self._coeffs(i), **v).solve()
            if result['frozen'] and not frost_protected:
                # The field log is explicit: frost meant the night was lost.
                out[i] = 0.0
            else:
                out[i] = result['yield_mm'] * area * 1000.0   # litres -> mL
        return out

    def diagnostics(self, config, window=None):
        """Frost and condensation rates — why a site behaves as it does."""
        window = window or self.window
        model_keys = [k for k in config if k != 'area_m2']
        frozen = condensing = 0
        for i in range(self.n):
            v = {k: config[k] for k in model_keys}
            v.update(self._weather(i, window))
            result = DewEnergyBalance(coeffs=self._coeffs(i), **v).solve()
            frozen += bool(result['frozen'])
            condensing += bool(result['yield_mm'] > 0 and not result['frozen'])
        return {'frozen': frozen / self.n, 'condensing': condensing / self.n}


def summarise(base, cand):
    """
    Robustness summary for a change, in mL/night.

    The headline is the MEAN, not the median. In these climates most nights
    produce nothing at all, so the median night is zero for almost every
    configuration and would rank everything as identical. What a water supply
    cares about is the total over a season, which is the mean per night.

    P(better) is computed only over nights where SOMETHING happened under
    either configuration. On a night when neither the old nor the new build
    makes water, the modification has not failed — it was never in play. Scoring
    those nights as "did not help" would drag every verdict toward AVOID purely
    because the site is quiet, which is a statement about the site, not the
    modification.
    """
    delta = cand - base
    active = np.maximum(base, cand) > 1e-9
    n_active = int(active.sum())
    if n_active:
        p_improve = float((delta[active] > 1e-9).mean())
        p_harm = float((delta[active] < -1e-9).mean())
    else:
        p_improve = p_harm = 0.0
    return {
        'mean': float(np.mean(delta)),
        'median': float(np.median(delta)),
        'p10': float(np.percentile(delta, 10)),
        'p90': float(np.percentile(delta, 90)),
        'p_improve': p_improve,
        'p_harm': p_harm,
        'active_nights': n_active / len(delta),
    }


# A yield change smaller than this is treated as negligible when judging a
# modification that recovers money. The field log is written in tens of mL, so
# a fraction of a mL per night is not a reason to keep hardware.  [ASSUMED]
NEGLIGIBLE_ML = 1.0


def verdict(stats, mod):
    """Turn a distribution into an instruction."""
    if mod.kind == 'information':
        return 'MEASURE FIRST'
    if mod.cost_usd < 0:
        # Pays you back. The question is whether it costs meaningful water.
        return ('RECOVERS COST' if stats['mean'] > -NEGLIGIBLE_ML
                else 'TRADE-OFF')
    if mod.cost_usd <= 0 and stats['p_harm'] < 0.10:
        return 'NO-REGRET'
    if stats['p_improve'] >= 0.90 and mod.cost_usd <= 10:
        return 'NO-REGRET'
    if stats['p_improve'] >= 0.75:
        return 'GOOD BET'
    if stats['p_improve'] >= 0.40:
        return 'UNCERTAIN'
    return 'AVOID'


# ---------------------------------------------------------------------------
# Path building
# ---------------------------------------------------------------------------

def available(mod, applied, config):
    """Can this modification be applied next?"""
    if mod.mod_id in applied:
        return False
    if any(req not in applied for req in mod.requires):
        return False
    if mod.group and any(MOD_BY_ID[m].group == mod.group for m in applied):
        return False
    # A modification that would not change the configuration is not available.
    if (mod.kind == 'performance' and mod.apply(config) == config
            and not mod.site_shift and not mod.prevents_freeze):
        return False
    return True


def build_staged_plan(evaluator, budgets, verbose_log=None):
    """
    Greedy staged transition. After each step the configuration is updated and
    everything is re-scored against the new state, so interactions are handled
    by construction rather than assumed to be additive.
    """
    config = dict(BASELINE)
    applied, frost_protected = [], False
    window = None
    spent = 0.0
    baseline_yield = evaluator.evaluate(config, frost_protected)
    current = baseline_yield
    stages = []

    for budget in budgets:
        stage_steps = []
        while True:
            best = None
            for mod in MODIFICATIONS:
                if not available(mod, applied, config):
                    continue
                if spent + mod.cost_usd > budget:
                    continue
                if mod.kind == 'information':
                    continue   # scored separately, never competes on yield
                candidate = mod.apply(config)
                protected = frost_protected or mod.prevents_freeze
                cand_window = SITES[mod.site_shift] if mod.site_shift else window
                cand_yield = evaluator.evaluate(candidate, protected,
                                                cand_window)
                stats = summarise(current, cand_yield)
                if stats['mean'] <= 1e-9 and mod.cost_usd >= 0:
                    continue
                # Leverage per dollar; free and paying modifications sort first.
                if mod.cost_usd <= 0:
                    score = float('inf') if stats['mean'] > 0 else 1e6
                else:
                    score = stats['mean'] / mod.cost_usd
                if best is None or score > best[0]:
                    best = (score, mod, candidate, stats, protected, cand_window)
            if best is None:
                break
            _, mod, candidate, stats, protected, cand_window = best
            config = candidate
            frost_protected = protected
            window = cand_window
            applied.append(mod.mod_id)
            spent += mod.cost_usd
            current = evaluator.evaluate(config, frost_protected, window)
            stage_steps.append((mod, stats, float(np.mean(current))))
        stages.append({
            'budget': budget, 'steps': stage_steps, 'spent': spent,
            'yield': float(np.mean(current)),
            'config': dict(config),
        })

    return {
        'baseline_yield': float(np.mean(baseline_yield)),
        'stages': stages, 'applied': applied, 'final_config': config,
        'final_window': window, 'frost_protected': frost_protected,
    }


def score_singles(evaluator):
    """Every modification scored on its own against the untouched baseline."""
    config = dict(BASELINE)
    base = evaluator.evaluate(config)   # array of draws, not a summary
    rows = []
    for mod in MODIFICATIONS:
        if mod.kind == 'information':
            rows.append({'mod': mod, 'stats': None, 'per_dollar': None})
            continue
        if mod.requires:
            # Score it with its prerequisites in place, and charge for them.
            staged = dict(config)
            extra_cost = 0.0
            protected = False
            for req in mod.requires:
                staged = MOD_BY_ID[req].apply(staged)
                extra_cost += MOD_BY_ID[req].cost_usd
                protected = protected or MOD_BY_ID[req].prevents_freeze
            candidate = mod.apply(staged)
            protected = protected or mod.prevents_freeze
            total_cost = mod.cost_usd + extra_cost
        else:
            candidate = mod.apply(config)
            protected = mod.prevents_freeze
            total_cost = mod.cost_usd

        window = SITES[mod.site_shift] if mod.site_shift else None
        stats = summarise(base, evaluator.evaluate(candidate, protected, window))
        per_dollar = (stats['mean'] / total_cost) if total_cost > 0 else None
        rows.append({'mod': mod, 'stats': stats, 'per_dollar': per_dollar,
                     'total_cost': total_cost})
    return float(np.mean(base)), rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(evaluator, base_yield, singles, plan, budgets):
    site = SITES[evaluator.site]
    print("=" * 74)
    print("Transition Paths — from the deployed build to a better one")
    print("=" * 74)
    print(f"Site      : {evaluator.site} — {site['label']}")
    print(f"Draws     : {evaluator.n} (weather x our own coefficient uncertainty)")
    print(f"Baseline  : {base_yield:.1f} mL/night (mean over all nights,\n            including the ones that produce nothing)")
    print()
    print("Baseline is the trailer build as actually deployed:")
    for key, note in BASELINE_NOTES.items():
        print(f"  {key:<18} {BASELINE[key]:<8g} {note}")

    diag = evaluator.diagnostics(BASELINE)
    print(f"\nSite behaviour at the baseline configuration:")
    print(f"  nights the surface freezes  : {diag['frozen']:.0%}")
    print(f"  nights that make water      : {diag['condensing']:.0%}")
    if diag['frozen'] > 0.25:
        alt = 'field_sep_mn'
        if alt in SITES and evaluator.site != alt:
            alt_diag = evaluator.diagnostics(BASELINE, SITES[alt])
            print(f"\n  ! This site spends {diag['frozen']:.0%} of its nights below")
            print(f"    freezing. A frozen night is a lost night — that is the")
            print(f"    night-4 failure in the field log, not bad luck.")
            print(f"    The same site in September: {alt_diag['frozen']:.0%} frozen,")
            print(f"    {alt_diag['condensing']:.0%} making water. WHEN you run this")
            print(f"    is a free variable and it dominates what you bolt onto it.")

    if base_yield <= 1e-6:
        print("\n  Note: the baseline collects essentially nothing in these")
        print("  conditions. Improvements below are therefore large in relative")
        print("  terms and small in absolute ones. Read the litres, not the")
        print("  percentages.")

    # ---- single modifications ----
    perf = [r for r in singles if r['stats'] is not None]
    perf.sort(key=lambda r: (r['per_dollar'] is None, -(r['per_dollar'] or 0)))

    print("\n" + "-" * 74)
    print("1. SINGLE MODIFICATIONS — each applied alone to the current build")
    print("-" * 74)
    print(f"{'modification':<26} {'$':>6} {'hrs':>4} {'d mL/night':>11} "
          f"{'P(better)':>9}  verdict")
    for r in perf:
        m, s = r['mod'], r['stats']
        print(f"{m.name[:26]:<26} {m.cost_usd:>6.0f} {m.labor_hours:>4.1f} "
              f"{s['mean']:>+11.1f} {s['p_improve']:>8.0%}  {verdict(s, m)}")
    print("\nP(better) is the fraction of draws — across weather AND across our")
    print("own uncertainty about the assumed coefficients — in which the change")
    print("helped. A high number means the recommendation survives not knowing.")

    # ---- no-regret ----
    print("\n" + "-" * 74)
    print("2. NO-REGRET MOVES — do these without waiting for anything")
    print("-" * 74)
    nr = [r for r in perf if verdict(r['stats'], r['mod']) == 'NO-REGRET']
    nr.sort(key=lambda r: -(r['per_dollar'] or float('inf')))
    for r in nr:
        m, s = r['mod'], r['stats']
        cost = 'free' if m.cost_usd == 0 else (
            f"pays ${-m.cost_usd:.0f}" if m.cost_usd < 0 else f"${m.cost_usd:.0f}")
        print(f"  {m.name}")
        print(f"      {cost}, {m.labor_hours:.1f} h, {m.actor}, "
              f"{'reversible' if m.reversible else 'permanent'} — "
              f"{s['mean']:+.1f} mL/night, better in "
              f"{s['p_improve']:.0%} of draws")
        print(f"      {m.note}")
    if not nr:
        print("  None. Every performance change here is either uncertain or")
        print("  costs more than its evidence supports.")

    # ---- information ----
    print("\n" + "-" * 74)
    print("3. INFORMATION MOVES — buy knowledge, not water")
    print("-" * 74)
    print("These produce no water. They are listed separately because a project")
    print("whose central problem is that nothing has been measured should not")
    print("rank them against yield — they are what makes yield claims checkable.")
    info = [r['mod'] for r in singles if r['mod'].kind == 'information']
    info.sort(key=lambda m: (m.cost_usd, m.labor_hours))
    for m in info:
        cost = 'free' if m.cost_usd == 0 else f"${m.cost_usd:.0f}"
        print(f"  {m.name:<44} {cost:>6}, {m.labor_hours:.1f} h, {m.actor}")
        print(f"      closes: {m.closes}")
        print(f"      {m.note}")
    total_info = sum(m.cost_usd for m in info)
    total_hours = sum(m.labor_hours for m in info)
    print(f"\n  All of them together: ${total_info:.0f} and "
          f"{total_hours:.1f} hours.")
    print(f"  That is the entire cost of making this project's claims testable.")

    # ---- staged plan ----
    print("\n" + "-" * 74)
    print("4. STAGED TRANSITION PLAN")
    print("-" * 74)
    print("Each stage is re-scored after the previous one is applied, so these")
    print("are cumulative effects, not a sum of independent estimates.\n")
    prev_yield = plan['baseline_yield']
    for stage in plan['stages']:
        if not stage['steps']:
            print(f"  Up to ${stage['budget']:.0f} — nothing further worth doing")
            continue
        print(f"  Up to ${stage['budget']:.0f} per unit:")
        for mod, stats, running in stage['steps']:
            print(f"    - {mod.name}")
            print(f"        ${mod.cost_usd:.0f}, {mod.labor_hours:.1f} h, "
                  f"{mod.actor}  |  {stats['mean']:+.1f} mL/night "
                  f"({stats['p_improve']:.0%} of draws)")
        gain = stage['yield'] - prev_yield
        print(f"    => spent ${stage['spent']:.0f}, "
              f"yield {stage['yield']:.1f} mL/night ({gain:+.1f} this stage)")
        prev_yield = stage['yield']
        print()

    total_gain = plan['stages'][-1]['yield'] - plan['baseline_yield']
    final_spend = plan['stages'][-1]['spent']
    ratio = ''
    if plan['baseline_yield'] > 1.0:
        ratio = (f", {plan['stages'][-1]['yield'] / plan['baseline_yield']:.1f}x")
    print(f"  Overall: ${final_spend:.0f} per unit, "
          f"{plan['baseline_yield']:.1f} -> {plan['stages'][-1]['yield']:.1f} "
          f"mL/night ({total_gain:+.1f}{ratio})")
    if plan['baseline_yield'] <= 1.0:
        print("  No ratio is quoted because the baseline is essentially zero;")
        print("  a multiple of nothing is not a meaningful number.")

    assumed = [m for m in (MOD_BY_ID[i] for i in plan['applied'])
               if 'ASSUMED' in m.note or 'ASSUMED' in m.evidence]
    if assumed:
        print("\n  ! This plan leans on channels this project invented:")
        for m in assumed:
            print(f"      {m.name} — {m.evidence}")
        print("    Their size is a guess (O10). Treat those steps as trials to")
        print("    be measured, not as recommendations to be trusted.")

    # ---- ordering effects ----
    single_by_id = {r['mod'].mod_id: r['stats'] for r in perf}
    in_plan = [(mod, stats) for stage in plan['stages']
               for mod, stats, _ in stage['steps']]
    amplified = [(mod, single_by_id[mod.mod_id]['mean'], stats['mean'])
                 for mod, stats in in_plan
                 if mod.mod_id in single_by_id
                 and stats['mean'] > 4 * max(single_by_id[mod.mod_id]['mean'], 0.05)
                 and stats['mean'] > 1.0]
    if amplified:
        print("\n" + "-" * 74)
        print("5. ORDER MATTERS MORE THAN THE PARTS")
        print("-" * 74)
        print("These modifications are nearly worthless on their own and become")
        print("the largest wins once the free decisions above them are made:\n")
        print(f"{'modification':<34} {'alone':>9} {'in sequence':>13}")
        for mod, alone, seq in amplified:
            print(f"{mod.name[:34]:<34} {alone:>+8.1f} {seq:>+12.1f}  mL/night")
        print("\nThe reason is that a frozen or bone-dry night cannot be improved")
        print("by a better bracket. Fix WHEN and WHERE first — both free — and")
        print("the cheap hardware suddenly has something to work with. A build")
        print("guide that lists these parts without that ordering is selling")
        print("upgrades that will appear not to work.")

    # ---- by actor ----
    print("\n" + "-" * 74)
    print("6. WHO HAS TO ACT")
    print("-" * 74)
    print("A transition stalls where nobody owns the next step. Grouped by who")
    print("must move, including the information moves:\n")
    chosen = [MOD_BY_ID[i] for i in plan['applied']] + info
    for actor in ('builder', 'site', 'funder', 'standards'):
        group = [m for m in chosen if m.actor == actor]
        if not group:
            continue
        cost = sum(m.cost_usd for m in group)
        hours = sum(m.labor_hours for m in group)
        print(f"  {actor.upper():<10} ${cost:>6.0f}  {hours:>4.1f} h")
        for m in group:
            print(f"      - {m.name}")
    print("\n  Everything above needs a screwdriver or a decision about where a")
    print("  collector sits. Nothing needs a funding round, which is the point:")
    print("  the leverage is in changes an owner can make unilaterally.")

    # ---- what not to do ----
    print("\n" + "-" * 74)
    print("7. WHAT NOT TO DO")
    print("-" * 74)
    avoid = [r for r in perf
             if verdict(r['stats'], r['mod']) in ('AVOID', 'UNCERTAIN')
             or (r['per_dollar'] is not None and r['mod'].cost_usd >= 10
                 and r['stats']['mean'] <= 0)]
    seen = set()
    for r in avoid:
        m, s = r['mod'], r['stats']
        if m.mod_id in seen:
            continue
        seen.add(m.mod_id)
        print(f"  {m.name} — ${r['total_cost']:.0f}, {s['mean']:+.1f} "
              f"mL/night, helps in {s['p_improve']:.0%} of draws")
        print(f"      {m.note}")
    if not avoid:
        print("  Nothing in the catalogue scored badly enough to warn about.")

    print("\n" + "-" * 74)
    print("CAVEAT")
    print("-" * 74)
    print("  Costs and labour hours are estimates, not quotes. The yields come")
    print("  from a model that has never been compared against a field")
    print("  measurement — which is exactly why the information moves in")
    print("  section 3 come first. Any performance number here could move once")
    print("  O1 is closed; the ranking is built to be robust to that, not immune")
    print("  to it. See docs/research-log.md.")
    print()


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_transition(base_yield, singles, plan, site):
    perf = [r for r in singles if r['stats'] is not None]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    # (a) leverage per dollar
    ax = axes[0]
    rows = [r for r in perf if r['total_cost'] > 0]
    rows.sort(key=lambda r: r['per_dollar'] or 0)
    ax.barh([r['mod'].name[:24] for r in rows],
            [(r['per_dollar'] or 0) for r in rows],
            color=['#2e7d32' if (r['per_dollar'] or 0) > 0 else '#c62828'
                   for r in rows], alpha=0.85)
    ax.set_xlabel('mL/night gained per dollar spent')
    ax.set_title('Leverage per dollar')
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3, axis='x')

    # (b) robustness: does it survive our own uncertainty?
    ax = axes[1]
    for r in perf:
        s = r['stats']
        color = ('#2e7d32' if s['p_improve'] >= 0.9 else
                 '#f9a825' if s['p_improve'] >= 0.75 else '#c62828')
        ax.scatter(max(r['total_cost'], 0.5), s['p_improve'], s=90,
                   color=color, alpha=0.85, zorder=3)
        ax.annotate(r['mod'].mod_id, (max(r['total_cost'], 0.5), s['p_improve']),
                    fontsize=7, xytext=(4, 4), textcoords='offset points')
    ax.axhline(0.9, color='#2e7d32', linestyle='--', linewidth=1, alpha=0.6)
    ax.set_xscale('log')
    ax.set_xlabel('cost, $ (log scale)')
    ax.set_ylabel('fraction of draws where it helped')
    ax.set_title('Does it survive our own uncertainty?')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # (c) staged staircase
    ax = axes[2]
    # Real cumulative spend, negatives included: a step that pays you moves the
    # line left, which is the point.
    xs, ys, labels = [0.0], [base_yield], ['baseline']
    for stage in plan['stages']:
        for mod, _stats, running in stage['steps']:
            xs.append(xs[-1] + mod.cost_usd)
            ys.append(running)
            labels.append(mod.mod_id)
    ax.step(xs, ys, where='post', color='#1565c0', linewidth=2)
    ax.scatter(xs, ys, color='#1565c0', zorder=3, s=30)
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(lab, (x, y), fontsize=7, xytext=(4, -10),
                    textcoords='offset points')
    ax.axvline(0, color='#666666', linewidth=1, linestyle=':')
    ax.set_xlabel('cumulative spend per unit, $  (negative = it paid you)')
    ax.set_ylabel('mean yield, mL/night')
    ax.set_title('Staged transition')
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Transition paths — {site}\n'
                 f'model output, unvalidated against field data', fontsize=13)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def list_mods():
    print("=" * 74)
    print("Modification catalogue")
    print("=" * 74)
    for kind in ('information', 'performance', 'removal'):
        print(f"\n[{kind}]")
        for m in MODIFICATIONS:
            if m.kind != kind:
                continue
            cost = 'free' if m.cost_usd == 0 else (
                f"pays ${-m.cost_usd:.0f}" if m.cost_usd < 0
                else f"${m.cost_usd:.0f}")
            print(f"  {m.mod_id:<22} {cost:>10}  {m.labor_hours:>4.1f} h  "
                  f"{m.actor}")
            print(f"      {m.name}")
            if m.requires:
                print(f"      requires: {', '.join(m.requires)}")
            if m.closes:
                print(f"      closes  : {m.closes}")
            print(f"      evidence: {m.evidence}")
            print(f"      {m.note}")


def main():
    parser = argparse.ArgumentParser(
        description='Find the most leveraged small modifications from the '
                    'deployed build toward a better one.')
    parser.add_argument('--site', default='field_nov_mn',
                        choices=list(SITES.keys()))
    parser.add_argument('--samples', type=int, default=2000)
    parser.add_argument('--budget', type=float, nargs='+',
                        default=[0.0, 10.0, 25.0, 50.0],
                        help='cumulative budget tranches per unit')
    parser.add_argument('--list-mods', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', default='transition_paths.png')
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()

    if args.list_mods:
        list_mods()
        return

    budgets = sorted(args.budget)
    evaluator = TransitionEvaluator(site=args.site, samples=args.samples,
                                    rng_seed=args.seed)
    base_yield, singles = score_singles(evaluator)
    plan = build_staged_plan(evaluator, budgets)
    print_report(evaluator, base_yield, singles, plan, budgets)

    if not args.no_plot:
        fig = plot_transition(base_yield, singles, plan, args.site)
        fig.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"Graph saved to: {args.output}")
        plt.close(fig)


if __name__ == '__main__':
    main()
