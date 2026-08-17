# Resilience Hydrology Core

Physics-based water harvesting using natural atmospheric gradients.
Open research — models, firmware, and a record of what has and hasn't held up.

## The problem

During drought, conventional irrigation fails. Wells go dry, rivers stop flowing.
People and crops suffer.

## The approach

Instead of pumping or transporting water, amplify the natural process of dew
formation using temperature, pH, and light gradients that exist everywhere.

## Where this actually stands

Read this before quoting any number from this repository.

| | Status |
|---|---|
| Simulations | Run and produce output |
| Physics models | **Not validated.** Core coefficients underived |
| Prototype hardware | Built, collected water (Nov 2025, northern MN) |
| Field vs. model | **Never compared** — the field log lacks collector area |
| Seed optimisation | **Degenerate.** Objective does not select a seed |
| Energy-balance model | Added round 2; structurally sound, still unvalidated |
| Transition analysis | First stage costs −$3 and works today |

Modelled output, `simulations/01_basic_dew.py`:

- **0.054–0.100 mm/day** unamplified, across the four climate presets
- **0.162–0.300 mm/day** with the assumed 3× system gain

The 3× gain is an assumption, not a measurement. Earlier versions of this README
advertised "0.034–0.14 mm/day"; that figure came from a different model that is
no longer in this repository and has been withdrawn — see
[`docs/method-log.md`](docs/method-log.md) M-01.

**Honest summary of scale:** at modelled output levels this supplements water
supply by a fraction of a percent of crop demand. It is not drought mitigation.
Closing a drought needs roughly 3 mm/day — about 10× the most optimistic figure
this model produces (M-05).

## What actually moves the yield

`simulations/04_variable_search.py` replaces the bare scaling relation with a
surface energy balance — radiative loss, convection, latent heat, conduction —
and searches a constrained variable space for the *ranges* that improve yield.
Unlike `01_basic_dew.py` it checks whether the surface actually reaches the dew
point, which turns out to matter a great deal:

- **At the climate presets this repo ships, dew is thermodynamically
  impossible.** Radiative cooling delivers 3–9 K of depression; those humidities
  need 12–18 K. `01_basic_dew.py` reports water forming anyway, because it never
  checks (research-log H7).
- **Collector tilt is the largest lever a builder controls** — a real best range
  of roughly 19–53°, and no build guide specified an angle.
- **Siting beats electronics.** Canopy openness and upwind soil moisture outrank
  every hardware variable except tilt.
- **Active cooling is inert at the field power budget** — 6% of the radiative
  cooling the surface already does for free, worth 1.11x. Reaching 3× would take
  ~19× the power the $45 build has (H8).

## Already built one? Start here

`simulations/05_transition_paths.py` takes an existing collector and returns the
cheapest ordered set of changes, with costs, hours, and who has to act:

- **Run it in the dew season.** At the one site this project deployed to, in the
  month it deployed, 47% of nights freeze and 4% make water. September at the
  same site: 0% and 16%. The night-4 frost failure was the season, not the
  hardware.
- **The first stage costs −$3.** Removing the Peltier pays for the bracket, the
  foam, and the mulch, with change left over.
- **Order beats the parts list.** Angling the collector gains +0.1 mL/night alone
  and +10.9 after the free season and siting decisions. A guide that lists parts
  without that ordering sells upgrades that appear not to work.

Every recommendation is scored across a Monte Carlo varying both the weather and
this project's own assumed coefficients, so what survives doesn't depend on us
being right about numbers we invented.

## Repository layout

```
simulations/       Python models (numpy/matplotlib/scipy)
  01-03            original models, with their known defects documented
  04, 05           energy-balance search and transition analysis
firmware/          MicroPython for ESP32 sensor nodes
  esp32_basic/     temperature only
  esp32_validation/ adds surface temp, humidity, volume — use this one
docs/              Build guides, theory notes, and two falsification logs
legacy/            Superseded files, frozen — the precedence record
```

## Quick start

```bash
pip install -r requirements.txt

python simulations/01_basic_dew.py --climate arid --days 14
python simulations/02_crop_response.py --water 0.27
python simulations/03_seed_optimization.py
python simulations/04_variable_search.py --condensing-only
python simulations/05_transition_paths.py
```

See [`simulations/README.md`](simulations/README.md) for what each model does and
where each one is known to be wrong.

## Three ways in

**Understand the science** — start with `simulations/01_basic_dew.py`, then read
`docs/method-log.md`. The second file is the more useful of the two: it says
which parts of the first are load-bearing and which are placeholders.

**Build hardware** — `docs/build-guide.md` for builds by budget,
`docs/trailer-build.md` for a real build with its failures recorded, and
`firmware/esp32_basic/` for the sensor node. Roughly $45 for the basic build.
Build in the order the guide gives: season, siting, tilt. The order is the
finding, not a formality.

**Make your build testable** — `firmware/esp32_validation/` adds collector
surface temperature, humidity, and nightly volume for about $23. The trailer
build reported 85 ml and 110 ml, but nobody recorded its area or tilt — and the
energy-balance model reproduces those volumes under a well-configured collector
while producing almost nothing under a poorly-configured one. Both readings fit
the notebook. A tape measure and a protractor are the difference between a
validated model and a stalled project (H9).

**Contribute a measurement** — the single most valuable thing anyone can add.
M-07 in the method log lists exactly what a comparable field run needs: collector
area, a paired unpowered control, per-night conditions, and every night including
the failures. One careful week of that closes the largest gap in this project.

## How this project handles claims

Hypothesize → run → compare → if falsified, **edit the claim, not the model** →
list what you didn't know → rerun.

Falsified claims are kept, not deleted. There are two records:
[`docs/method-log.md`](docs/method-log.md) (M-nn) came first and holds
precedence, and [`docs/research-log.md`](docs/research-log.md) (H-nn, O-nn)
continues into rounds 2–3. They were written independently and reached the same
round-1 findings by different routes — an accidental replication, kept as one.
Between them they record: every claim, what happened when it was tested, what it was changed to,
and what that revealed we didn't know. Superseded files go to `legacy/` frozen,
because a falsification you can't trace back to its source is just an assertion.

The recurring failure mode this repo has already hit four times: a number
outliving the model that produced it, getting re-attached to different code, and
being repeated until it reads as established. If you add a number, make it
traceable to a command someone can run today or a measurement someone recorded
with its conditions. If it's neither, label it a hypothesis and give it an ID.

## Contributing

Most useful, in order:

1. Field measurements taken to the M-07 protocol
2. Derivation or a source for `DEW_COEFF` (M-01). `simulations/04_variable_search.py`
   is a first attempt at the replacement condensation model this asks for — it is
   structured from physics rather than fitted, but its own transfer coefficients
   are still assumed (O10) and it has never met field data either
3. Hardware confirmation of the GPIO15 SD chip-select fix (M-06)
4. Plain-English explanations, translations, and use cases

Report failures as readily as successes. `docs/trailer-build.md` records icing
and a dead battery, and is more useful for it.

## License

MIT for code, CC-BY-SA 4.0 for documentation. Use it, modify it, share it — just
don't blame us if it breaks.
