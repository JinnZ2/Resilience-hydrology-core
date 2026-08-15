# Resilience Hydrology Core

Physics-based water harvesting using natural atmospheric gradients.

## The problem

During drought, conventional irrigation fails. Wells go dry, rivers stop
flowing, water becomes scarce. People and crops suffer.

## This approach

Instead of pumping or transporting water, amplify the natural process of dew
formation using temperature, pH, and light gradients that exist everywhere.
No pumps, no wells, no infrastructure.

## Status — read this before citing any number

This is open research in progress, and the honest summary is short:

- ✅ Simulations run and are internally consistent
- ✅ Prototype hardware built and logging (northern Minnesota, Nov 2025)
- ⚠️ **Models are not validated against field data.** No comparison between
  modelled and measured output has been performed in this repository.
- ⚠️ **The 3x amplification factor is an assumption, not a measurement.** It is
  hard-coded into the simulation, so the ON/OFF comparison illustrates that
  assumption rather than testing it.
- 🚧 Field testing in progress — one site, partial data

Every claim below is either a model output with the command that reproduces it,
or it is marked untested. What has been checked, what was falsified, and what
was revised as a result is recorded in
**[docs/research-log.md](docs/research-log.md)**.

## What the model produces

Running `simulations/01_basic_dew.py` across the four climate presets:

| Climate | System OFF | System ON |
|---|---|---|
| arid | 0.100 mm/day | 0.300 mm/day |
| semi_arid | 0.091 mm/day | 0.273 mm/day |
| mediterranean | 0.054 mm/day | 0.162 mm/day |
| tropical_dry | 0.080 mm/day | 0.240 mm/day |

**These are model outputs, not measurements**, and the ON column inherits the
assumed 3x factor. An earlier headline figure of "0.034–0.14 mm/day" was
withdrawn — it is not reproducible from any code here (research log, H1). The
original wording is preserved in
[legacy/docs/README_2025-12-07.md](legacy/docs/README_2025-12-07.md).

The one field measurement is 85 ml and 110 ml on two nights from the trailer
build ([docs/trailer-build.md](docs/trailer-build.md)). Comparing it to a model
has been attempted once (research log, H9) and came out **undetermined** — not
because the physics failed, but because the collector's area and tilt angle were
never written down, and the answer depends on both.

## What actually moves the yield

`simulations/04_variable_search.py` samples a constrained variable space against
a surface energy-balance model and ranks variables by how much they change the
outcome. Three results are worth knowing before you build anything:

- **Collector tilt is the largest design lever, and no build guide specifies an
  angle.** It has a genuine best range (roughly 19–53°) because steeper drains
  better but sees less cold sky.
- **Siting beats electronics.** Canopy openness and upwind soil/plant moisture
  rank above every hardware variable except tilt. Where you put the collector
  matters more than what you put in it.
- **Active cooling is nearly inert at the field build's power budget.** A 3 W/m²
  electrical budget buys about 6% of the radiative cooling the surface already
  does for free — worth 1.11x, not 3x. Reaching 3x would take roughly 19x the
  power the $45 build has.

Full numbers and method: [docs/research-log.md](docs/research-log.md), Round 2.
These are model results, not measurements — see the caveat above.

## Already built one? Start here

[`simulations/05_transition_paths.py`](simulations/05_transition_paths.py) takes
an existing collector and returns the cheapest ordered set of changes. Its
findings changed this project's build guidance:

- **Run it in the dew season.** At the one site this project has deployed to, in
  the month it deployed, the model puts 47% of nights below freezing and 4%
  making water. The same site in September: 0% frozen, 16% productive. The frost
  failure in the field log was the season, not the hardware.
- **The first stage costs −$3.** Removing the Peltier pays for the bracket, the
  foam, and the mulch, with change left over. No funding decision required.
- **Order beats the parts list.** Angling the collector gains +0.1 mL/night on
  its own and +10.9 once the free season and siting decisions are made. A guide
  that lists parts without that ordering sells upgrades that appear not to work.

Every recommendation is scored across a Monte Carlo varying both the weather and
this project's own assumed coefficients, so what survives does not depend on us
being right about numbers we invented.

## Quick start

```bash
pip install -r requirements.txt

python simulations/01_basic_dew.py                        # 7-day dew model, ON vs OFF
python simulations/01_basic_dew.py --climate arid --days 14
python simulations/02_crop_response.py                    # crop yield during drought
python simulations/03_seed_optimization.py                # seed search scaffold (see caveat)
python simulations/04_variable_search.py --condensing-only  # what actually moves yield
python simulations/05_transition_paths.py                   # cheapest changes to an existing build
```

Each writes a PNG to the working directory. See
[simulations/README.md](simulations/README.md) for details.

## Repository structure

```
simulations/     Python models (numpy / matplotlib / scipy)
firmware/        MicroPython for ESP32 sensor nodes
  esp32_basic/     temperature only
  esp32_validation/ adds surface temp, humidity, volume - use this one
docs/            Build guides, theory notes, research log
legacy/          Superseded originals, archived with dates — never deleted
```

## Three ways in

### 1. Understand the science

Start with [`simulations/01_basic_dew.py`](simulations/01_basic_dew.py) (runs in
seconds), then read [docs/research-log.md](docs/research-log.md) to see which of
its assumptions survive scrutiny and which do not. The theory behind the seed
approach is in
[docs/atmospheric-seed-theory.md](docs/atmospheric-seed-theory.md), condensed
from the full 2025-12-07 research session in
[legacy/notes/](legacy/notes/2025-12-07_seed-expansion-session.md).

### 2. Build hardware

[docs/build-guide.md](docs/build-guide.md) covers builds by budget.
[docs/trailer-build.md](docs/trailer-build.md) is a real $45 build with its
results *and its failures* — frost on night 4, dead battery on day 6. Firmware
and flashing steps are in [firmware/](firmware/README.md).

The most useful thing a builder can contribute right now is a paired
measurement: collector area, nightly volume, and logged temperature/humidity.
That is the missing piece that would let the model be checked against reality
(research log, O1).

The whole measurement kit is **$23 and about four hours**:
[`firmware/esp32_validation/`](firmware/esp32_validation/) logs collector
surface temperature, humidity, and nightly volume, and takes the collector area
and tilt angle as constants you write down once.

Those last two matter more than they sound. The trailer build reported 85 ml and
110 ml, but nobody recorded its area or angle — and the model reproduces those
volumes under a well-configured collector while producing almost nothing under a
poorly-configured one. Both are consistent with the notebook. A tape measure and
a protractor are the difference between a validated model and a stalled
project (research log, H9).

### 3. Deploy at scale

Not yet supported. `03_seed_optimization.py` currently returns a degenerate
answer — the minimum of its search range for every climate, plus two unused
random bytes — because its objective always prefers less amplification
(research log, H5). **Do not deploy seeds it publishes.** The scaffold is kept
because the search structure is sound; the cost model is what is missing.

For siting and configuration decisions, use
[`simulations/04_variable_search.py`](simulations/04_variable_search.py)
instead. It answers the same question without the degeneracy, reports ranges
rather than points, and flags when an "optimum" is really just a bound.

## Contributing

Most valuable, in order:

1. **Field measurements** that can be compared to the model (see O1 in the
   research log)
2. **Falsifications** — run something here, show it does not do what it claims,
   open an issue with the numbers
3. Plain-English explanations, use cases, translations

If you revise a claim, follow the convention this repository runs on: state the
new claim with its evidence, and archive the old wording in `legacy/` rather
than overwriting it. Precedence stays with whoever wrote it first —
see [legacy/README.md](legacy/README.md).

## License

MIT — do whatever you want with this, just don't blame us if it breaks.
Documentation under CC-BY-SA 4.0.
