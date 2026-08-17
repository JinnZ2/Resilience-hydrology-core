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

## Repository layout

```
simulations/     Python models (numpy/matplotlib/scipy)
firmware/        MicroPython for ESP32 sensor nodes
docs/            Build guides, theory notes, and the method log
legacy/          Superseded files, frozen — the precedence record
```

## Quick start

```bash
pip install -r requirements.txt

python simulations/01_basic_dew.py --climate arid --days 14
python simulations/02_crop_response.py --water 0.27
python simulations/03_seed_optimization.py
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

**Contribute a measurement** — the single most valuable thing anyone can add.
M-07 in the method log lists exactly what a comparable field run needs: collector
area, a paired unpowered control, per-night conditions, and every night including
the failures. One careful week of that closes the largest gap in this project.

## How this project handles claims

Hypothesize → run → compare → if falsified, **edit the claim, not the model** →
list what you didn't know → rerun.

Falsified claims are kept, not deleted. `docs/method-log.md` is the running
record: every claim, what happened when it was tested, what it was changed to,
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
2. Derivation or a source for `DEW_COEFF` (M-01) — or a replacement condensation
   model that isn't a bare scaling relation
3. Hardware confirmation of the GPIO15 SD chip-select fix (M-06)
4. Plain-English explanations, translations, and use cases

Report failures as readily as successes. `docs/trailer-build.md` records icing
and a dead battery, and is more useful for it.

## License

MIT for code, CC-BY-SA 4.0 for documentation. Use it, modify it, share it — just
don't blame us if it breaks.
