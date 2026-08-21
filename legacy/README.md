# Legacy

Superseded files, kept deliberately.

These are not backups and not dead weight. They are **precedence**: the record of
what this project claimed before, and the reasoning that produced those claims.
When a current claim is traced back — as in `docs/method-log.md` M-01 and M-03 —
this is where the trail leads. Delete these and the falsification record becomes
unverifiable assertion.

Everything here is frozen. Nothing in `simulations/`, `firmware/` or `docs/`
imports from this directory, and nothing here is maintained or expected to run.

## Contents

### `2025-original/` — pre-standardisation state

The repository as it stood before the 2026-03 structural cleanup, which deleted
or heavily rewrote these files. Filenames are flattened with `__` marking the
original directory (`firmware__main.py` was `firmware/main.py`).

| File | Was | Superseded by | Why kept |
|---|---|---|---|
| `README.md` | `README.md` | current `README.md` | Origin of the "0.034-0.14 mm/day" headline (M-01) |
| `firmware__02_crop_response.md` | `firmware/02_crop_response.py` | `docs/atmospheric-seed-theory.md` | **The important one.** 4,404 lines of research notes misfiled as a `.py`. Only surviving trace of the ion-coupling PDE model (M-01, M-08) |
| `firmware__main.py` | `firmware/main.py` | `firmware/esp32_basic/main.py` | Python wrapped in markdown fences; original ESP32 logger |
| `firmware__01_basic_dew_simulation.py` | `firmware/01_basic_dew_simulation.py` | `simulations/01_basic_dew.py` | Simulation misfiled under firmware, wrapped in markdown |
| `firmware__01_basic_dew_2simulation.py` | `firmware/01_basic_dew_2simulation.py` | `simulations/01_basic_dew.py` | Direct ancestor of the current dew model |
| `simulations__01_basic_dew_simulation.py` | `simulations/01_basic_dew_simulation.py` | `simulations/01_basic_dew.py` | Byte-identical duplicate of the `_2simulation` file |
| `BUILD-README.md` | `BUILD-README.md` | `docs/build-guide.md` | Archived 2026-08-15. Was a pure rename until the guide was restructured to lead with operating season and siting; it also lists the Peltier as a component, which the current guide tells you to remove (H8, Round 3) |
| `trailer-build.md` | `trailer-build.md` | `docs/trailer-build.md` | Archived 2026-08-15. Holds the withdrawn "Average: 95ml/night" figure (M-07/H4) and the original next-iteration list — frost heater, bigger panel — both of which existed to keep the Peltier running |
| `docs__README.md` | `docs/README.md` | `docs/README.md` | The original documentation index |

`Requirements.txt` is not copied here: it was a pure rename to `requirements.txt`
with 100% content match, so nothing was lost.

The last three rows were added on 2026-08-15. When this table was first written
those files were pure renames with nothing to preserve — which was true at the
time. Rewriting the live versions is what created something to preserve. That is
the rule working: the trigger for archiving is a claim changing, not a file
moving.

## What the legacy files establish

**The 0.034 / 0.14 mm/day figures have a source, and it is not this code.**
`firmware__02_crop_response.md` documents an ion-coupling PDE model with two
operating modes — natural gradient coupling at 0.034 mm/day and ~0 kWh/day, and
active ion injection at 0.14 mm/day and 5.8 kWh/day. That model was never
committed as code. Only its printed output survives, here. The current
`01_basic_dew.py` is an unrelated empirical scaling relation that inherited the
headline number without inheriting the physics. See M-01.

**The dew model's 0.02 coefficient was never derived.** It is identical in every
version above, back to the first commit. Its provenance is "it was there
already."

**Two 40-bit seed layouts, from two different models.** The theory notes describe
ion amplitude / altitude modulation / horizontal wavelength; the current
optimiser decodes temperature / pH / light amplification. Same bit count, no
stated correspondence. See M-08.

## Rules

1. **Nothing here is edited.** Frozen at the state it was superseded in. If a
   legacy file is wrong, that wrongness is the data.
2. **Nothing here is deleted**, including files whose claims were falsified —
   especially those.
3. **Nothing current imports from here.** If legacy code is worth running again,
   port it forward into `simulations/` and give it a method-log entry.
4. **Cite, don't restate.** Referencing a legacy number means citing the legacy
   file and its status in `docs/method-log.md`. Copying a number out of here into
   current docs is the exact failure M-01 records.

## Retiring something into legacy

1. Move it under a dated directory, flattening the path with `__`.
2. Add a row to the table above: what it was, what supersedes it, why it's kept.
3. If it carried a claim, open an entry in `docs/method-log.md` (or
   `docs/research-log.md`, whichever round you are working in) before the claim
   disappears from the live docs.
4. Point every remaining reference at the new location.
