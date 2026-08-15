# Legacy

Superseded material, kept intact. **Nothing here is deleted, and precedence still carries.**

## Why this folder exists

This project moves by the scientific loop: hypothesize → run → compare → revise
the claim → find the unknowns → rerun. That loop only works if the *earlier*
version of a claim survives. A claim you quietly overwrite cannot be checked
against the result that falsified it, and the person who wrote it first loses
their standing on it.

So when a file is replaced, corrected, or condensed, the original lands here with
its date. The working tree carries the current best claim. This folder carries
the record of what was claimed before, and when.

Rules for this folder:

- **Archive, do not edit.** Files here are frozen at the state they were
  archived in — including their typos, broken numbers, and stray markdown. If
  something here is wrong, that is the point; the correction goes in the working
  tree and the reason goes in [`docs/research-log.md`](../docs/research-log.md).
- **Priority is by original date**, not by the date something was moved here.
  The dates below are the dates the material was first committed to this
  repository.
- **Never delete from here.** Superseding a file means adding a newer file
  elsewhere, not removing the older one.

## Contents

### `notes/2025-12-07_seed-expansion-session.md` — 4,404 lines

First commit: 2025-12-07 (as `firmware/02_crop_response.py` — misfiled under a
Python name; it is prose and code notes, not a module).

The full original research session on atmospheric seed expansion: the
orbital↔atmospheric analogy, the ion/supersaturation "decompressor" chain, the
40-bit seed layout, resilience testing, adaptive-strategy comparison, node and
network design, deployment protocols, economic analysis, manufacturing BOMs, and
a full firmware draft.

This is the source document for [`docs/atmospheric-seed-theory.md`](../docs/atmospheric-seed-theory.md),
which is a 53-line condensation of it. The condensation dropped the large
majority of the material. **Priority for everything in the theory doc dates to
this file, 2025-12-07.** Several ideas here have never been implemented or
tested and are not represented in the working tree at all — see the open
questions in the research log.

### `simulations/01_basic_dew_simulation.py` — 114 lines

First commit: 2025-12-07. Class `SimpleDewSimulator`.

The original dew model. Its physics core —
`natural_dew = RH * delta_T * 0.02` and a hard-coded `amplification = 3.0` —
was carried forward *unchanged* into the current
[`simulations/01_basic_dew.py`](../simulations/01_basic_dew.py) as
`DewSimulator`. The rewrite changed packaging (CLI arguments, climate presets,
a returned figure), not physics. A near-identical duplicate of this file also
existed at `firmware/01_basic_dew_2simulation.py`, differing only by one blank
line; it is not reproduced separately.

### `simulations/01_basic_dew_simulation_annotated.md` — 209 lines

First commit: 2025-12-07 (as `firmware/01_basic_dew_simulation.py`, Python code
wrapped inside markdown fences, which is why it would not run).

A longer variant of the same model. It carries material the current simulation
does **not** have and which was lost in the merge:

- a `lat` (latitude) parameter on the constructor
- a per-night humidity/moisture-availability treatment
- inline commentary on where the constants came from

If the dew model is ever rebuilt from real observations, start here rather than
from the current file.

### `firmware/esp32_basic_main_annotated.md` — 182 lines

First commit: 2025-12-07 (as `firmware/main.py`, again wrapped in markdown
fences).

The original `TemperatureLogger`. Includes SD-card logging via `SoftSPI`/
`SDCard` and wiring notes (4.7 kΩ pull-ups on the OneWire data lines) that the
current [`firmware/esp32_basic/main.py`](../firmware/esp32_basic/main.py) drops.
The SD-card path is unfinished, not rejected.

### `docs/README_2025-12-07.md` — 148 lines

The original project README, verbatim, as first written on 2025-12-07.

It is archived because the working README has since been rewritten, and several
of its claims were revised after being checked against the code — in particular
the headline output range, "physics models validated", and "hardware is proven".
**The original wording is preserved here so the revision is legible as a
revision.** What changed and why is recorded in
[`docs/research-log.md`](../docs/research-log.md), entries H1–H4.

It also documents structure that was planned but never built (`/theory`,
`/hardware`, `hardware/trailer_dew_collector/`, `CONTRIBUTING.md`,
`quick-start.md`, `FAQ.md`) — useful as a statement of intent, which is why the
dead paths were removed from the working README rather than from here.

### `docs/docs-README_2025-12-07.md` — 29 lines

The original documentation index, verbatim. Superseded by
[`docs/README.md`](../docs/README.md).

## Recovering anything from here

These files are also in git history and can be recovered directly:

```bash
# list the tree as it stood before the 2026-03-23 restructure
git ls-tree -r --name-only 1acdd38^

# print any file at that point
git show 1acdd38^:firmware/02_crop_response.py
```

Archiving into this folder is belt-and-braces: git history is authoritative, but
history is not visible to someone reading the repository, and an unread record
protects no one's priority.
