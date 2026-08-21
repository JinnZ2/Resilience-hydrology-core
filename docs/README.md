# Documentation

## Contents

- **[method-log.md](method-log.md)** - **Start here.** Claims, tests, falsifications, and open unknowns
- **[research-log.md](research-log.md)** - Second falsification record, covering
  rounds 2-3: constrained variable search, transition analysis, and the first
  model-vs-measurement comparison
- **[enso-context.md](enso-context.md)** - What El Niño does to dew yield, and
  why the drought premise needs qualifying. Time-sensitive; re-check the state.
- **[alternative-systems.md](alternative-systems.md)** - What to build when the
  air is too dry for dew. Sorption, active condensation, and where each becomes
  possible.
- **[build-guide.md](build-guide.md)** - **Which build?** Two paths, chosen by
  your site's pre-dawn humidity
- **[build-dew.md](build-dew.md)** - Build A: passive dew collector, cool humid sites
- **[build-sorbent.md](build-sorbent.md)** - Build B: sorbent harvester, hot dry sites
- **[trailer-build.md](trailer-build.md)** - Real-world trailer dew collector with results
- **[atmospheric-seed-theory.md](atmospheric-seed-theory.md)** - Research notes on seed expansion physics

The method log is listed first on purpose. It says which numbers elsewhere in
these docs are supported and which are hypotheses, and several published figures
have already been withdrawn there. Reading a claim in this directory without
checking its method-log status is how those figures survived as long as they did.

Two logs, deliberately. `method-log.md` (M-nn) and `research-log.md` (H-nn, O-nn)
were written independently and reached the same round-1 findings by different
routes — a natural replication, and worth keeping as one. `method-log.md` came
first and holds precedence; `research-log.md` continues past it into rounds 2-3.
Where they overlap, the M-entry is the original record. See
[log-format-comparison.md](log-format-comparison.md) for what each format is
good at, tested rather than argued — `python tools/log_audit.py` reruns that
test.

Superseded documents live frozen in [`../legacy/`](../legacy/README.md), which is
where the method log's provenance trails lead.

## Who is this for?

- Farmers facing drought
- Off-grid residents needing water
- Disaster relief organizations
- Researchers studying water scarcity
- Makers wanting to build something useful

## Contributing

We need:
- Field measurements that can be compared against the models - collector area,
  tilt angle, nightly volume, and logged temperature/humidity together. This is
  the biggest gap in the project (method-log M-07, research-log O1/H9)
- Falsifications: run something here, show it doesn't do what it claims
- Plain-English explanations of the science
- Use cases and field reports
- Translations to other languages

## License

CC-BY-SA 4.0 - share and adapt freely
