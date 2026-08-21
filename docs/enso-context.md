# ENSO Context

What the El Niño–Southern Oscillation means for this project, what is sourced,
and what is guessed.

Compiled 2026-08-21. **ENSO state is perishable information** — anything below
about the current event is wrong within months. The reasoning is durable; the
numbers are not. Re-check before relying on them.

## Why a dew project cares

Round 3 of [`research-log.md`](research-log.md) found that *when* you run a
collector dominates every hardware change you can make to it. ENSO is the
largest interannual control on "when": it moves humidity, cloud cover and
temperature together, over whole regions, on a schedule forecast months ahead.

It also moves *where the drought is*, which is where this project claims to be
useful.

## Current state (August 2026)

An **El Niño Advisory** is in effect, with a very strong event forecast to peak
between **October 2026 and January 2027**.

| | Reported |
|---|---|
| Chance of a very strong event, NH fall/winter 2026–27 | >90% |
| Chance of exceeding every El Niño since 1950 | ~69% |
| NOAA RONI forecast, Oct–Dec median | +2.66 °C (middle half +2.37 to +2.95) |
| RONI, Dec–Feb median | +2.23 °C |
| El Niño persisting through Jan–Mar 2027 | ~100% |

> **A conflict in the sources, left visible.** Search summaries reported the
> August Niño-3.4 anomaly as both **+1.4 °C** and **+2.7 °C** (weekly value,
> week centred 12 August 2026). These cannot both be the monthly Niño-3.4
> anomaly. They are plausibly a monthly-vs-weekly or Niño-3.4-vs-RONI mix-up,
> but that is a guess and it is not resolved here. **Do not quote a Niño-3.4
> value from this page.** Resolving it needs the primary CPC discussion, which
> this environment could not reach — see Provenance below.

### Regional pattern

Documented teleconnection, robust across events even when magnitudes vary:

| El Niño raises drought risk | El Niño typically wetter |
|---|---|
| Australia (sharpest S. hemisphere signal) | Southern US |
| Southern Africa (maize belt) | Coastal Peru / Chile |
| South-East Asia / Indonesia | Horn of Africa |
| Central America | |
| Northern Amazon | |

A positive Indian Ocean Dipole is also forecast, which can amplify impacts
around the Indian Ocean basin.

**The overlap is the point.** The drought column is close to a list of the
places this project describes itself as serving.

## What the model says about it

[`../simulations/06_enso_response.py`](../simulations/06_enso_response.py) runs
the question rather than assuming an answer. El Niño drought does two opposite
things to radiative dew:

- **drier air** → less vapour available to condense → less dew
- **clearer skies** → stronger radiative cooling → more dew

Both are real, and the energy-balance model contains both, so it can say which
wins. It is the first question this project has had that *required* the
energy-balance model — the old scaling relation has no cloud term at all.

Result: **drying wins, by roughly 1.5–2x, in every drought region tested.**
Clearing is a genuine and substantial positive (+3.3 to +4.3 mL/night in the
model); it is simply outweighed (−6.3 to −7.1). Full numbers in
[`research-log.md`](research-log.md), H11.

The consequence is uncomfortable and worth stating plainly: **this system
produces least exactly where and when it is most needed.** That is a property of
the physics, not a design flaw to be engineered away. Any deployment plan that
assumes drought is when the collector earns its keep has the sign backwards.

## What is sourced and what is not

| Element | Status |
|---|---|
| El Niño advisory, forecast strength, peak timing | **Sourced** (see below) |
| Regional drought/wet pattern | **Sourced** — documented teleconnection |
| Sign of the RH and cloud shifts per region | **Sourced** via that pattern |
| **Magnitude** of RH, cloud, temperature shifts | **[ASSUMED]** — invented by this project |
| Dew response to those shifts | Model output, unvalidated |

Per-region pre-dawn RH and cloud anomalies for a strong El Niño were **not
obtainable** from the sources reachable here. The magnitudes in
`06_enso_response.py` are therefore guesses, chosen to be plausible and clearly
labelled. **Read the sign and the ranking of channels, not the absolute mL.**
The decomposition — drying beats clearing — is the robust part, because it
depends on the ratio of the two effects rather than on either magnitude.

→ Open question **O15**.

## Reading the sources critically

A checklist for anyone updating this page. Each item can change an anomaly
number by more than the effect being reported.

**1. Which baseline?** 1991–2020, 1951–1980 and pre-industrial give materially
different anomaly magnitudes for the same ocean. A later baseline absorbs prior
warming into "normal", so a *smaller* anomaly number may be a re-baselining
rather than a weaker event. This is a candidate resolution of the +1.4 / +2.7
conflict above: RONI (Relative Niño index) is explicitly re-baselined — it
subtracts tropical-mean SST change to remove the background warming trend — so
RONI and a raw Niño-3.4 anomaly are *different quantities* and should not be
expected to agree. Candidate, not confirmed; it still needs the primary source.

**2. Surface-only or full column?** A headline SST number excludes the
subsurface entirely. In August 2026 the equatorial Pacific carried anomalies
above **+8 °C at 50–150 m** between 150°W and 80°W, with some reports of
+11 °C — none of which appears in a surface figure.

**3. What is being averaged over?** A regional anomaly is diluted toward nothing
by global averaging. Ask what area the number covers before comparing two
numbers.

**4. "Not that bad" compared to what?** Against worst-case projections, plenty of
results look reassuring. Against a pre-industrial baseline, far fewer do. For
questions about whether a system's behaviour has changed in kind, the second
comparison is the relevant one.

### Displacement is not heat

The most important distinction for subsurface anomalies, and the easiest to
lose:

A **fixed-depth temperature anomaly** across a sharp vertical gradient is
largely a *displacement* signal. The equatorial thermocline spans roughly
8–12 °C over a few tens of metres. A downwelling Kelvin wave depresses it, so a
sensor at a fixed 100 m now samples water from above the thermocline where it
previously sampled from below. The anomaly is approximately
`vertical displacement × vertical gradient` — **a large fixed-depth anomaly can
occur with no additional heat in the column at all.** It is a rearrangement.

The budget-relevant quantity is **integrated upper-ocean heat content**
(0–300 m), which is displacement-invariant. Australia's Bureau of Meteorology
reported equatorial upper-ocean heat content in July 2026 as the highest for any
month in its record back to 1979 — that *is* a budget statement, and it is a
different claim from the +11 °C figure, resting on different evidence.

The two questions separate cleanly:

| Question | Diagnostic |
|---|---|
| How displaced is the thermocline? | Z20, depth of the 20 °C isotherm |
| Has the heat budget changed? | integrated OHC 0–300 m |
| Is mixing structurally suppressed? | stratification / buoyancy frequency N², as a trend rather than an event anomaly |

If OHC stays elevated after Z20 relaxes, the budget changed. If both relax
together, it was wave dynamics. That is the test that distinguishes "structural"
from "temporary", and this project is not in a position to run it.

**Provenance warning on the +11 °C figure specifically**: the >8 °C value and
the BoM heat-content record come from better-attested reporting; the +11.1 °C
figure traces to low-reliability secondary outlets. By this page's own
criterion 1, that number should not be quoted without the primary source.

## Fog is not dew

There is real published work on ENSO and fog:

- El Niño **intensified** fog formation in the Namib Desert (Li et al., *Earth's
  Future*, 2025)
- ENSO 3+4 SST above +1 °C in summer raised Atacama fog-water yield and
  explained 79% of summer fog variability
- Coastal California saw **less** fog and more humidity during El Niño

These describe **advection fog**: marine air driven onshore over cold water,
where sea-surface temperature sets the outcome. This project models **radiative
dew**: a surface cooling by longwave loss to the sky until it falls below the
dew point of the air already above it. Different driver, different mechanism,
and the sign need not agree — as the California result shows it does not even
agree between fog sites.

**These studies are cited as context and deliberately not used as evidence for
the dew results.** Carrying a number across from the model that produced it to
one that did not is exactly the failure recorded as M-01 in
[`method-log.md`](method-log.md). The fog literature is a reason to take the
question seriously, not a source of an answer.

If anything, the divergence is an argument for measuring: two fog sites disagree
about the sign of the ENSO effect, and nobody has measured a dew site at all.

## Provenance

Compiled 2026-08-21 from web search result summaries. **Primary sources could
not be retrieved directly**: this environment's network policy blocked
`cpc.ncep.noaa.gov`, `gfdl.noaa.gov` and `iri.columbia.edu`, so the figures above
are as reported by search summarisation, not as read from the source documents.
That is a weaker chain of custody than this project normally accepts, and it is
why the Niño-3.4 conflict above could not be resolved.

**Anyone with unrestricted network access should verify these against the
primary sources and correct this page.** The canonical references:

- NOAA CPC, ENSO Diagnostic Discussion —
  https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso_advisory/ensodisc.shtml
- NOAA GFDL, August 2026 El Niño Predictions (SPEAR) —
  https://www.gfdl.noaa.gov/prediction/august-2026-el-nino-predictions/
- IRI, ENSO Forecast — https://iri.columbia.edu/our-expertise/climate/forecasts/enso/current/
- WMO, "Strong El Niño expected to intensify" —
  https://wmo.int/news/media-centre/strong-el-nino-expected-intensify
- EC Joint Research Centre, "Potentially historic El Niño to come" —
  https://joint-research-centre.ec.europa.eu/jrc-news-and-updates/potentially-historic-el-nino-come-analysis-shows-humanitarian-toll-2026-06-15_en
- Yale Climate Connections, "This could be the strongest El Niño on record" —
  https://yaleclimateconnections.org/2026/07/this-could-be-the-strongest-el-nino-on-record/
- Li et al. 2025, "El Niño Intensified Fog Formation in the Namib Desert",
  *Earth's Future* — https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024EF005725
- "ENSO Influence on Coastal Fog-Water Yield in the Atacama Desert, Chile",
  *Aerosol and Air Quality Research* — https://aaqr.org/articles/aaqr-17-01-fog-0022
- NOAA drought.gov, El Niño and the Southern Plains —
  https://www.drought.gov/news/el-nino-horizon-can-warm-phase-end-six-years-drought-southern-plains-us-2026-03-11
