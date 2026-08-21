# Alternative Systems for Severe Drought

What to build when the air is too dry for dew.

## Why this document exists

[H11](research-log.md) falsified this project's founding premise. Radiative dew
needs humid air; severe drought is dry air; modelled yield falls 68–85% in
exactly the regions a strong El Niño puts at risk. Supply and need move in
opposite directions.

The useful response is not to abandon the work, and not to defend the premise.
It is to ask what mechanism *does* work in dry air, and to compare candidates on
the same physical basis. That is
[`../simulations/07_alternative_systems.py`](../simulations/07_alternative_systems.py).

## The structural point

**Passive dew is free but gated.** It requires a surface to reach the dew point,
and radiative cooling delivers 3–9 K of depression. Past that limit the yield is
not small — it is exactly zero. Money, tilt, coatings and cleverness do not
cross it, because the mechanism is switched off.

**Sorption is not gated.** A hygroscopic sorbent pulls vapour from air at 11–20%
RH, where condensation is thermodynamically impossible. It costs 1–3 kWh per
litre, as *heat* rather than work.

Free-but-gated versus costly-but-always-available is the entire decision.

## Where the dew wall sits

Modelled, at three air temperatures (mL/m² per night):

| RH | 15 °C | 22 °C | 32 °C |
|---|---|---|---|
| ≤50% | 0 | 0 | 0 |
| 60% | 1 | 0 | 0 |
| 70% | 8 | 1 | 0 |
| 80% | 59 | 17 | 0 |
| 90% | 179 | 130 | 6 |

**Dew is a cool-humid-night mechanism.** The wall sits near 70% RH at 15 °C and
near 90% at 32 °C, because warm air needs a larger absolute depression *and* a
warm humid sky radiates more heat back. Hot dry air is the worst case, and it is
exactly what severe drought looks like.

This also refines H11: the anti-correlation is worst in hot drought. Cool coastal
and highland sites with humid nights remain genuinely good for dew.

## The comparison at 32 °C, 25% RH

Severe dry-air drought:

| Mechanism | Feasible | Yield mL/m²/day | Energy |
|---|---|---|---|
| Passive radiative dew | **no** | 0 | free |
| Active condensation | yes | 317 | 3.41 kWh/L electrical |
| Sorption + solar heat | yes | 240–720 | 1–3 kWh/L thermal |

At 25% RH a condenser must chill **734 kg of air per kg of water**. That ratio,
not any equipment defect, is why cooling fails in dry air: you pay sensible heat
for all the air and collect water from a trace of it. It applies to a Peltier
exactly as it applies to a compressor.

**The thermodynamic floor is 0.054 kWh/L.** Nothing here is limited by physics —
real devices sit 20–60× above the floor. That is an engineering gap, not a law,
which is the one genuinely encouraging fact in this document.

## The inversion worth noticing

Sorption needs **heat**, and drought regions are sunny. The clear skies that
cannot save dew — because clear skies supply no vapour — do supply sun.

| Source | kWh/m²/day |
|---|---|
| The $45 build's electrical budget | 0.036 |
| 1 m² solar PV at 18% | 1.08 |
| 1 m² solar thermal at 12% | 0.72 |

A solar thermal collector delivers roughly **20× the energy the current build
has**, in the form the mechanism actually wants. H11's anti-correlation reverses
for this mechanism: the conditions that kill dew are conditions sorption is
suited to.

## A discrepancy left open

Two sourced numbers do not reconcile, and this document does not pick a winner:

| Route | Yield |
|---|---|
| Energy: 1 m² solar thermal (0.72 kWh/day) ÷ 1–3 kWh/L | 240–720 mL/m²/day |
| Device: published sorbent performance at low humidity | 5,500 mL/m²/day |

About a **10× gap**. Plausible explanations: the published devices use a solar
aperture larger than their sorbent area; they run several adsorb/desorb cycles
per day; or the 1–3 kWh/L figure does not describe those particular systems.
None of this could be settled from the sources reachable here.

The screen reports the **smaller, energy-limited** figure, because that is the
one derived from first principles rather than taken on trust. **If the device
figure is the right one, sorption is far better than this comparison suggests** —
the conclusion that sorption beats dew in dry air survives either way, which is
why the gap does not block the finding. → **O17**

## What this does not say

This is a feasibility screen. It does not design a sorbent bed, size a
collector, or cost a build.

The sorption numbers are **other groups' measurements of other groups'
hardware**. That makes them stronger evidence than anything else in this
repository — every other number here is an unvalidated model — and simultaneously
not transferable to a build nobody in this project has made. Published device
performance is not a promise about your device.

Nothing in `07_alternative_systems.py` has been built or measured by this
project.

## One calibration, and what it cost

The condensation model is anchored to a published measurement: a
dehumidifier-based harvester consuming **1.02 kWh/L at 30 °C / 62% RH**. This
model's raw thermal load at that condition is 1.762 kWh/L, implying an
end-to-end **COP of 1.73** — well under any nameplate figure, because it absorbs
fan work, cycling, and heat-exchanger approach.

This is the only calibration against measured data anywhere in this repository.
An earlier version of the script assumed COP 2.5, which flattered condensation
by about 1.4×. The measurement corrected it downward. That is what calibration
is for, and it is worth noting that the correction went against the more
interesting-looking result.

## Sources

Sorption performance and energy:

- Advanced MOF and composite salt-polymer sorbents: 5.5 kg/m²/day at low
  humidity, 16.9 kg/m²/day at higher humidity — https://www.patsnap.com/resources/blog/articles/mof-water-harvesting-technology-landscape-2026/
- MOFs programmable to capture at 10–20% RH; hydrogels 0.7–6.7 g/g over
  ~20–90% RH — https://www.sciencedirect.com/science/article/abs/pii/S2213343725037431
- LiCl-polyacrylamide hydrogel field-tested in the Atacama, producing water at
  **11% RH** — https://techxplore.com/news/2025-05-atmospheric-harvesting-optimization-hygroscopic-hydrogel.html
- MIT hydrogel device: 1.7 L/m²/day at 50% RH — same source
- Thermal energy for sorption cycles, 1–3 kWh/L; dehumidifier route 1.02 kWh/L
  at 30 °C/62% RH — https://www.sciencedirect.com/science/article/pii/S2213343724020918
- Global potential of continuous sorption-based AWH — https://pmc.ncbi.nlm.nih.gov/articles/PMC11964679/
- Breaking the thermal limit of AWH, *Nature Reviews Clean Technology* (2026) —
  https://www.nature.com/articles/s44359-026-00154-5
- Vibrational regeneration at 0.09 kWh/kg, desorption in under 2 minutes —
  https://www.sciencedirect.com/science/article/pii/S2590174526004113
- Liter-scale AWH for dry climates on low-temperature solar heat —
  https://www.sciencedirect.com/science/article/abs/pii/S0360544222011987
- Solar-driven system, 12.3% thermal efficiency; costed systems at $0.087/L and
  $0.13/L — https://aiche.onlinelibrary.wiley.com/doi/10.1002/ep.14458?af=R
- Xanthan-gum hygroscopic hydrogel, *Scientific Reports* (2025) —
  https://www.nature.com/articles/s41598-025-23971-3

Retrieved 2026-08-21 via search summaries. As with
[`enso-context.md`](enso-context.md), this environment could not fetch primary
PDFs, so these figures have not been read from the source documents. **Verify
before building anything.** → **O17**

## What would change the answer

1. **A measured RH at a real site.** Every claim here turns on humidity, and this
   project still has no humidity sensor in the field (O1). The dew/sorption
   decision cannot be made without it.
2. **Sorbent cost and lifetime.** The comparison above is energy-only. A sorbent
   that costs more than the water it saves, or degrades in a season, changes the
   conclusion and is not modelled here.
3. **Whether the site is hot-dry or cool-humid.** These are different projects.
   The repository should say which one it serves rather than implying both.
