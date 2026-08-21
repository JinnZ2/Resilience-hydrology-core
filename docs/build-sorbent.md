# Build B — Sorbent Harvester (hot dry sites)

A salt-and-sunlight water harvester for air too dry for dew. No pump, no
compressor, no electricity.

**This is one of two build paths.** Use it where nights are hot or dry —
below about 60% pre-dawn RH, where [`build-dew.md`](build-dew.md) produces
literally nothing. See [`build-guide.md`](build-guide.md) to choose.

> **Status: not yet built by this project.** Everything below is derived from
> published material properties and this project's own cycle model. The
> chemistry is well established and widely reproduced; *this particular build*
> has never been made or measured by anyone here. Treat it as a starting bill of
> materials to iterate from. If you build one, the measurements you take will be
> the most valuable data this repository has ever held.

## How it works

```
  night   bed open to the air; the salt pulls water vapour out of it   (free)
  day     bed sealed under glazing; sun drives the vapour off; it
          condenses on a cooler surface and runs into a collector      (sun)
```

That is the whole machine. The energy is sunlight, as heat, which is what the
mechanism wants — and what dry regions have in abundance. The clear skies that
make dew impossible are the same skies that power this.

## The salt choice is the design decision

A hygroscopic salt only takes up water in bulk above its **deliquescence
humidity (DRH)**. Below it, uptake falls off a cliff — the same kind of wall dew
has, just further into dry air.

Modelled uptake, g of water per g of dry composite:

| Pre-dawn RH | CaCl₂ (DRH 30%) | LiCl (DRH 11%) | Silica gel |
|---|---|---|---|
| 15% | 0.04 | 0.70 | 0.04 |
| 20% | 0.08 | 0.88 | 0.07 |
| 30% | 0.91 | 1.17 | 0.15 |
| 40% | 1.08 | 1.40 | 0.22 |
| 60% | 1.33 | 1.75 | 0.31 |
| 80% | 1.50 | 2.00 | 0.35 |

**Above ~30% RH → use calcium chloride.** Cheap (~$2/kg), an FDA GRAS food
additive, and no lithium question. This is the default.

**Below ~30% RH → CaCl₂ collapses.** At 15% RH it needs about 44 kg of composite
for 1 L/day instead of 1.5 kg. Only LiCl (or a MOF) still works — and that
brings the safety problem below.

**Silica gel** has no deliquescence cliff and no brine at all, which makes it the
safest way to learn the cycle. Its capacity is low and its isotherm is S-shaped,
so it is genuinely weak in dry air — 0.04 g/g at 15% RH. Good first build, poor
desert solution.

## Sizing

```bash
python simulations/08_sorbent_sizing.py --rh 0.35 --target-l 1.0
python simulations/08_sorbent_sizing.py --compare-salts --rh 0.20
python simulations/08_sorbent_sizing.py --sweep
```

For **1 L/day at 35% pre-dawn RH with CaCl₂**, the model gives roughly:

| Item | Quantity |
|---|---|
| Composite (35% salt by dry mass) | 1.5 kg |
| Bed area | 0.19 m² |
| Regeneration heat | 0.77 kWh/day |
| Solar aperture | 0.44 m² |
| Rough cost | ~$8 |

Those numbers assume a 65% cycle efficiency, which is this project's guess. Plan
for worse on a first attempt.

## Parts

| Part | Notes |
|---|---|
| Calcium chloride | **Food-grade.** Sold as a food additive, cheese-making salt, or pure de-icer. Not road salt with anti-caking additives. |
| Porous substrate | Vermiculite, perlite, coarse sand, or sawdust. Cheap and inert; holds the brine and keeps surface area high. |
| Shallow trays | Non-metallic if possible — CaCl₂ brine corrodes steel and aluminium. Food-grade plastic is ideal. |
| Glazing | Glass or polycarbonate sheet over a shallow box. |
| Absorber | Anything black under the bed. |
| Condenser | A sloped metal or plastic sheet, **shaded and outside the hot box**, draining into a channel. |

## Building it

1. **Make the composite.** Dissolve CaCl₂ in warm water (roughly 40 g per 100 mL
   — dissolution is strongly exothermic, so add salt to water slowly). Soak the
   substrate, then dry it in the sun until it is free-flowing rather than wet.
   Target about a third salt by dry mass.
2. **Load the trays shallow.** 1–3 cm deep. Air only reaches the top few
   millimetres overnight, so a deep bed holds salt that never sees moist air.
   Bed area matters more than bed volume.
3. **Expose at night.** Trays open to moving air, protected from rain and dew
   drip. A mesh cover keeps insects and debris out.
4. **Seal and heat by day.** Cover with the glazed box. Internal temperature
   needs roughly 60–80 °C to drive the vapour off.
5. **Condense outside the hot box.** This is the part first builds get wrong.
   The condenser must be *cooler* than the bed — shaded, exposed to ambient air,
   ideally with airflow across its back. Vapour that never finds a cold surface
   is just vented.
6. **Drain to a covered container.**

## Safety — read before drinking anything

**Salt carryover is the failure mode of this design.** The vapour that leaves
the bed is pure water, but liquid brine can creep along surfaces and reach the
condensate. Deliquescent salts are very good at creeping.

1. **Never drink water that tastes salty.** That is brine carryover. Discard it,
   find the path, fix it.
2. **Keep the condensate physically separated from the bed.** No shared surface
   that liquid can travel along; a drip break in the path.
3. **Test it.** A cheap TDS meter is the right tool. Clean condensate should
   read very low. A rising TDS reading over time means creep is developing.
4. **Use food-grade CaCl₂.** It is GRAS as a food additive, and safe in the small
   amounts that matter here — but industrial grades carry additives that are not.
5. **Do not use lithium chloride for drinking water** without a verified
   no-carryover design and testing. Lithium is pharmacologically active — it is a
   psychiatric medication with a narrow therapeutic window — and unlike CaCl₂ it
   has no food-additive status. LiCl is the right choice for very dry air and the
   wrong choice for a casual build. If your site needs LiCl to work, that is a
   reason to be more careful, not less.
6. **Concentrated brine is an irritant.** Gloves and eye protection while mixing.
7. **This produces distilled water.** It is mineral-free and flat-tasting, and as
   with any harvested water, it is not automatically microbiologically safe —
   the condenser is an outdoor surface.

## What to measure if you build one

This is the most valuable thing anyone reading this repository could do, because
the sorbent numbers here are second-hand and two of them disagree by 10×
(research-log O17).

- **Pre-dawn RH and temperature** at the site — everything depends on it
- **Mass of the bed** before and after the night, on a kitchen scale. That is
  uptake in g/g directly, and it is the number the whole design turns on
- **Condensate volume** per cycle, and its TDS
- **Bed temperature** during regeneration
- **How much did not come back** — the gap between what the bed absorbed and what
  you collected is the cycle efficiency, guessed at 65% here

Post it as an issue, successes and failures alike.

## Sources

- CaCl₂ composites, 0.78–2.44 g/g water uptake depending on salt content —
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6359452/
- CaCl₂/sawdust composite, solar-powered atmospheric water production —
  https://sciencedirect.com/science/article/abs/pii/S0011916415002544
- Desiccant-based solar still: 671 mL/m² at 25 °C and **80% RH** (a humid-air
  figure, not a dry-air one) — https://iopscience.iop.org/article/10.1088/2631-8695/ad970a
- Solar-driven adsorption AWH: principles, materials, configurations —
  https://www.mdpi.com/1996-1073/18/16/4250
- Hygroscopic salt in a hydrogel-derived matrix, *Communications Chemistry* —
  https://www.nature.com/articles/s42004-018-0028-9
- Glycerol-stabilised hydrogel preventing salt leakage (the carryover fix) —
  https://techxplore.com/news/2025-05-atmospheric-harvesting-optimization-hygroscopic-hydrogel.html
- CaCl₂ handling and food-additive status, USDA technical report —
  https://www.ams.usda.gov/sites/default/files/media/2024TechnicalReportCalciumChlorideHandling.pdf
- Reduced uptake of airborne organic pollutants in salt-based AWH —
  https://pubs.acs.org/doi/10.1021/acsestwater.5c00836

Retrieved 2026-08-21 via search summaries; primary PDFs were not reachable from
this environment (O17). Verify before building.
