# Drought Survival Builds

Practical, buildable water harvesting systems for drought conditions.

## I Need Water Right Now

Only one of these builds is written up. The rest are planned, not available.

1. **Budget < $50**: [trailer-build.md](trailer-build.md) — written up, built,
   partially measured
2. **Budget < $200**: Basic Field Node — *planned, not written*
3. **Budget < $2000**: Complete Field System — *planned, not written*
4. **I'm a farmer**: Farm-Scale Deployment — *planned, not written*

## What These Do
These systems don't create rain. They amplify natural condensation (dew/fog) 
that would happen anyway.

Claimed output: 50-500ml per night depending on system size and climate. The
only measured figures in this repository are 85 ml and 110 ml on two nights from
the $45 trailer build; the rest of that range is an estimate. Whether these
systems amplify condensation at all is untested — see docs/research-log.md,
O3.

## Two things every build should get right

Both come out of the variable search (docs/research-log.md, Round 2), and
neither costs anything:

1. **Tilt the collector, and write down the angle.** Collector tilt is the
   largest design lever in the model — steeper drains droplets into the
   collector, but too steep and the surface sees less cold sky. The useful band
   is roughly 19–53°. No build in this repository currently specifies an angle,
   which means it is an uncontrolled variable in every result we have.
2. **Site it under open sky, out of the wind.** Canopy openness and shelter
   outrank every hardware choice except tilt. A collector under partial canopy
   loses the cold sky it needs to radiate to.

Adding a humidity sensor is the highest-value upgrade to any build: humidity
drives whether dew forms at all, and no build here measures it.

## Build Difficulty
- ⭐ = Hand tools, no electronics knowledge
- ⭐⭐ = Basic soldering, can follow tutorials
- ⭐⭐⭐ = Comfortable with Arduino/code
- ⭐⭐⭐⭐ = Can design and debug systems

## Climate Zones
The simulations carry presets for:
- Arid (hot deserts)
- Semi-arid (dry grasslands)
- Mediterranean (dry summers)
- Tropical dry (monsoon climate)

Note: builds do **not** currently ship optimized seeds. The seed optimizer
returns a degenerate answer for every climate and its output should not be
deployed — see docs/research-log.md, H5.

## Start Building
Pick your build, get the parts, follow the guide.

Post your results (success or failure) so others can learn.

## Safety
This amplifies natural processes. It's not weather modification.
It's not dangerous. But don't be stupid.

## Questions?
Open an issue. Field measurements — especially collector area alongside nightly
volume — are the single most useful thing you can contribute.
