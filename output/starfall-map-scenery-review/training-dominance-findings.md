# Training dominance findings before the Cinder adjustment

**Pre-adjustment analysis.** The numbers below describe the preserved Cinder 1.00 configuration. The approved 0.93 normal-kill multiplier was subsequently implemented and tested in a fresh uniform 1,575-run matrix. Random elites retain full rewards, so actual aggregate reductions are smaller than the initial all-XP projection. See [observed confirmation](training-cinder-observed-adjustment.json) and [final comparisons](training-comparison.json). Historical values below remain unchanged.

In this historical analysis, “current” refers to the archived 1.00 configuration, not the final adjusted release candidate. The proposed Cinder roster experiment was subsequently [tested and rejected](training-cinder-candidate-findings.md); its numbers and the original causal diagnosis remain review evidence.

These findings use the final 525-current / 525-before main-circuit matrices, with three fixed seeds and 300 measured seconds after 60 seconds of warmup. All compared runs completed their circuits and passed the unchanged movement/coverage gates. The before authoring runs on the same corrected engine. Numbers are three-seed means; XP per kill is total awarded XP divided by total kills. Incoming damage includes leader shield absorption. Consumable cost uses actual replacement prices, with the documented 99+99 starting stock and no purchase-budget constraint.

Detailed enemy/group kill counts, authored weights, damage dealt, costs and before/current metrics are in [training-dominance-findings.json](training-dominance-findings.json). No production edits or additional runs were made for this diagnosis.

## Cinder Hollow versus Rustcoil Ruins, level 19

| Class | Cinder / Rust XP per minute | Kills per minute | XP per kill | Incoming damage per minute | Travel | Consumable cost per minute |
|---|---:|---:|---:|---:|---:|---:|
| Fighter | 11,060 / 8,745 | 35.7 / 34.4 | 310.1 / 254.2 | 113.9 / 145.0 | 35.6% / 30.5% | 923.7 / 1,014.3 |
| Mage | 11,498 / 8,413 | 37.6 / 33.4 | 305.8 / 251.9 | 203.9 / 251.5 | 33.2% / 29.5% | 1,110.7 / 1,116.3 |
| Archer | 12,081 / 8,302 | 39.6 / 32.4 | 305.1 / 256.2 | 238.2 / 343.3 | 35.4% / 29.7% | 1,252.3 / 1,331.7 |

Cinder's XP advantage is **26.5%, 36.7% and 45.5%** for Fighter, Mage and Archer. This combines **3.7–22.2% more kills** with **19.1–22.0% more XP per kill**. Cinder takes more travel time than Rust but has lower incoming damage and lower consumable costs in every class. The advantage therefore is not supported by greater measured combat pressure in these samples.

The earlier authored maps were much closer: Cinder's advantage was **+0.4%, +6.4% and −2.0%**. Across the full pass, Cinder XP/min increased **41.7–54.0%**, versus **3.7–16.2%** for Rust. Cinder travel fell from **42.6–47.9% to 33.2–35.6%**, while its XP per kill increased only **1.6–2.2%**; the large Cinder improvement is predominantly increased kill throughput. The added west ramp and mid-height bridge remove detours between the first two hunting columns. Ground-pack authoring also replaces a uniform mixture of ground and flying enemies, increases the first group from 8 to 9, and changes main-group respawns from 6 to 5 seconds. This combined before/after comparison does not isolate the exact contribution of each change.

Current Cinder kills are **52.6% Lava Tick, 37.3% Ash Crawler and 10.1% Cinder Spitter**; **64.6%** come from the Ash Floor group and **35.4%** from the Vent group. Rust kills are **58.0% Rust Ratchet, 27.1% Clockbug and 14.9% Coil Sentry**, divided **64.3% / 35.7%** between its first two groups. Neither result depends on farming an optional eastern pocket.

The authoring explains the efficiency difference. These maps do not scale enemies to the map range: creation clamps player level ±1 to each enemy's native range. At player level 19, Lava Ticks are level 22 and Spitters level 28; their HP multipliers are only 0.72 and 0.88. Rust's common Clockbugs have 1.70 HP and 1.70 defense multipliers. Moving level-24-minimum Scrap Wardens out of Rust's ordinary groups reduced Rust's realized XP per kill from roughly 278 to 252–256, while improving its kill rate. Global XP formulas were not changed.

**Recommended next tuning:** keep the traversal fixes. Test a bounded ordinary-pocket roster adjustment that reduces Cinder's concentration of low-HP, higher-native-level prey at the low end of its range, reserving the strongest native-level mismatch for an optional pocket. Test Rust's heavy Clockbug share against a biome-consistent lighter/sentry mixture. Recheck levels 19 and 28 across classes before any reward scalar; slowing travel again would conceal the encounter mismatch.

## Thornpath versus Starfall Verge (Meadow), level 6

| Class | Thorn / Verge XP per minute | Kills per minute | XP per kill | Incoming damage per minute | Travel | Consumable cost per minute |
|---|---:|---:|---:|---:|---:|---:|
| Fighter | 2,165 / 1,270 | 26.4 / 20.8 | 82.0 / 61.0 | 91.1 / 62.9 | 47.2% / 49.8% | 578.7 / 571.7 |
| Mage | 1,867 / 1,397 | 22.8 / 22.9 | 81.9 / 61.1 | 134.8 / 49.9 | 47.0% / 46.3% | 541.3 / 497.0 |
| Archer | 2,120 / 1,435 | 25.5 / 23.6 | 83.2 / 60.8 | 157.7 / 38.3 | 43.6% / 49.7% | 611.3 / 499.3 |

Thorn's final advantage is **70.5%, 33.7% and 47.8%**. Its **34–37% higher XP per kill** is the shared driver: Mage kills at almost the same rate on both maps, yet earns 33.7% more XP on Thorn. Fighter additionally kills 26.9% faster. Thorn does carry more pressure—**1.45–4.11 times** incoming damage—but the XP advantage substantially exceeds the planned moderate premium. Both routes still miss the 30% travel target; respawn-only waiting stays below 0.5%, so merely shortening respawn timers is not supported by these measurements.

Thorn's current kill mix is **42.0% Dew Slime, 35.2% Vine Snapper, 13.6% Mossback and 9.3% Thorn Sprout**; **75.6%** of kills come from the first group. Vine Snappers have a native minimum level of 8, while all Verge prey can spawn at level 5–7 and carry lower XP multipliers. Verge kills are **56.5% Glassback, 35.1% Fault Skitter and 8.4% Rift Lantern**; **66.5%** come from Glass Basin.

The gap predates this pass (earlier advantages **54.7%, 45.5%, 30.2%**). Thorn's new ordinary groups remove the heavy Briar Stag from the main circuit, increase the lighter prey share, and move two population slots into the central group. Its XP per kill fell from about 101–104 to 82–83, but faster kills more than compensated. Verge's added Glass Basin slot and balanced upper/lower spawn weights improved XP **0.1–14.7%** and shortened travel; they did not erase the cohort gap.

**Recommended next tuning:** improve Verge's remaining empty transit before adding more bodies to its safe arrival. Review Thorn's first-group level-8 Vine Snapper share at the level-6 overlap and concentrate higher-level threats deeper in the map. Preserve its stronger danger identity, then retest the same classes at levels 6 and 14; a map-wide XP adjustment can improve one overlap while worsening the other.
