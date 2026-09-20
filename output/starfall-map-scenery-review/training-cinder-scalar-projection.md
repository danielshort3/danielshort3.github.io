# Cinder XP scalar decision history

**Pre-adjustment analysis.** The numbers below describe the preserved Cinder 1.00 configuration. The approved 0.93 normal-kill multiplier was subsequently implemented and tested in a fresh uniform 1,575-run matrix. Random elites retain full rewards, so actual aggregate reductions are smaller than the initial all-XP projection. See [observed confirmation](training-cinder-observed-adjustment.json) and [final comparisons](training-comparison.json). Historical values below remain unchanged.

At the time of this analysis, **a provisional 0.93 multiplier was recommended; production had not yet changed.** This follows the measured geometry/encounter improvements and the rejected roster experiment. It is a seven-percent correction to ordinary Cinder kill XP, within the approved map-specific range; global enemy XP, rare drops and progression rewards remain outside this proposal.

These are **offline projections from observed runs**, not new engine measurements. Only Cinder XP is scaled. Kill throughput, travel, incoming damage, loot and potion costs stay at their measured values. XP rounds per kill in the runtime, so exact values and boundary passes require a fresh confirmation if the proposal is adopted. The full grid and every class/map row are in [the JSON](training-cinder-scalar-projection.json); [CSV](training-cinder-scalar-projection.csv) includes all selected scalar scenarios. The original 1,575 observed runs remain unchanged.

## Decision across all affected cohorts

The score is the mean percentage-point distance **outside** the existing target bands across 51 rows: ±15% of the ordinary median, or a 10–20% deep-field premium. Lower is better. Deep fields are excluded from the ordinary median. A separate 25% upper-premium check is a conservative diagnostic against creating a new extreme alternative; it does not replace the 10–20% target. Every tested eligible class and the fixed companion party are included.

| Cinder multiplier | XP target failures | Mean violation (points) | Worst violation (points) | Deep premiums above 25% |
|---|---:|---:|---:|---:|
| 1.000 | 31/51 | 10.2 | 47.5 | 0 |
| 0.975 | 27/51 | 9.0 | 46.6 | 0 |
| 0.950 | 28/51 | 7.9 | 45.6 | 0 |
| 0.930 | 28/51 | 7.0 | 44.8 | 0 |
| 0.925 | 27/51 | 6.8 | 44.6 | 1 |
| 0.900 | 23/51 | 6.1 | 43.6 | 2 |
| 0.875 | 21/51 | 5.7 | 42.5 | 4 |
| 0.850 | 21/51 | 5.3 | 41.4 | 5 |

At **0.93**, mean violation falls **30.8%** (10.164→7.029 points), with 31→28 failing rows. Level 19 error falls 8.280→6.400; level 28 falls 10.744→7.222. Five rows enter their target bands and two party rows leave them. Bandit/Quarry L28 party premiums become 21.5%/24.6%, versus 13.0%/15.9% currently; these are disclosed target misses, but neither exceeds25%. A 0.90 cut improves more under-rewarded classes, yet creates 25.6%/28.8% party premiums. A 0.85 cut minimizes mean error while producing five new premiums above 25%, reaching 36.3%; it is too aggressive for a shared map multiplier.

The fine diagnostic grid finds 0.928 as the lowest-error value without a premium above 25%. **0.93 is the conservative rounded proposal**, rather than selecting three-decimal precision to fit these seeds. Archer L19 remains a strict miss at 15.017% from the ordinary median; do not round this into acceptance. No single permitted scalar can satisfy all modes: for example, L28 fighter's combined deep-field constraints require at least 0.926 while mage would require at most 0.783. The latter is already outside the permitted range. A scalar cannot repair the remaining class-specific encounter differences.

## Per-class ordinary deviations and deep premiums

Values below show **observed 1.00 → projected 0.93**. At level 19 there are two ordinary fields, so Rust has the equal-and-opposite ordinary deviation shown for Cinder. At level 28 Cinder is the only ordinary reference; its zero deviation is mathematical, not independent evidence of good balance. Bandit and Quarry use the ordinary reference, not each other.

### Level 19

| Mode | Cinder deviation from ordinary median | Bandit deep premium | Quarry deep premium |
|---|---:|---:|---:|
| fighter | 11.7% → 8.1% | -10.8% → -7.2% | — |
| mage | 15.5% → 11.9% | -2.6% → 1.5% | — |
| archer | 18.5% → 15.0% | -37.5% → -34.8% | — |
| party | 12.9% → 9.3% | -0.4% → 3.7% | — |

### Level 28

| Mode | Cinder deviation from ordinary median | Bandit deep premium | Quarry deep premium |
|---|---:|---:|---:|
| fighter | 0.0% → 0.0% | 11.1% → 19.5% | 9.5% → 17.8% |
| mage | 0.0% → 0.0% | -2.3% → 5.0% | -13.9% → -7.4% |
| archer | 0.0% → 0.0% | -19.9% → -13.8% | -1.0% → 6.5% |
| guardian | 0.0% → 0.0% | 0.9% → 8.5% | -8.4% → -1.5% |
| berserker | 0.0% → 0.0% | 0.8% → 8.4% | 3.0% → 10.7% |
| duelist | 0.0% → 0.0% | 0.5% → 8.1% | -6.0% → 1.1% |
| fireMage | 0.0% → 0.0% | 8.7% → 16.9% | -12.1% → -5.4% |
| runeMage | 0.0% → 0.0% | 1.1% → 8.7% | -1.9% → 5.5% |
| stormMage | 0.0% → 0.0% | 0.9% → 8.5% | -20.8% → -14.8% |
| sniper | 0.0% → 0.0% | -33.6% → -28.6% | -22.4% → -16.6% |
| trapper | 0.0% → 0.0% | -6.7% → 0.3% | -12.5% → -5.9% |
| beastArcher | 0.0% → 0.0% | -37.3% → -32.5% | -15.7% → -9.4% |
| party | 0.0% → 0.0% | 13.0% → 21.5% | 15.9% → 24.6% |

## Cinder throughput versus before authoring

XP/min below is observed before, observed current, and projected current with 0.93. The before maps use reconstructed old authoring on the same corrected engine. Actual kill throughput is unchanged by this arithmetic; the proposal preserves the traversal and encounter efficiency gains. Incoming pressure, travel and costs are also unchanged. Because the benchmark fixes level/mastery, it does not model how altered XP affects a human's long-term progression.

### Level 19

| Mode | Before XP/min | Current XP/min | Projected 0.93 XP/min | Kills/min before → current | Improvement over before: current → projected |
|---|---:|---:|---:|---:|---:|
| fighter | 7,807 | 11,060 | 10,285 | 25.7 → 35.7 | 41.7% → 31.7% |
| mage | 7,704 | 11,498 | 10,694 | 25.7 → 37.6 | 49.3% → 38.8% |
| archer | 7,844 | 12,081 | 11,236 | 26.1 → 39.6 | 54.0% → 43.2% |
| party | 7,859 | 11,439 | 10,639 | 25.7 → 37.3 | 45.6% → 35.4% |

### Level 28

| Mode | Before XP/min | Current XP/min | Projected 0.93 XP/min | Kills/min before → current | Improvement over before: current → projected |
|---|---:|---:|---:|---:|---:|
| fighter | 9,248 | 12,165 | 11,314 | 26.1 → 33.2 | 31.6% → 22.3% |
| mage | 9,546 | 14,843 | 13,804 | 26.9 → 40.7 | 55.5% → 44.6% |
| archer | 8,784 | 14,422 | 13,412 | 24.7 → 39.4 | 64.2% → 52.7% |
| guardian | 8,865 | 12,421 | 11,551 | 24.9 → 33.9 | 40.1% → 30.3% |
| berserker | 8,573 | 12,340 | 11,476 | 24.3 → 33.7 | 43.9% → 33.9% |
| duelist | 8,818 | 12,800 | 11,904 | 24.8 → 35.1 | 45.2% → 35.0% |
| fireMage | 9,060 | 13,482 | 12,538 | 25.5 → 36.9 | 48.8% → 38.4% |
| runeMage | 9,356 | 13,427 | 12,487 | 26.3 → 36.7 | 43.5% → 33.5% |
| stormMage | 9,065 | 13,017 | 12,106 | 25.7 → 35.5 | 43.6% → 33.5% |
| sniper | 8,033 | 11,625 | 10,811 | 22.7 → 31.7 | 44.7% → 34.6% |
| trapper | 8,580 | 12,570 | 11,690 | 24.1 → 34.2 | 46.5% → 36.3% |
| beastArcher | 7,965 | 13,161 | 12,240 | 22.4 → 35.9 | 65.2% → 53.7% |
| party | 9,791 | 13,329 | 12,396 | 27.6 → 36.3 | 36.1% → 26.6% |

The proposal preserves **31.7–43.2%** improvement at level 19 and **22.3–53.7%** at level 28. Cinder's optional ordinary kills would receive the same map multiplier, so its own optional/main XP ratio remains approximately unchanged; this does not resolve optional-branch premium failures. Travel-target failures, lower-performing ranged deep-field classes, and the separate Thorn/Verge dominance remain explicit follow-up findings. Only an observed rerun after adoption can replace the current published comparisons.
