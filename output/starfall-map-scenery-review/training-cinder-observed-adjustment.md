# Observed Cinder XP adjustment

A fresh uniform **1,575-run matrix** confirms the approved **0.93 multiplier for normal, non-elite Cinder kills**. The reconstructed before authoring retains multiplier 1. Bosses and random elites are excluded by the existing engine predicate; this preserves their rewards and produces an aggregate reduction of **6.35% on the main circuit** and **6.34% on the optional circuit**, rather than exactly 7%. These are actual earned-XP measurements, not the earlier linear projection.

Every unaffected run payload matches the previous matrix exactly: **1,473 unchanged scenarios**. The remaining **102 Cinder main/optional scenarios differ only in earned-XP fields**; movement, kills, incoming damage, healing, costs, loot and companion recovery are unchanged. All three reports record their actual final source hashes. Original reports are preserved in [the archive](training-pre-cinder-adjustment/README.md), without rehashing or reusing rows.

Across the 51 affected level-19/28 cohort rows, XP target failures change from **31 to 29**, and mean distance outside the existing target bands improves **28.02%** (10.164 → 7.316 percentage points). No deep premium exceeds 25%. This does not establish complete balance: the final comparison retains class-specific XP deficits, transit failures and optional-branch limitations.

The table shows three-seed mean XP/min before this scalar → after it. Geometry, rosters and every non-XP result remain the same for these paired runs. It is distinct from the final main comparison against the reconstructed earlier layout/encounter authoring.

| Level | Mode | Main XP/min | Main reduction | Optional XP/min | Optional reduction |
|---|---|---:|---:|---:|---:|
| 19 | fighter | 11,060 → 10,303 | 6.84% | 9,876 → 9,203 | 6.81% |
| 19 | mage | 11,498 → 10,715 | 6.81% | 10,124 → 9,449 | 6.66% |
| 19 | archer | 12,081 → 11,251 | 6.87% | 10,497 → 9,799 | 6.66% |
| 19 | party | 11,439 → 10,657 | 6.84% | 10,598 → 9,878 | 6.79% |
| 28 | fighter | 12,165 → 11,410 | 6.21% | 11,039 → 10,385 | 5.93% |
| 28 | mage | 14,843 → 13,923 | 6.20% | 11,800 → 11,063 | 6.25% |
| 28 | archer | 14,422 → 13,524 | 6.22% | 10,917 → 10,238 | 6.22% |
| 28 | guardian | 12,421 → 11,661 | 6.12% | 10,493 → 9,845 | 6.18% |
| 28 | berserker | 12,340 → 11,576 | 6.19% | 10,733 → 10,071 | 6.17% |
| 28 | duelist | 12,800 → 12,003 | 6.23% | 10,455 → 9,803 | 6.23% |
| 28 | fireMage | 13,482 → 12,646 | 6.20% | 11,130 → 10,413 | 6.44% |
| 28 | runeMage | 13,427 → 12,582 | 6.29% | 11,069 → 10,380 | 6.23% |
| 28 | stormMage | 13,017 → 12,191 | 6.34% | 11,392 → 10,683 | 6.22% |
| 28 | sniper | 11,625 → 10,902 | 6.22% | 10,220 → 9,602 | 6.05% |
| 28 | trapper | 12,570 → 11,796 | 6.16% | 10,509 → 9,842 | 6.35% |
| 28 | beastArcher | 13,161 → 12,341 | 6.23% | 10,539 → 9,880 | 6.26% |
| 28 | party | 13,329 → 12,499 | 6.23% | 11,494 → 10,758 | 6.40% |

[Full verification and scenario values](training-cinder-observed-adjustment.json) · [Final main/optional comparisons](training-comparison.json) · [Historical decision analysis](training-cinder-scalar-projection.md)
