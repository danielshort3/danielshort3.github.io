# Cinder roster mitigation experiment

This temporary candidate has **not been adopted**. It changes only Ash Floor weights from 6 Ash Crawlers / 4 Lava Ticks to 8 / 2, and Vent weights from 7 Ticks / 3 Spitters to 3 Crawlers / 5 Ticks / 2 Spitters. Geometry, population, respawn cadence, route IDs and the optional encounter remain unchanged.

All 51 candidate runs completed their four-platform circuits; maximum measured stall time was 5.03 seconds. Fresh controls use the same runtime and protocol, so the [candidate comparison](training-cinder-candidate-comparison.json) passes the strict source and sampling guards.

| Level 19 class | Current XP/min | Candidate XP/min | Change | Candidate advantage over Rust |
|---|---:|---:|---:|---:|
| Fighter | 11,060 | 10,418 | −5.8% | +19.1% |
| Mage | 11,498 | 10,717 | −6.8% | +27.4% |
| Archer | 12,081 | 11,768 | −2.6% | +41.8% |

The candidate misses its intended 15–20% throughput reduction. It removes the condition of exceeding Rust by 25% across **every** class, but Mage and Archer still exceed that threshold. At level 28, XP changes range from −10.1% to +2.4%; Fighter and Archer XP increase. Using unchanged published peer means as a decision aid, affected cohort XP failures improve only from 31 to 30. This is not a new full-cohort measurement.

The heavier roster does not reduce throughput uniformly: less mobile prey can be easier to keep within attacks, and HP changes need not cross a class's number-of-hits threshold. These are possible explanations for the mixed response, not an isolated causal measurement. The observed result does not support adopting this variant simply to remove an aggregate dominance flag.

The control runs also verify the later NPC art correction. All **51 entire scenario payloads match the published originals exactly**, including movement samples, rewards, damage and companion recovery. The original 1,575 runs retain their measured source hashes. [Verification and source hashes](training-post-measurement-visual-verification.json) and the [isolated source diff](training-post-measurement-visual.patch) document the three asset-mapping edits; only five NPC definitions outside the measured fields gained an existing approved asset.

Detailed class results and the explicit peer-reference caveat are in [training-cinder-candidate-findings.json](training-cinder-candidate-findings.json). This experiment made no production edits.
