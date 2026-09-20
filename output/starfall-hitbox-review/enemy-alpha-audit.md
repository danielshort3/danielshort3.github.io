# Enemy sprite / combat-body pixel audit

Audited 51 enemy IDs / 48 unique sheets; 5508 frame/facing/reaction cases. Alpha threshold 64/255.

Distances are at actual game scale, not the 160px atlas scale. Red overlay is the current generic body. Gold circles show a proven empty-space witness; pixels are original shipped sprites.

| Enemy | Visibility | Worst active pose | Empty radius at least | Idle at least | Transparent old body |
| --- | --- | --- | ---: | ---: | ---: |
| Aurelion, Stormbreak Roc | live | projectile 3 | 55.16px | 36.5px | 52.6% |
| Clockwork Titan | live | hit 1 | 36px | 19.14px | 18.99% |
| Quarry Colossus | live | move 6 | 32.3px | 10.66px | 13.3% |
| Eclipse Sovereign | live | move 6 | 31.36px | 14.81px | 34.47% |
| Slimelet | live | move 4 | 30.44px | 5.74px | 61.86% |
| Rimewarden | live | attack 2 | 29.74px | 13.68px | 28.46% |
| The Astral Archivist | live | buff 5 | 29px | 19.58px | 17.74% |
| Emberjaw Golem | live | telegraph 5 | 27.5px | 12.2px | 20.7% |
| Lava Tick | live | telegraph 1 | 26.6px | 13.31px | 55.25% |
| Brambleking | live | hit 3 | 26.35px | 21.32px | 18.86% |
| Glassback | live | move 4 | 22.45px | 8.67px | 50.71% |
| Shardling | live | move 4 | 22.45px | 8.67px | 50.71% |
| Dew Slime | live | move 4 | 21.18px | 3.31px | 40.45% |
| Briar Stag | live | attack 3 | 19.9px | 7.91px | 42.53% |
| Bristle Boar | future | move 2 | 19.61px | 9.3px | 27.22% |
| Bandit Thrower | live | hit 2 | 17.75px | 10.75px | 33.6% |
| Eclipse Duelist | live | move 5 | 17.47px | 9.54px | 30.95% |
| Bandit Cutter | live | buff 2 | 16.78px | 8.03px | 32.7% |
| Direct Sheet Bandit | debug | buff 2 | 16.78px | 8.03px | 32.7% |
| Reference Sheet Bandit | debug | buff 2 | 16.78px | 8.03px | 32.7% |
| Hybrid Keyframe Bandit | debug | buff 2 | 16.78px | 8.03px | 32.7% |
| Puppet Composite Bandit | debug | buff 2 | 16.78px | 8.03px | 32.7% |
| Rimeback Brute | live | projectile 2 | 16.43px | 7.69px | 33.88% |
| Rift Aberration | live | attack 3 | 16.34px | 7.96px | 40.12% |
| Cracked Mimic | live | projectile 5 | 15.79px | 12.4px | 18.88% |
| Ember Wisp | live | projectile 2 | 15.64px | 9.05px | 23.07% |
| Thunder Ram | live | attack 4 | 15.36px | 7.97px | 31.57% |
| Dust Imp | future | move 3 | 15.11px | 8.89px | 29.35% |
| Icebloom Oracle | live | move 5 | 13.43px | 7.33px | 29.68% |
| Ash Crawler | live | move 3 | 13.28px | 11.31px | 18.9% |
| Frostling Scout | live | move 6 | 12.86px | 8.79px | 22.4% |
| Fault Skitter | live | move 5 | 12.56px | 9.11px | 17.06% |
| Clockbug | live | move 5 | 12.56px | 9.11px | 17.06% |
| Stormbound Archer | live | move 5 | 12.18px | 9.19px | 22.54% |
| Cloudcall Acolyte | live | attack 1 | 11.61px | 5.64px | 17.86% |
| Snowglare Wisp | live | attack 3 | 11.29px | 7.96px | 13.95% |
| Oreback Beetle | live | buff 6 | 11.09px | 8.4px | 16.97% |
| Rust Ratchet | live | buff 1 | 10.98px | 8.75px | 19.66% |
| Lumen Sentinel | live | attack 3 | 10.92px | 6.43px | 16.45% |
| Scrap Warden | live | hit 1 | 10.88px | 6.75px | 20.37% |
| Cinder Spitter | live | attack 2 | 10.83px | 6.62px | 15.41% |
| Index Scribe | live | buff 3 | 10.64px | 8.12px | 16.73% |
| Vine Snapper | live | telegraph 4 | 10.57px | 7.86px | 14.79% |
| Thorn Sprout | live | telegraph 4 | 9.85px | 4.63px | 13.14% |
| Mossback | live | move 3 | 9.62px | 8.51px | 12.62% |
| Gale Harrier | live | move 5 | 8.46px | 5.08px | 9.24% |
| Coil Sentry | live | hit 1 | 7.03px | 6.57px | 11.48% |
| Glowcap Healer | live | hit 1 | 5.79px | 2.47px | 9.36% |
| Rift Lantern | live | hit 2 | 5.35px | 4.68px | 8.73% |
| Void Mote | live | hit 2 | 5.35px | 4.68px | 8.73% |
| Glacier Sentinel | live | buff 6 | 4.34px | 3.39px | 8.6% |

Current runtime visual transform; all registered frames both facings. Old rectangle sampled at world-pixel centers. Worst candidate selected using exact Euclidean distance transform then refined against alpha pixel rectangles. Reported empty radius is a proven lower bound on the maximum, not a continuous optimizer. Defeat frames are evidence only; neutral and peak critical-hit reaction separated. No production artwork or runtime files edited.

Keep platform/physics body. Derive combat hurt geometry from each current sprite frame alpha as row spans or exact occupancy, using the exact renderer registration, scale, facing, and recoil box; query narrow phase after broad bounds. Do not replace the generic body with one inflated AABB.
