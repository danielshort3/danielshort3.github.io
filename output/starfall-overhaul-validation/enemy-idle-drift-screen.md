# Enemy idle translation screening

Six-frame idle row only. Fit premultiplied RGB and alpha under translation versus frame 0. Core excludes the outer 17% horizontal width, top 12%, and bottom 26% to reduce limbs and ground contacts; full silhouette fit is an independent cross-check. Search +/-40 px X and +/-18 px Y. All figures are runtime sheet pixels. Translation includes both intentional animation and unwanted packing drift, and changes in anatomy can bias fits. These are visual-review flags, not automatic corrections, acceptance, or proof of registration quality.

| Enemy | Core X span | Whole silhouette X span | Bounds-center X span | Loop endpoint X | Core/full disagreement |
|---|---:|---:|---:|---:|---:|
| eclipse-sovereign | 24 | 23 | 23 | -24 | 1 |
| rust-ratchet | 16 | 15 | 11 | -6 | 1 |
| snowglare-wisp | 15 | 17 | 16.5 | -9 | 2 |
| void-mote | 14 | 12 | 5.5 | -1 | 4 |
| cinder-spitter | 13 | 13 | 11.5 | -13 | 0 |
| clockbug | 12 | 12 | 11 | -8 | 0 |
| briar-stag | 12 | 11 | 12.5 | -8 | 1 |
| ember-wisp | 12 | 12 | 10 | -12 | 0 |
| lava-tick | 12 | 9 | 8.5 | -9 | 3 |
| cracked-mimic | 12 | 12 | 9.5 | -12 | 0 |
| bandit-cutter | 10 | 10 | 10.5 | -10 | 0 |
| clockwork-titan | 10 | 11 | 11.5 | -10 | 2 |
| stormbreak-roc | 10 | 10 | 10.5 | -10 | 1 |
| slimelet | 9 | 9 | 8.5 | -9 | 0 |
| brambleking | 9 | 9 | 7 | -9 | 1 |
| rift-aberration | 9 | 9 | 6.5 | -4 | 0 |
| dew-slime | 8 | 8 | 7.5 | -8 | 0 |
| glacier-sentinel | 8 | 7 | 3 | -1 | 1 |
| index-scribe | 8 | 8 | 8.5 | -8 | 1 |
| rimewarden | 8 | 7 | 5 | -8 | 1 |
| thorn-sprout | 7 | 7 | 7 | -7 | 1 |
| dust-imp | 7 | 7 | 6 | -7 | 1 |
| emberjaw-golem | 7 | 7 | 7 | -6 | 1 |
| astral-archivist | 7 | 7 | 6 | -7 | 2 |
| vine-snapper | 6 | 6 | 3.5 | 0 | 0 |
| quarry-colossus | 6 | 4 | 3.5 | -3 | 2 |
| shardling | 5 | 5 | 5 | -2 | 1 |
| rimeback-brute | 5 | 3 | 2 | -2 | 2 |
| thunder-ram | 5 | 5 | 4 | -4 | 0 |
| mossback | 4 | 4 | 4 | -4 | 0 |
| scrap-warden | 4 | 3 | 4.5 | 1 | 2 |
| ash-crawler | 4 | 4 | 4.5 | -1 | 1 |
| cloudcall-acolyte | 4 | 4 | 3 | -2 | 1 |
| bristle-boar | 3 | 2 | 2.5 | -2 | 1 |
| coil-sentry | 3 | 2 | 2 | -1 | 3 |
| glowcap-healer | 3 | 4 | 4.5 | 0 | 1 |
| gale-harrier | 3 | 3 | 3.5 | 3 | 1 |
| stormbound-archer | 3 | 3 | 3 | 1 | 0 |
| lumen-sentinel | 3 | 1 | 2 | 2 | 2 |
| oreback-beetle | 2 | 2 | 1.5 | -1 | 0 |
| icebloom-oracle | 2 | 2 | 3 | -2 | 1 |
| eclipse-duelist | 2 | 2 | 2 | 0 | 1 |
| bandit-thrower | 1 | 1 | 1.5 | 0 | 1 |
| frostling-scout | 1 | 1 | 1.5 | 0 | 1 |
