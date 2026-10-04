# Trail arrival rewards

`trail-deliveries.js` owns the saved travel cycle and its rewards. Core owns the production-rate adapter, simulation boundaries, completion hooks and persistence. The renderer consumes the resulting progress; animation never pays a reward.

The first completed Trail landmark gives an additional 30 seconds of current guild coin production, with a minimum of 24 coins. Each Trail landmark pays once per expedition run and stage index, in addition to its existing completion reward. Migrated saves baseline historical landmarks without paying them again.

After establishing the Trail, the explorer makes repeat deliveries to the outpost. A trip takes 30 work; speed is the square root of actual Trail travel, clamped between one and eight work per second. Cargo accrues at 40% of ordinary guild coin production during travel. Reaching the destination pays that saved cargo, with a minimum of six coins. The explorer rests at the endpoint for 1.2 seconds before starting the next trip. No cargo accrues during this rest. Increasing income at the last instant cannot reprice the entire trip.

`state.trailDeliveries` is optional additive schema-6 metadata. Missing metadata migrates to an empty travel cycle; malformed supplied data is rejected. The ledger contains the current phase, work, rest time, cargo, delivery count, lifetime delivery coins, last award, event sequence and current-run landmark receipt. Core partitions simulation at arrival/rest boundaries whenever an automatic purchase or dispatch could react to the payout. Otherwise, complete trips within a constant-rate interval settle in a batch, preserving the first partial cargo, final phase, minimum award per trip and individual last reward. Other rate-changing simulation events remain boundaries. This lets very long absences settle without exhausting the event safety limit. Selecting another area does not stop the Trail or create an arrival.

Refit and Charter clear unfinished work and cargo while retaining delivery history. Testing reset creates a fresh ledger. Coin awards update both spendable coins and lifetime earned coins. The UI baselines the saved event sequence on load, so previously paid awards are not celebrated as new.

Focused checks:

```powershell
node --test tests/games/wayfarers-guild-trail-deliveries.test.cjs
```
