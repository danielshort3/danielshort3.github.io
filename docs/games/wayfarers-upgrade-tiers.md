# Intentional upgrade tiers

`upgrade-tiers.js` owns purchase visibility, saved readiness, and explicit tier claims. Core injects the same gate into both the current six-area progression and retained expedition-v2 engine; the UI does not decide what a player has earned.

The save remains schema 6. Optional additive `upgradeTiers` metadata contains exactly `version:1`, `claimed`, `pending`, and `prompted` arrays of authored tier IDs. Strict validation rejects unknown IDs, duplicate/disjoint-state violations, impossible current tiers, and acknowledgements outside the pending set. Older valid saves without metadata are normalized once: previously learned, purchased, or available tiers are silently claimed. Their production, stocks, plans, paid ownership and RNG are unchanged.

New areas start with their first purchase track. Later tracks become ready under the existing rank/time/research requirements, but only an explicit claim reveals their purchase row. An earlier local tier must be claimed before the next one becomes ready. Project completion and physical production keep their established behavior; the new tier gate adds no hidden production penalty or currency charge. Existing researched branches and caps remain permanent through resets.

Global operations are grouped by profession and prerequisite milestone; research groups use their discovery milestone. Each network project has its own ready tier. Future catalog rows are absent until both the claimed tier and the real discovery prerequisites permit them. A funded research commission remains visible until completion. Wallet affordability never controls readiness, so spending currency cannot repeatedly hide or announce a tier.

`Core.getView(state).expedition.tiers` and the `upgradeTiers` view alias expose:

- `ready`: current pending descriptors, each with `id`, `areaId`, `tier`, `label`, `shortEffect`, `icon`, `tracks`, `unlockAction`, and `deferAction`.
- `notice`: one coalesced group of unprompted ready descriptors, or null. No locked future tier list is sent to the prompt.
- `claimedCount`: permanent acknowledged access.

`upgrade-tier-unlock` claims one ready ID. It spends nothing, buys no rank, draws no RNG and rejects repeats. `upgrade-tier-defer` accepts an exact set of pending IDs; it durably marks them as prompted without removing their ready badge. The UI saves this acknowledgement before showing a popup and does not chain popup dialogs. Offline readiness is independent of the capped milestone/event history. Refit and Charter keep claimed tiers, pending tiers and prompt acknowledgements; adoption of a retained run also preserves equivalent learned local tracks.

Core actions, exact local quotes, both automatic area purchasers, planned guild purchases and catalog descriptors enforce the same claim. Tests use explicit visit-time claims; simulation `advance()` never chooses a new tier while the player is away.

Run focused checks:

```powershell
node --test tests/games/wayfarers-guild-upgrade-tiers.test.cjs tests/games/wayfarers-guild-progression.test.cjs
node tests/games/wayfarers-guild-progression-simulation.cjs opening
node tests/games/wayfarers-guild-progression-simulation.cjs no-focus
```

The opening policy claims ready tiers during its active decisions. It buys first at 6 seconds, reaches the first worthwhile Refit at 2265 seconds (37m45s), and restores every positive pre-Refit canonical production rate at 731 seconds, confirmed for a further 60 seconds (32.3% of the first run). These are deterministic policy observations, not human retention measurements.
