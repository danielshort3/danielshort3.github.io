# Stacked-station contracts

The game has six areas, each with five productive stations. Each station has three starter upgrades and three earned techniques, for 180 local upgrades. Three additional shared upgrades support every built station in an area. Local upgrades remain separate from cards and equipped player gear.

## Ownership and simulation

`station-content.js` defines stable identities and historical aliases. `stations.js` owns canonical quotes, requirements, station effects and production rates. Core owns shared wallets, simulation time, reward receipts and confirmed resets. Production and offline catch-up do not depend on scene rendering. Quantity modes are earned, exact and atomic; mastery counts purchased ranks. Lifetime output gates survive spending. Offline progress remains unlimited.

The first station begins with one upgrade. Its other two starter icons remain visible with concise lock requirements on the station row. Later techniques are intentionally revealed as their milestones are met. Building a ready station is explicit, followed by a destination callout and an action lesson. New stations add work without stopping earlier baseline production.

Crew Training adds 3% of base output per rank, capped initially at 30%. Shared Tools discounts ordinary local upgrades by 2% per rank, capped at 20%; it does not discount premium purchases, research, resets, scrolls or paid voyage manifests. Shift Planning adds two seconds of optional area boost per rank, capped at 30 extra seconds.

## Presentation

Currency HUD, area header and bottom navigation remain fixed. The illustrated world scrolls vertically, using 384-pixel logical art, 208-pixel station scenes and a taller first surface. Each scene retains its original size. Three compact starter controls sit inside its upper-right scenery; a separate question mark beside the station name opens contextual help. Each illustrated starter is one purchase, unlock or requirements control. Locked starters name their requirement and show current progress; ready unlocks pulse gold and affordable purchases use aqua. Long labels and prices cannot expand the scene. Earned batch quantities remain shared.

Station help opens a temporary bottom sheet above navigation. Selecting a starter displays its real effect and current-to-next comparison, currency Have/Need amounts, unlock progress and exact missing amounts. The sheet uses the same canonical purchase or unlock action as the scene control. Opening help never purchases, closing restores the scene without moving its camera, and sheet gestures cannot scroll the world. Multiple requirements and enlarged text scroll inside the sheet.

Upgrades opens a drawer without moving the world camera. It contains earned techniques, plans, projects and expansion controls in Station scope, plus introduced Area and Global scopes. Starter purchases are not duplicated there; its station link returns to the actual world controls. Legacy economies display only their real mapped starter purchases, including stations with fewer than three, until confirmed economy adoption. Floating optional boosts and caches occupy visible scenery and avoid the compact upgrade controls.

Mandatory lessons open the station's question mark, inspect the intended starter in its requirement sheet, close the sheet, then spotlight its real scene purchase control. They advance through successful actions, with supplied practice materials and once-only rewards. Native Back closes details before the drawer. Horizontal area gestures retain safeguards, drawer gestures do not navigate the world, and each area's camera position is remembered. Tutorial purchase prices and quantity indicators use ordinary purchase styling.

`station-scene.js` uses one visibility-aware scheduler. Workers use separate 64 × 96 pose cells, machinery and earned additions remain separate layers, and animation speed is capped independently of output. Reduced motion preserves working and boost states. Source prompts, measured crops, dimensions and SHA-256 hashes are recorded in the [art source directory](../../asset-sources/wayfarers-guild/stations/README.md) and published art manifest.

## Save compatibility

Schema 8 validates historical saves before adding dormant station metadata. An existing economy remains active for the complete current run. Its interface quotes and depicts only real legacy work; the old Quarry furnace remains Refinery until station adoption. The next confirmed Refit or Charter adopts economy version 4 while retaining learned content, highest-rank knowledge, projects, cap permissions, collections, tutorial receipts and reward ownership. Fresh games and complete testing resets start in version 4.

APK 17 bundles a schema-8-capable content-version-3 baseline and preserves the existing package, signing identity, appassets origin and update trust. Signed content updates above baseline 3 require APK 17 and schema 8. APK 16 rejects incompatible content cleanly. Paired content/save backups recover interrupted activation, and compatible patches continue applying in the same Activity.

## Validation

Validate quotes, flexible lifetime gates, actual effects, offline partitions, migration, supplied action lessons and ownership retention. Check rendered portrait sizes 320/390/430, short landscape, enlarged text and reduced motion against both approved references. Native release gates include unit/lint/device checks, a real installed upgrade, public byte verification, offline cold launch and same-Activity patching.
