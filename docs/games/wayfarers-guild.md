# Wayfarers' Guild economy and save contracts

The current six-area economy, bulk purchases, reset ownership and schema-5 migration are documented in [Connected progression](wayfarers-progression.md). The historical progression and reset sections below document the released 0.4 economy. An existing published run keeps its production formulas until its next confirmed prestige; that reset adopts the current reset contract. The premium, reward, account and numerical contracts continue to apply to both engines.

Wayfarers' Guild is an automatic incremental adventure game with progressively discovered, interconnected operations. `js/games/wayfarers-guild/content.js` defines shared guild content, `progression-content.js` defines the six-area network, `numbers.js` implements scientific quantities, and `core.js` integrates production, spending, random rewards and reset rules. `progression.js` owns new expeditions; `expeditions.js` retains compatibility with published runs. Presentation never awards resources. The modules expose both CommonJS exports and browser namespaces.

## Released progression and professions

The opening exposes one local Boots purchase, the shared coin wallet, and Old Footpath. Porters and Scouts reveal after the player has learned the first track. Undiscovered areas and their resource controls remain hidden. Local repeatable tracks continue beyond the opening milestones; later development is not limited to twelve purchases per track.

The first three discoveries establish permanent operations with different mechanical problems:

- **Trail:** travel to a crossing, choose a short path or a supply detour, and finish a bridge. Boots improve movement, Porters improve income and repairs, and Scouts improve movement and bridge work. The established route continues working and accepts further investment while the Quarry opens.
- **Quarry:** extraction feeds a finite ore buffer, carts feed a finite furnace buffer, and the furnace completes an ingot order. Actual flows are limited by supply and buffer capacity, so extra cart capacity cannot fix a furnace bottleneck. The established chain keeps producing after its first order and supports equipment and Trail development.
- **Tower:** allocate workers among building and protection, improve the lift, and light the beacon. Periodic pressure slows work but never destroys earned progress. The restored tower supports research and changes what earlier operations can do.

Opening cadence and production choices are tested with deterministic purchase policies. These are simulation measurements, not human playtest results or guaranteed times for every build. Spending the shared wallet on equipment, local tracks or global developments changes the timeline.

All unlocked areas simulate together, independently of the selected screen. Opening or expanding another area retains previous ranks, choices and production. Selecting Trail, Quarry or Tower only changes the view. Regional contracts provide further objectives for these same operations rather than replacing them with disposable stages. Prestige remains an explicit action with a separate preview.

Later discoveries change earlier operations. Quarry materials develop Trail logistics; freight can relieve Quarry hauling at a trade-income opportunity cost; Tower surveying unlocks a Trail mapping option; and refining developments introduce different yield and throughput choices. Later infrastructure can combine previously competing transport uses. These are directed dependencies with actual input and output effects, not just universal multipliers. Blueprints, mastery, equipment, specialists, companions, meals, doctrines, prepared kits and map preparations continue to support the applicable rates. Inspection compares the affected operation's rates.

The global **Upgrades** catalog brings network developments, repeatable improvements and guild research together. It shows exact costs, current and next effects, prerequisites, and affected areas. Opening Develop from an area filters this same catalog. A purchase may unlock a new decision, change a production recipe, relieve a bottleneck or improve a specific rate; choices should have different useful situations. Neither a selected screen nor an unclaimed completion popup is required to keep production running.

| Development | Changed earlier decision | Useful constraint |
| --- | --- | --- |
| Caravan routes | Trail couriers can carry freight instead of full coin deliveries | Quarry hauling is limiting; coin income can be sacrificed |
| Survey routes | Trail Scouts can produce maps instead of most coin deliveries | A map-funded development is the next objective |
| Precision smelting | Quarry trades furnace speed for greater usable-ore yield per raw unit | Extraction is scarce and the furnace has spare capacity |
| Two-lane relay | Trail can run trade and freight together at reduced individual strength | Both coins and hauling matter; a single-purpose plan still wins for one output |
| Overflow recovery | Blocked Quarry extraction can sell surplus rock | The raw queue is full, rather than starved |

Expansion does not multiply local upgrade prices or grant a free exponential income jump. Track prices depend on retained ranks; regions raise project work and open additional decisions and rank capacity. Published saves retain a fixed compatibility output factor for their active operation. That factor does not grow again merely by visiting or expanding an area.

The network has eighteen authored developments. Optical processing arrives in the first regional arc; the Prospecting network, Queue control room and Survey exchange additionally require the first, second and third earned Charters respectively. Their frontier and material prerequisites still apply. Charter previews identify the next transformation. Once built, these capabilities survive later renewals. Further regions, equipment and repeatable investment continue after these authored discoveries; this is not a claim of indefinitely new handcrafted mechanics.

Guild production runs through the corresponding area's working plan and capacity. Network factors apply once to the existing profession base, followed by the existing global modifiers; direct area production provides the opening bootstrap as well. Thus a later guild cannot make its Quarry's hauling or refining decisions negligible simply by buying profession upgrades. Capacity figures and actual total guild output are separate comparison metrics. Legacy states without the optional network record keep their historical formulas until normalization introduces the network.

Study and Map Room remain material-funded guild projects. Kitchen and other professions retain their original discovery rules. The already published profession economy, currencies, rare reward schedules and premium entitlements remain in the engine.

The eight professions are adventuring (Trail), mining (Mine), smithing (Forge), foraging (Forager's Camp), cooking (Kitchen), scholarship (Study), leadership (Guild Hall), and cartography (Map Room). Professions continue basic work when specialists move. Lifetime mastery gives a discrete rank bonus, rather than a continuously changing modifier that would make foreground and offline results differ.

- Six authored realms contain eighteen routes: Greenway, Copper Hills, Mistwood, Frostpass, Sunken Reach, and Starfall Heights. A dynamically generated Endless Frontier follows them. Every three completed routes increase the material tier used by production and conversions. Materials remain one ore wallet; the displayed tier describes its productive quality rather than introducing an inventory of interchangeable currencies.
- Ore buys tools, expedition boots, and research instruments. Instrument crafting also spends herbs. Workshop improvement reduces ore costs. Later upgrade prices use a steeper curve after rank eight, requiring stronger connected professions instead of a constant stream of equivalent clicks.
- The Kitchen turns herbs into provisions. Meals improve travel, mining, or scholarship. For new guilds, demand scales with material tier, travel development, and equipped tool/boot investment. Save suspends meals, Steady uses normal demand/strength, and Push doubles demand for a 50% stronger meal bonus. An under-supplied meal scales its bonus to actual available flow; ordinary travel always continues. Cooking and meals protect authored reserves and the saved objective/queue head.
- Research unlocks automation and alternative alloy/map conversions, changes ore efficiency and supply consumption, adds rewards to discoveries, and develops connections among mastery, specialists, and cartography.
- Two equipped specialists and one companion emphasize different outputs. Specialists are recruited once and can be reassigned; ownership survives both prestige layers. Their unselected professions retain production.
- Supply routes improve materials at the expense of travel. Discovery routes improve knowledge/maps at the expense of travel. Frontier routes retain full travel speed. Doctrines add a second emphasis and guide automatic forge ordering.
- Three challenges temporarily prohibit meals, equipment effects, or crew effects. Starting one resets the expedition without awarding prestige; route selection must follow that fresh expedition's sequential progress, even when later routes were discovered previously. Completing its target grants a one-time permanent benefit. Leaving a challenge restores ordinary rules without claiming its reward.
- Realm landmarks grant permanent collection bonuses. Frontier distance, reinforcement levels, profession mastery, and repeatable prestige improvements continue beyond the authored landmarks.

## Released presentation and shared introductions

The interface starts with a portrait pixel scene, the shared coin balance and its total rate, a local checkpoint meter and one Boots purchase. A fixed-height destination bar progressively reveals Trail, Quarry, Tower, Upgrades and Guild, with undiscovered operations absent. Switching areas retains their state and does not change the simulation. The three primary local tracks stay in a compact bottom tray; advanced developments live in the shared Upgrades catalog. Its short rows expose the next effect and complete price, while deliberate inspection shows dependencies and cross-area consequences. Crew, plans, renewal previews and the regional atlas remain under Guild.

The selected concept's sample numbers are illustrative. Displayed costs, percent effects, rates and unlock conditions come from the engine; the layout accommodates all eight professions and responsive desktop/landscape views. Pixel backgrounds and renderer geometry are documented in `img/wayfarers-guild/README.md`. Run `npm.cmd run test:wayfarers-guild:living:browser` for isolated-save opening, earned-floor, contextual-dock, complete-price and responsive checks.

`getView(state).presentation` is the single presentation gate contract; `getPresentation(state)` exposes it without assembling the full view. It does not change production, costs, prerequisites, random events or elapsed-time handling. A new guild starts with one coin wallet, one Trail, the boots purchase and one exact next-unlock explanation. The sole Trail destination does not need a navigation bar. Undiscovered rooms and future systems are absent from navigation; a single next-unlock descriptor explains what will be added and its real prerequisites.

| Earned trigger | Newly available presentation |
| --- | --- |
| Complete Old Footpath, route 1 | Guild with the Mine; ore and its actual production |
| First actual trail supply find | Journal with Finds and Record; the reward is already granted |
| Complete Copper Quarry, stage 2 | Forge equipment, a chosen equipment rank and its shared ore choice |
| Complete Watchtower Road, stage 3 | Forager's Camp, Guild Hall, crew choices, blueprints and optional local automation |
| Meet the actual first Refit requirements | Renewal preview; the game does not perform the reset automatically |
| First Refit, or Workshop ledgers research | Guild Plan with earned automation, reserves and objectives, even when Study is not built yet |
| First earned Starshards or owned premium item | Optional keepsake shop; a zero-balance Forge alone does not advertise it |
| Kitchen discovered | Provisions, meals and the supply choice |
| Study built, including its route, prior-room and material requirements | Research and the active relic search |
| First owned relic | Relic Collections; realm landmarks alone do not reveal an empty relic page |
| Actual caravan arrival or pending quote | Optional caravan control; eligibility alone does not show an empty offer |
| Guild Hall built | Crew, companion and doctrine choices |
| Complete Mosslight Way, route 7 | First usable contract; future locked contract rows remain hidden |
| Map Room built | Route modes, map preparations, targeted hunts and regional Charter planning |

The exact order can vary with player purchases, earned discoveries and preserved historical progress. Descriptions name game events and material costs, not fixed wall-clock promises. Historical room unlocks and caravan eligibility retain their original rules. Native paid-wallet balances or pending billing/reward recovery can require utility access independently of the engine's locally earned progression; the UI preserves those recovery paths.

The contract contains `primary` destination IDs (`trail`, `guild`, `journey`), usable `guildSections` (`rooms`, `research`, `crew`, `planning`), usable `journalSections` (`discoveries`, `collections`, `renewals`, `challenges`, `record`), `roomIds`, `resourceIds`, and explicit `show` flags. `systems` contains earned `{id,label,requirement,effect,action}` descriptions for contextual help. `nextUnlock` is one such future room descriptor with `progress` in 0–1 and `progressText`; a material-ready building still requires the actual purchase. `unlockedIds` supports stable identification, and `introductions` contains only earned descriptions that have not been acknowledged.

Schema 4 admits one optional non-economic record, `introductions:{seen:[systemId,...]}`. New states contain an empty list. Current-version saves predating this field remain valid; normalization adds the currently earned systems as seen, preserving every original gameplay field and avoiding a replay of old teaching. Strict v1–3 migration also baselines already-earned systems. Legacy versions cannot be enriched with this later field. Present records reject unknown IDs, duplicates and unexpected fields; the catalog bounds their size. No separate clock, reward or ownership is stored in this record.

`{type:'introduction-seen',ids:[...]}` acknowledges only currently earned IDs. It is idempotent and changes only the introduction record, leaving resources, random state, simulation time and ad receipts intact. Reading a view never acknowledges anything. The host saves acknowledgment before showing a contextual introduction; if saving fails it restores the prior seen list and keeps the introduction pending. Several unlocks, including an offline return, are summarized together rather than creating a mandatory queue of dialogs. The record survives Refit, Charter and export/import, while the same descriptions remain available as contextual help.

## Released persistent-area save contract

Schema 4 admitted an optional `expedition` record; the 0.4 release created nested expedition version 2. Current new saves use outer schema 6 and nested version 3, while migrated version-2 runs keep their current economy. Strict migration also accepts valid published expedition-version-1 records, preserves their active area's exact ranks, choices, work and buffers, and creates the already discovered earlier operations. Past ranks already discarded by version 1 cannot be reconstructed. Migration preserves the old outpost benefits without replaying route rewards, equipment gifts or premium milestones. Supplied malformed state is rejected before normalization.

The record separates frontier progress from `selectedArea`. Its `areas` map holds durable `greenway`, `quarry` and `watchtower` operations with their own ranks, choices, finite buffers and clocks. Developments, blueprints, family mastery, automation and bounded acknowledged events remain part of the network state. Area-specific purchase and choice actions carry `areaId`; changing the viewed screen cannot retarget a previously opened purchase. Only a new frontier capstone awards canonical route completion. Established production and revisiting a region never regrant it.

Area rates expose both station capacity and actual flow; scene motion follows actual flow. Every unlocked area is processed online and offline regardless of the selected view. Integration respects phase completion, finite-buffer boundaries, pressure changes, affordability, planning, supply and reward events. A pending offline backlog is resumable. New frontier contracts require the explicit discovery action unless auto-dispatch has been enabled; established operations need no repeated collection or rebuilding action. The rare premium scheduler counts only time after actual Forge eligibility, even when one long catch-up crosses that unlock.

`view.expedition.stage`, `cards`, `choices` and `scene` describe the selected area. `view.expedition.areas` exposes every discovered operation and its production; `links` describes directed dependencies. `view.globalUpgrades` is the canonical catalog with namespaced identities, area targets, actual effects, dependencies and canonical actions. Area-local purchases and choices use `{type:'expedition-buy'|'expedition-choice', areaId, id}`; selection uses `{type:'expedition-select', areaId}`; one-time network purchases use `{type:'expedition-development', id}`. Reading these descriptors is pure: it cannot spend resources, change an assignment or acknowledge an event.

Run `node --test tests/games/wayfarers-guild-expeditions.test.cjs tests/games/wayfarers-guild-opening.test.cjs` for the new opening and local-engine acceptance checks. The older core/overhaul suites also retain fixtures without the optional field to verify the historical profession formulas independently; persistence tests exercise migration into the new system.

## Historical reset contracts before six-area adoption

Every preview is calculated from the same state and formula used by the action. No prestige is performed automatically.

| Category | Expedition Refit | New Guild Charter |
| --- | --- | --- |
| Coins, route progress, field preparation, current meal | Reset | Reset |
| Ore, herbs, provisions, knowledge, maps | Retained | Reset |
| Room operations and equipment reinforcement | Retained | Reset; Guild Foundations provide initial operation levels |
| Room discoveries and equipment patterns | Retained | Retained |
| Profession mastery | Retained | Retained |
| Research, conversion recipes, automation licenses | Retained | Retained |
| Specialist/companion ownership and assignments | Retained | Retained |
| Field notes and Refit improvements | Retained; new notes awarded | Reset |
| Collections and completed challenges | Retained | Retained |
| Authored reserves, objective, queue, priorities, kit/preparation plans, loadouts and purchased capabilities | Retained | Retained |
| Paid route preparation | Consumed without refund | Consumed without refund |
| Area discoveries, local ranks, working choices and network developments | Retained | Retained |
| Local production buffers | Retained | Retained |
| Current run and frontier contract work | Renewed without replaying completed rewards | Renewed without replaying completed rewards |
| Retained historical outpost benefits, local family mastery, blueprints and area automation plan | Retained | Retained |
| Current regional project completion | Retained | Cleared for the new chapter |
| Crests and legacy upgrades | Retained | Retained; new crests awarded |
| Earned Starshards, lasting charms, banners, rare-drop schedule | Retained | Retained |
| Relics, active relic, duplicate study, luck research, hunts, prepared/active kits, discovery ledger | Retained | Retained |
| Caravan arrival, locked quote, pending receipt, completed-receipt IDs, remaining surge | Retained | Retained |

Refit needs the built Forge, at least one mining-tools and expedition-boots reinforcement, Watchtower Road completed in the current expedition, and a minimum amount of newly traveled distance. Its distance threshold doubles with each previous Refit in the current charter. Notes depend sublinearly on this expedition's work. The run's work is cleared after awarding them, so pressing Refit again cannot generate another reward from retained stock or equipment.

For new guilds, Charter requires a completed regional project and its current-expedition landmark, not a repeated Refit counter. Choose supply camps (provisions/herbs/maps), an atlas (knowledge/maps), or a waystation (ore/provisions/maps). Costs scale with the destination tier and relevant production investment, keeping the allocation meaningful after early levels. Each new Charter advances its landmark by three routes. Chapter work and the project are cleared only when the Charter is claimed, and the next landmark must be completed in a new expedition. Migrated guilds explicitly retain their former two-Refit/one-route-per-Charter rules.

## Authored plans and recurring expedition choices

After the first Refit or Workshop ledgers, players can reserve each ordinary resource, save for a specific upgrade/research/project, and choose operation and forge purchase priorities. A saved objective is purchased first when its cost plus reserves is available; it does not wait for a lower-priority queue's full cost. Standing purchase plan costs 2 notes and adds a six-item queue; Expedition outfitter costs 4 notes and renews a chosen Crucible kit. These are permanent capabilities, not another resettable percentage level.

Crests unlock Guild playbooks (three saved crew/doctrine/meal/relic/route-mode/planning configurations), Survey dispatch (one automatically purchased route preparation), Regional logistics (a supply preparation improves herbs and reduces meal demand), and Archive network (survey preparations help fund their own map supply). Each has a stated prerequisite and one-time cost. The existing repeatable pace/supply/insight and legacy upgrades remain available alongside these distinct capabilities.

Map stock has a recurring purpose: scouting grants 35% travel, a field camp grants 50% ore, and a survey grants 60% knowledge for the current route. Only one preparation can be chosen, and its costs are consumed on route completion, route change, or reset. Mistwood rewards field camps, Frostpass/Starfall scouting, and Sunken Reach surveying by removing an explicit regional travel detour. Basic expeditions remain possible without preparation. Production investment scales preparation costs; the amounts are shown before spending.

Automatic work runs on the same retained one-minute clock online and offline. It honors absolute reserves and the current objective/queue head; continuous herb/provision use stops at the same protected boundary. Manual purchases can explicitly spend those reserved resources. The plan never chooses a prestige. Its status states the current blocker or next planning action, and a bounded 20-entry audit records actual spending. Kit renewal pays the current recipe cost on every renewal, requires the Crucible equipped, and cannot stack bursts. A different already-prepared kit pauses renewal with an explicit reason until the player uses it or chooses that kit type; the planner never silently replaces it.

A playbook restores settings only: it cannot restore resources, route progress, preparations, rewards or old timers. Applying a whole playbook is unavailable during a constrained challenge. All plan settings, purchased capabilities and playbooks survive both resets; materials needed to execute the plan must be earned again. Reset previews list immediately affordable capabilities and explicitly qualified recovery information.

### New descriptor/action contract

- `view.development`: grandfathered flag, current stage, funded room projects, and three chapter project descriptors.
- `view.expedition`: supply choices, current demand, one-route preparation choices and the regional condition.
- `view.planning`: reserves, saved goal, priorities, ordered queue, kit/preparation choices, playbooks, capability descriptors, blocker and spending audit.
- `view.goal`: direct action and actionLabel, ready, remainingText and nullable etaSeconds; ETA assumes current rates and excludes future purchases. Late goals add longGoal context and at most two usual action descriptors in options. A free plan-goal action can be ready while its target still needs the displayed materials; rendering must retain that target shortage/ETA.
- Purchase descriptors expose effectText, shortageText and etaSeconds. Relic descriptors add family/chapter and a numerical before/after comparison. Conditional effects state when they apply.
- Actions: `project`, `capability`, `supply-plan`, `prepare-route`; `plan-reserve {id,amount:string}`, `plan-goal {action:null|{type,id}}`, `plan-priority {group,id}`, `plan-queue {action}`, `plan-remove {index}`, `plan-kit {id}`, `plan-preparation {id}`, `loadout-save {id:0..2,name}`, `loadout-use {id}`, `loadout-delete {id}`. Descriptors carry the exact action rather than requiring the UI to recreate costs.
- Advance summaries contain bounded completed/changes/spending arrays, discovery/route counts, blockedReason and nextChoices. Storage forwards them as `offline.summary`. A short return summary does not grant any rewards.

## Legacy continuity

Every valid v1–3 save migrates with `guild.grandfathered=true`. Discovered rooms, old route distances, old upgrade-cost curve, existing meal consumption, old kit costs, and prior reset requirements remain usable. New planning and chapter rewards are optional additions. Migration does not simulate time, remint gifts, import account-owned currency, or discard an ad quote. A pre-existing save with a future lastUpdate retains its time boundary. Current-schema imports must validate before normalization; the persistence layer rejects mismatched envelope/schema versions.

Corrupt or unsupported local records pause automatic saving and retain their original bytes. A validated preview alone does not authorize replacement; an explicit reviewed import can replace them. A storage write failure retains the previous main/backup recovery path. Native paid ownership remains an ephemeral cache applied before deferred offline catch-up.

## Starshards and permanent shop items

Building the Forge reveals the Starshard shop. It adds no controls or visible currency to the opening. Every item is available with earned Starshards; purchases are optional and no room, profession, recipe, challenge, realm, or automation requires payment.

After Forge construction, expedition time can produce very rare finds of one earned Starshard. Intervals follow a seeded exponential schedule with a mean of **120,000 seconds (2,000 minutes; 33 hours 20 minutes)**, quantized to milliseconds. This is an average, not a guaranteed interval or a daily reward. Foreground and offline time have identical eligibility; route speed, repeating short routes, Refit, and Charter cannot increase the drop rate or reroll the countdown. The initial seed is mixed from the game's creation timestamp; xorshift32 advances only when scheduling the next find. Saved RNG state and the remaining interval are retained across imports and resets. This is deterministic game simulation, not a source of cryptographic randomness.

The rare scheduler counts absolute simulation-clock millisecond boundaries rather than rounding each foreground tick. It loops over actual rare finds, not elapsed minutes. Only time processed by the main simulation earns finds; a pending offline backlog cannot be claimed twice. A ninety-five-year regression exercises roughly 25,000 finds, offline partitions and a save exactly on a drop boundary are compared, and reload resumes the same schedule.

One-time progression gifts complement the rare finds: **3 Starshards for the first Refit, 5 for the first Charter, and 2 for the first completion of each of the three challenges**. Previews disclose the gift. Claim records are retained and validated against completed progress, so replaying or resetting cannot repeat it.

| Item ID | Cost | Permanent benefit |
| --- | --- | --- |
| `compass` | 10 | +10% travel |
| `artisan` | 10 | +10% ore, herbs, and provision output; no extra herb consumption |
| `scholar` | 10 | +10% knowledge and maps |
| `banner-amber` | 5 | Amber cosmetic pennant |
| `banner-moon` | 5 | Moonlit blue cosmetic pennant |

Each item has one ownership level. Charm effects apply automatically, and one owned banner can be selected at a time. `premium-buy` spends the exported **earned** Starshard wallet; it rejects already owned items. Account purchases are owned and verified by the separate native/server billing integration. The engine accepts their item IDs through `setPremiumEntitlements(state, ids)` in an ephemeral `WeakMap`; they never mint earned Starshards or become local earned ownership. Effects use the union of earned and account ownership, never a stacked duplicate. Replacing or normalizing a state discards runtime account entitlements, so the host must reapply its verified account snapshot. A paid banner's saved selection is only a preference: it remains inactive until ownership is verified again. Local exports are not billing receipts or authoritative purchased balances.

Schema version 2 introduced `resources.starshards` and `premium: {rng, eligibleSeconds, untilDrop, drops, claimedMilestones, owned, equipped}`. `owned` records only earned purchases. The current schema is version 5. `migrateState` strictly validates historical versions 1–4, preserves original nested values, and adds only required later-schema fields. Version 3 migrations retain existing random clocks, pity, ownership, discoveries, active effects, receipt IDs, and exact locked/pending quotes; new relic progress keys start at zero. Malformed, enriched, missing-field, or unsupported legacy states return `null`. A v1 migration starts the earned wallet and eligible time at zero, preserves ordinary progress, and marks already achieved milestone gifts claimed without retroactive currency. Migration is pure and cannot import supplied premium data into a v1 save.

## Discoveries, relics, and kits

Random rewards add no extra opening control. After the Mine, common trail finds arrive every **8–12 eligible expedition minutes**, granting **30–60 seconds** of one unlocked resource's unboosted output. These small finds complement ordinary production. They do not award notes, crests, or Starshards. The Study starts a separate relic search: its first discovery arrives within **45 eligible minutes**. Subsequent searches use a six-hour exponential mean with an eight-hour maximum wait. Saved timers, random state, ownership, study progress, and research survive both resets, so faster routes, imports, or repeated Refits cannot reroll them.

| Relic | Rarity / base relative weight | One-slot effect |
| --- | --- | --- |
| Golden Pickaxe | Rare / 65 | Ore output ×1.50. |
| Surveyor's Lens | Epic / 30 | Knowledge ×(1.25 + 0.02 per Map Room mastery rank), capped at ×1.75; maps ×(1.20 + 0.02 per Study rank), capped at ×1.60. |
| Living Crucible | Legendary / 5 | Turns ore, provisions, and knowledge into a prepared mining or travel kit. While equipped, an active kit doubles its selected output for 30 expedition minutes. |

Later chapter families add distinct conditional builds:

| Charter earned | Relic | Effect while selected |
| --- | --- | --- |
| 1 | Marsh Lantern | Herbs +30%, meal demand −40% |
| 1 | Frost Compass | Scouting adds another 40% travel |
| 2 | Archive Quill | Surveying adds another 50% knowledge and 35% maps |
| 3 | Starsteel Anvil | Alternative alloy output triples; equipment ore cost−20% |
| 4 | Wayfarer's Standard | Any prepared route gains 60% coins and 25% travel |

Searches, pity targets, duplicate progress and new caravan targets are limited to earned families. The current saved search is never rerolled on a selection change. Existing pending quotes keep their disclosed target and fallback. The rarity numbers are relative weights among eligible relics, not per-minute drop probabilities.

Only one relic is active. The Crucible recipe costs 20 ore, 12 provisions, and 8 knowledge, each scaled by the material tier and, for new guilds, the relevant profession/equipment investment. This makes repeated kits a recurring supply choice rather than a negligible late-game expense. One kit can be prepared and one can run; a second active kit cannot stack. Unequipping the Crucible suppresses the kit's benefit while its timer continues. All costs and current numeric effects are exposed in descriptors.

The Map Room permits targeted hunts. A selected relic receives four times its base draw weight on **future** searches, while travel is reduced 15% and coin production 10%. The already scheduled hunt retains its saved target. Changing the selector neither rerolls nor shortens that search. At a pity deadline, the scheduled target is guaranteed if still missing, otherwise another missing relic is found. Duplicate draws give 20 percentage points toward a missing relic (preferring the scheduled target), and 100 points grants ownership without another roll. After every relic in the currently earned families is owned, duplicates instead grant a modest five-minute unboosted coin cache. A later Charter makes additional families eligible. The original three-relic Founders collection continues to unlock the constellation pennant, including for migrated players who already earned it. Later family completion is displayed as cosmetic collection recognition without another universal multiplier.

Three permanent researches use earned knowledge: Careful Salvage costs 120 and increases common caches 25%; Relic Lore costs 500 and changes subsequent search mean/maximum to four/six hours; Duplicate Study costs 1,500, requires the Map Room, and increases duplicate progress to 25 points. Acquisition, research, and collection completion are durable milestones rather than daily chores.

`luck.ledger` stores a monotonic `seq`, acknowledged `seen`, and at most 32 recent structured entries. Each entry includes sequence, type, rarity, title, reward, effect, simulation timestamp, and optional relic ID. The grant happens during simulation, before display. `{type:'discovery-seen',seq}` only advances the seen marker; it never grants or repeats a reward. The host saves this acknowledgment before showing feedback, and can summarize unseen entries after an absence. Legacy twelve-item textual notices remain informational and may group multiple Starshard drops differently by advance-call boundaries; the structured discovery ledger, RNG, clocks, and economy are partition invariant.

## Optional verified caravan rewards

For new guilds, the first natural Refit makes caravans eligible after the Forge is built. Established migrated guilds retain their Forge eligibility, and parked/locked/pending arrivals remain intact. An eligible caravan arrives after **60–90 expedition minutes**. Exactly one arrival waits without expiry. Ten percent are golden, and their doubled value is visible before requesting an ad. Dismissing an arrival grants nothing and schedules another interval. The server enforces at most **three verified completions per rolling 24 hours**; the engine mirrors the recent timestamps for clear feedback. No local timer, click, or simulated ad success can grant a reward through `act`.

The player chooses one guaranteed reward:

- **Supply shipment:** the arrival's fixed 90–120 minutes of current unboosted coin output and one chosen unlocked material. Golden shipments double both amounts.
- **Profession surge:** one chosen resource's production ×3 for 45 minutes, or 90 for a golden offer. Repeated surges in the same profession extend remaining duration up to 90 minutes; multipliers never compound. Choosing another profession replaces the previous surge. Expiry is a simulation event, including offline.
- **Relic expedition:** 20 percentage points toward a chosen missing relic from an earned chapter family plus a five-minute coin cache; golden offers give 40 points and a ten-minute cache. If that relic is found while the ad is pending, study transfers to another missing relic. If every currently eligible relic is completed meanwhile, the fallback is another copy of the exact quoted coin cache. The quote discloses this behavior; there is no second luck roll.

Shipments and coin caches exclude temporary kits, meals, and caravan boosts. Permanent development, owned charms, the active relic, and current expedition choices remain part of the quoted production base. Amounts are scientific numbers, not native-float estimates. Once the first watch attempt is begun, the saved arrival's `locked` flag fixes its exact quote even after an unearned cancellation or a reset. Retrying uses that quote; explicit dismissal allows a different future arrival. Unstarted selections are cleared on Refit/Charter for repricing. A pending verified reward and its old disclosed amounts survive both resets.

The shared quote is `{version:1,offerId,kind,golden,issuedAt,runId,charters,reward:{resources,surge?,relic?}}`. `resources` contains only `coins`, `ore`, `herbs`, `provisions`, `knowledge`, or `maps` as normalized `{m,e}` values. A surge is `{resource,multiplier:3,seconds:2700|5400}`; relic progress is `{id,progress:20|40}`. Neither purchased currency nor earned Starshards nor prestige currency appears in an ad grant.

The dedicated `beginCaravanReward(state,quote)` validates and persists the reservation. `cancelCaravanReward(state,offerId)` clears a reservation only after a confirmed unearned cancellation, retaining the locked arrival. `grantCaravanReward(state,{receiptId,offerId,quote,completedAt})` requires the exact persisted quote and checks the receipt/timestamp before applying it. Native/server code must verify AdMob completion before calling it; the local engine is not a verifier of external ad signatures. Save the mutation before presenting feedback. Repeated receipts return `{ok:true,duplicate:true}` without another grant. The most recent 256 IDs are retained; older IDs still cannot match a newer arrival's unique persisted quote. The pending receipt must be reconciled after reload rather than cleared by a local timeout.

## Simulation and numerical handling

Scientific resources are normalized JSON objects `{m,e}`: zero is `{m:0,e:0}`, and other mantissas satisfy `1 <= m < 10`. Arithmetic compares exponents without expanding enormous values into native floats. Wallets are not clamped to a gameplay maximum. The numerical representation admits exponents through magnitude `1e12`; upgrade counters and timestamps have separate defensive validation limits. Resource amounts above native float range remain serializable and display as scientific notation.

Rates are piecewise constant between meaningful events: a route completion, a mastery rank, supply exhaustion, an automated purchase, or expiry of a kit/surge. `advance` jumps to those events rather than simulating frames. Common finds, relic discoveries, and caravan arrivals are resolved inside the same simulation. In stable intervals with nondepleting supplies, the engine replays every retained RNG draw in order but aggregates additive payouts and materializes only the recent ledger. Conservative upper bounds prevent this batch from skipping a newly affordable automated purchase. When a find could change a constrained kitchen or meal supply, individual event boundaries remain explicit. Automatic purchases occur on a retained sixty-second planning cadence. Repeated known routes are batched; their fractional visual phase is discarded only beyond the mantissa's representable precision.

Foreground play, imports, and offline catch-up use the same function. Backward clocks earn nothing and leave the newer simulation timestamp intact. `lastUpdate` is the authority for earned time; the save envelope's `savedAt` is metadata.

A defensive 2,048-economy-event budget prevents pathological imported states or extremely long automated absences from monopolizing the page. The result returns `pendingSeconds` and advances `lastUpdate` only through the processed interval. A caller must continue with `advanceTo` and display a catching-up status if a backlog remains. This is a per-call work budget, not a loss or expiry of offline earnings. A fresh 400-day absence and a quiet ninety-five-year absence complete in one call, replaying every find. A mature ninety-five-year absence with active automation can leave a resumable backlog; tests explicitly check timestamp preservation rather than claiming it finished. Temporary effects and deferred ad receipts remain tied to processed simulation time.

## Public engine API

- `createState(now)` creates schema version 6. `validateState(input)` returns `{valid,errors}` and accepts only v6. `migrateState(v1orV2orV3orV4orV5)` returns a validated v6 state or `null`. `normalizeState(input,now)` returns a whitelisted clone of valid v6 input, otherwise a fresh state; persistence must migrate supported legacy input and reject invalid imports before normalizing.
- `advance(state,seconds)` and `advanceTo(state,now)` mutate state and return `{seconds,pendingSeconds,gained,events,summary}`. Invalid or backward intervals do not advance time. Gains are nonnegative wallet changes; consumed resources are represented by current balances and production/drain rates, not negative offline rewards.
- `act(state,action)` mutates only after checking the action's prerequisites and affordability, returning `{ok,message}`.
- Premium actions are `{type:'premium-buy',id}` and `{type:'premium-equip',id}`; `id:null` restores the default banner. `setPremiumEntitlements(state,ids)` replaces verified runtime account ownership and returns `{ok,message}`; invalid IDs leave the previous set untouched.
- `getView(state)` supplies formatted resources, rooms, actions, recipes, research, routes/modes, crew, doctrines, challenges, automation, both previews, collections, and the next objective. `progression.routePercent` is in `0..1`. Descriptors include their exact action, costs, visibility, affordability, reason, and selected/owned state. Conversion descriptors also include exact scaled outputs.
- `getView(state).premium` supplies `unlocked`, `balance` (scientific number), `balanceText`, `dropDescription`, `owned` (union), `earnedOwned`, `accountOwned`, effective `equipped`, `drops`, `eligibleSeconds`, and `items`. Item descriptors additionally have `kind`, fixed numeric `price`, `earnedOwned`, `accountOwned`, and `maxed`. Owned banners have a free equip action; owned charms are already active. Refit/Charter previews and challenge descriptors expose a numeric `premiumGift` alongside readable reward text.
- `getView(state).luck` exposes unlocks, active/owned relics, duplicate study, relic/research/hunt/kit descriptors, current kit, saved hunt, pity meter, collection cosmetic, and structured ledger. `getView(state).caravan` exposes arrival, exact quote and readable payout, locked/pending state, surge, quota, material/target choices, and reward descriptors. `getCaravanQuote(state)` returns a clone. Discovery actions are `relic-equip`, `relic-hunt`, `luck-research`, `kit-prepare`, `kit-use`, `discovery-seen`, `caravan-select`, and `caravan-dismiss`.
- Objective meters use the relevant requirement: equipment types reinforced, knowledge for research, routes for room discovery, funded room projects or the next unmet Charter requirement (the Refit counter appears only under grandfathered rules). Available prestige shows its ready reward; a free automation enablement shows readiness rather than unrelated route distance.
- `getRates`, `getRoute`, `getRefitPreview`, `getCharterPreview`, `getGoal`, and `format` support inspection. `Numbers` and `Content` expose the shared modules.

The durable state separates resources, upgrades, research, Refit upgrades, legacy, rooms, mastery/ranks, route, current run, current chapter, lifetime records, crew, meal/doctrine, automation, challenges, collections, the planning-clock remainder, and a maximum twelve recent notices.

## Balance and validation

Run `npm run test:wayfarers-guild:expeditions` for both the explicitly retained v4 fixtures and current progression, storage, reward and transaction checks. Run `npm run test:wayfarers-guild:progression:browser` for the six-area interface. The following price notes describe the historical profession economy, not the new local-track price curve.

For new guilds, starter boots, mining teams and the first two equipment types have a rank-based price discount: multiply the ordinary price by `(firstCost/base)^(1−rank/untilRank)` until the specified rank. Boots start at 2.4 coins and meet the ordinary curve at rank 16; miners start at 12 coins, mining tools at 1 ore and forged boots at 2 ore, all meeting their ordinary curves at rank 8. This creates a continuous geometric price curve, without a timer, expiring bonus, special currency, or purchase prompt. Other upgrades retain their ordinary prices. The later price multiplier uses exponent `x²/(1+x/40)` where `x=max(0,level−8)`, on base 1.025. Its tail becomes linear in level instead of becoming indefinitely steeper. Migrated saves keep base 1.045 with the old quadratic exponent.

Each of the first eight Boots purchases multiplies travel by `1.28 + 0.17×0.5^rank` and coins by `1.18 + 0.17×0.5^rank`, starting at rank zero. This yields travel gains of 45%, 36.5%, 32.25%, 30.125% and progressively smaller additions, settling to 28% from purchase nine onward. Coin gains follow 35%, 26.5%, 22.25%, 20.125%, then settle to 18%. Gains never dip below the ordinary benefit or rebound at a later rank. The extra opening development remains as a bounded 28.87% travel / 31.55% coin uplift at established levels. This is an intentional positive rebalance, not a claim that mature rates remain identical. Prices converge exactly; existing saves are never nerfed or rewritten. The descriptor calculates the next level's actual effect; it must not use the static default upgrade description as an early-game promise. Initial route distances are 120 and 450; the third route onward, resource production, reset requirements, random schedules and premium rules retain their established formulas. Migrated grandfathered guilds retain all original prices, multipliers, Forge payment and route distances. Existing schema-4 saves need no migration or new state field.

Later new-world route distances grow fourfold rather than eightfold. These are prototype balance decisions supported by simulations, not research-established optimal timings.

The reproducible policy harness is exported by the overhaul test module; it spends actual earned materials, uses public actions, and never injects room ownership into progression simulations. Individual mechanics tests use explicitly controlled scenarios to isolate effects. Long-horizon reports record seed, time partitions, exact reset policy, and source hashes outside the repository; no live save is touched. Baseline paths exclude rewarded ads, paid/earned charms and luck research; separately labeled reward comparisons use synthetic QA receipts. They are reachable authored strategies rather than optimal play or measured human retention.

Current focused checks cover migration and corrupt-file protection, first-day staged progression, paid room and map preparations, scaled supply demands, regional Charter projects, distinct planning capabilities, reserve boundaries, real queues, repeated kit costs and partition equivalence, playbook state isolation, family target gates, pure numeric comparisons, delayed caravan eligibility, and bounded return summaries. Existing tests continue to cover premium ownership isolation, rare RNG, old resets, scientific quantities, quotes/receipts, strict migration and long catch-up backlog handling.

Browser validation must exercise actual compact controls, room/goal discovery, all new planning forms, reset previews, numerical relic effects, import/reload, background recovery, narrow viewports and the native verification paths. Simulation evidence does not establish that people understand those controls or enjoy the resulting pacing.

### Current opening and progression checks (2 October 2026)

A fresh creation-time seed-0 policy checks earned purchases once a second. It does not inject resources or use ads, charms or gifts. An upgrade-focused policy buys boots at 9, 20, 37 and 82 seconds and mining teams at 61 and 119 seconds: six useful purchases in two minutes. The Mine opens at 61 seconds, after three familiar boot purchases. After the fourth Boots purchase, coin production is about 2.5 times the initial rate and travel is 3.5 times the initial rate. Only Trail/Mine and coins/ore are visible; the Forge still needs its route and real ore payment. With no purchases, the Mine opens after two minutes of automatic travel.

The exported `simulateOpeningGoals` policy follows the actual visible goal and moves to its named room. It saves for that action and otherwise purchases only the two visible room cards; it cannot traverse hidden upgrade IDs. It buys Boots at 9, 20 and 37 seconds, the first miner at 61 seconds, builds the Forge at **4m56s**, makes mining tools at **5m20s**, and makes forged boots at **5m54s**. Thus the actual guided path teaches both sides of the ore/equipment loop within six minutes. Future systems still require their own routes, materials and decisions.

The exported policy harness also tests real intermittent check-ins. These seed-0 results use the existing five-minute purchase, equipment, funded-project and patient-reset policy, with no ads or premium effects:

| Policy | First observed purchase / Mine | Forge | First Refit | Study | Guild Hall | Map Room | First Charter |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Push, every 5 minutes | 5m / 5m | 10m | 40m | 1h55m | 10h30m | 1d2h35m | 2d15h55m |
| Supply, every 5 minutes | 5m / 5m | 10m | 40m | 1h55m | 10h30m | 1d6h30m | 3d1h50m |
| Discovery, every 5 minutes | 5m / 5m | 10m | 40m | 1h55m | 10h55m | 1d12h35m | 3d16h |

One-minute check-ins reach the first Refit at 22 minutes; five-minute check-ins reach it at 40 minutes. The fast opening therefore becomes longer linked production goals rather than exposing the entire guild in the first session. The focused tests enforce the 5–10-second first purchase, frequent useful opening purchases, one new profession after learning boots, the guided six-minute Forge/equipment loop, strictly increasing upgrade prices and outputs, smoothly diminishing per-purchase gains, the bounded established-rank uplift, save/reload equivalence, and reachable late-game recommendations. These are simulation results, not measured human retention.

### Historical baseline policies (1 October 2026; predates the faster opening)

The remainder of this section records the previous economy for comparison. These numbers, including the old 66.7-second first purchase, are historical evidence and must not be presented as current timings. The current opening and first-Charter measurements above supersede them. The old supplemental reward, long-horizon, and reset branch comparisons have not been rerun for the revised opening.

The audit uses fresh creation-time seed 0, actual public actions and fixed check-in intervals, draining any pending offline time before the next decision. The first upgrade becomes affordable after 66.7 seconds at the initial 0.3 coins/s; the table records when each policy actually visits and buys it. The five-minute push policy reserves a room project before routine spending, equips scout/quartermaster and fox after recruitment, uses travel meals and the Crucible when earned, and eventually authors scout preparations. Supply and discovery policies choose their corresponding crew, doctrine, relic, preparation and regional project. They use their resource route mode until funding the project, then push the frontier. None uses ads, charms, luck research, purchased resources or injected progression.

| Seed-0 policy | First observed purchase | Mine | Forge | First Refit | Study | Guild Hall | Map Room | First Charter |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Push, every 5 minutes | 5m | 15m | 30m | 1h10m | 2h30m | 11h50m | 1d6h40m | 2d23h50m |
| Supply, every 5 minutes | 5m | 15m | 30m | 1h10m | 2h30m | 12h | 1d11h55m | 3d13h55m |
| Discovery, every 5 minutes | 5m | 15m | 30m | 1h10m | 2h30m | 12h25m | 1d20h | 4d11h45m |
| Push, hourly | 1h | 1h | 1h | 4h | 8h | 23h | 2d1h | 4d12h |
| Push, every 8 hours | 8h | 8h | 8h | 8h | 1d | 2d16h | 5d8h | 10d16h |

The declared patient reset policy takes the first available Refit, then only after six hours and a new route accomplishment since its preceding Refit; it takes each Charter when available. A delayed policy uses 24 hours instead, and a frequent comparator refits after six hours once the current run recovers lifetime reach. These are tested choices, not an optimal-reset theorem. Delayed and frequent five-minute push policies both reached Charter 4 and route 18 by day 90, with different numbers of Refits.

| No-reward policy | Day 30: route / Charter | Day 90: route / Charter | Day 180: route / Charter |
| --- | --- | --- | --- |
| Five-minute push, seed 0 | 15 / 3 | 18 / 4 | 19 / 4 |
| Five-minute supply, seed 0 | 12 / 2 | 15 / 3 | 18 / 4 |
| Five-minute discovery, seed 0 | 12 / 2 | 15 / 3 | 18 / 4 |
| Hourly push, seed 0 | 14 / 2 | 18 / 4 | 19 / 4 |
| Eight-hour push, seed 0 | 12 / 2 | 16 / 3 | 18 / 4 |
| Daily push, seed 12345 | 9 / 1 | 13 / 2 | 17 / 3 |

At day 30, the supply policy produced about 2.03e13 ore/s versus push's 7.53e12; discovery produced about 7.29e7 maps/s versus push's 6.09e7. These are different valid allocations, not proof of global balance: the sample does not optimize every relic, challenge, or reserve. Provisions often limit meals to ongoing production, so the quoted full meal bonus is not assumed sustainable. Later chapter relic families arrive after successive Charters; the push run completed its fourth Charter around day 72. Day 90–180 advances only one further route under that fixed policy, a remaining long-horizon pacing risk rather than evidence of six months of engaging human play.

The late objective now shows a nearer funded improvement, its actual shortage/ETA and a free way to save for it, alongside the longer Charter condition. It also exposes up to two useful existing choices with numeric effects and direct actions. At the reproducible day-90 state, the next travel-boots level costs 3.88e15 coins, needs another 1.84e15 (about 25.1 hours at the current flow), and adds 28% travel plus 18% coins. One displayed alternative equips the already-owned Frost Compass on the scouted route; the other suspends the meal to fund the authored Crucible kit plan. Neither requires a new content unlock or payment.

Branching that same valid snapshot for 24 hours with retained automation and no further manual decisions earns 7.47e18 expedition distance under the existing plan, 1.05e19 after equipping Frost Compass (+40%), and 1.47e19 after saving provisions for the Crucible (+96.6%). The latter pays for 47 separate kit renewals, trading meal use and repeated ore/provision/knowledge spending for sustained bursts. The fixed long-horizon policy does not adapt to these recommendations, so its plateau is not an optimized-play ceiling. This branch comparison establishes useful choices, while future human playtests must still assess whether these long goals remain interesting.

Supplemental creation-time seeds 1, 42 and 12345 cover minute and daily decisions and matched zero/some/maximum optional-reward comparisons. Seed 1, minute check-ins, reached Forge at 24 minutes and Charter 1 at 2d23h32m; by day 30 it reached route 15 / Charter 3. A matched seed-42 five-minute run reached its first Charter after 258,900 seconds without rewards, 252,000 with one guaranteed shipment per rolling day, and 247,500 with the maximum three. Fourth Charter took 6,234,000, 6,060,000 and 5,735,700 seconds respectively: about 2.8% and 8% reductions. At day 180, the arms reached routes 19, 20 and 20, all Charter 4. Reward arms used 180 and 540 synthetic QA receipts through the dedicated quote/reservation/grant APIs, respecting their stated rolling allowance. They did not call native ads or simulate ad viewing time. This comparison tests the economic effect of verified rewards, not the verification service or ad availability. All three arms excluded charms.

Elapsed days, visits and spending are reported separately: the minute/30-day policy makes 43,200 check-ins and 916 successful manual purchases; the daily/180-day policy makes 180 check-ins and 482 purchases. Automated engine purchases are excluded from those purchase counts. Redundant policy API calls are recorded as diagnostic attempts, not human clicks. No human active-time estimate is inferred from machine runtime.

The first-Refit comparison branches the same valid 70-minute snapshot, retaining its previous route index and fractional progress as a fixed target. With one-second measurement resolution and ordinary purchase opportunities every five minutes, replay takes 623 seconds without a Notes purchase, 548 seconds after buying Established trails for 2 notes, or 546 seconds after buying Standing purchase plan for 2 notes and queueing boots. That is 13–15% of the original opening, faster than the earlier 20–40% prototype window. Continuing without a reset finishes the next route in 1,143 seconds; the reset branches take 1,502–1,585 seconds. Refit therefore has a measurable immediate opportunity cost in exchange for retained planning/travel benefits and earned operation automation. The preview's distance/current-rate value is explicitly a rough baseline; it is not this replay simulation.

Reproduction files are kept outside the repository in the task's `guild-best-practices-qa` output directory: `overhaul-economy-audit.cjs`, `overhaul-economy-results.json`, `overhaul-policy-snapshot.txt`, `overhaul-supplement-audit.cjs`, `overhaul-supplement-results.json`, `overhaul-reset-audit.cjs`, `overhaul-reset-results.json`, `overhaul-late-choice-audit.cjs`, and `overhaul-late-choice-results.json`. Reports record source hashes and exact policies. Rerun them after an economy change before reusing these timings.
