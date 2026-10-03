# Wayfarers' Guild connected progression

The new progression engine is `js/games/wayfarers-guild/progression.js`; its authored areas, tracks, projects and purchase milestones are in `progression-content.js`. `core.js` owns shared wallets, prestige, random rewards and engine dispatch. `expeditions.js` remains the compatibility engine for an existing run imported from a released save. Both browser and Android bundles load the same modules in dependency order.

## Six areas, thirty-six tracks

Only the first Trail track appears at the beginning. An area's second and third tracks are introduced during its establishment; later discoveries add its remaining three tracks. Every discovered area continues producing while another area, the upgrade catalog or the Guild is open.

| Area | Introductory tracks | Tracks discovered elsewhere |
| --- | --- | --- |
| Trail | Pathfinding, Porters, Scouting | Caravans, Waystations, Railways |
| Quarry | Excavation, Haulage, Refining | Geology, Recovery, Deepworks |
| Tower | Surveying, Signals, Command | Optics, Forecasting, Relay Grid |
| Workshop | Assembly, Toolmaking, Metallurgy | Mechanisms, Precision, Replication |
| Ruins | Delving, Archaeology, Recovery Teams | Restoration, Attunement, Resonance |
| Harbor | Shipbuilding, Seamanship, Stowage | Contracts, Navigation, Fleet Command |

Ranks improve a particular stage of a production chain. A capacity increase does not imply the same increase in total income: extraction, carrying capacity, conversion, materials and the selected working plan can be limiting. The engine supplies both capacity comparisons and actual wallet effects. The interface must not invent percentage benefits.

Cross-area projects change available plans as well as rates. Examples include Quarry wheelworks teaching Trail caravans, Tower surveys revealing selectable Quarry deposits, Workshop sorting recovering waste, railways separating bulk freight from trade, lenses revealing Tower survey targets, and Harbor standards enabling parallel Workshop templates. Late overseas trade still uses the original Trail network.

## Permanent rank ceilings

Every track starts with a ceiling of 100. The area's permission to buy higher ranks survives both resets. A ceiling project must be attainable using its previous ceiling and adds a working capability alongside the larger number.

| Area | Ceiling 250 | Ceiling 1,000 |
| --- | --- | --- |
| Trail | Regional rail network | Continental exchange |
| Quarry | Industrial supports and sorting | Deepworks commission |
| Tower | Workshop lenses | Forecasting and resonator network |
| Workshop | Ancient precision patterns | Standardized production |
| Ruins | Restoration laboratory | Deepwater equipment |
| Harbor | Ocean charts | Navigation artifact |

Early ordinary rank gains are deliberately stronger than later relative gains. Prices must support useful investment into the raised ceilings; simply extending the earlier exponential cost curve to 1,000 would make those permissions cosmetic. Milestones have real production consequences, while learned options remain available after ranks rebuild.

## Exact bulk purchases

| Quantity | Permanent requirement |
| --- | --- |
| ×1 | Beginning |
| ×5 | First lifetime Refit |
| ×10 | Third lifetime Refit |
| ×25 | First Charter |
| ×100 | Ten lifetime Refits and one Charter |

Earning a larger mode also earns the smaller modes. No purchase buys more than 100 ranks. A batch is exact: an unaffordable ×10 purchase cannot silently buy seven. A batch that crosses the current ceiling cannot partially execute either. The interface offers a deliberate switch to a smaller quantity.

Canonical quotes include the aggregate resource cost, resulting rank, crossed milestones and economic revision. Execution checks the quote and spends once, without simulated time or intervening automated purchases. Rejected transactions leave economic state unchanged. Shared quantity controls apply to repeatable local and guild ranks, never premium items, one-time research or resets. Automatic rebuilding follows retained plans and protects reserves without selecting new branches or specializations for the player.

## Reset ownership

| State | Refit | Charter |
| --- | --- | --- |
| Ordinary local, operation and equipment reinforcement ranks | Rebuild | Rebuild |
| Ordinary stocks, finite buffers and unfinished run work | Clear | Clear |
| Funded permanent research commission and its completed work | Keep | Keep |
| Discovered areas, learned tracks, recipes and cap permissions | Keep | Keep |
| Crew ownership, working choices and saved plans | Keep | Keep |
| Notes and Notes improvements | Keep; award new Notes | Clear |
| Crest improvements | Keep | Keep; award new Crests |
| Premium ownership, earned Starshards and reward receipt history | Keep | Keep |
| Shared Focus and random reward clocks | Keep | Keep |

All discovered areas have a productive rank-zero baseline. Reset starter funds cover one earned batch of the first Trail track. A reset is always explicit and previewed. Challenge and navigation actions must not accidentally invoke these new prestige semantics.

After a confirmed reset, the first three familiar landmarks require 68% less work. This retained experience accelerates rebuilding; it does not multiply wallet income or award free ranks. The reset preview discloses it, and the nested state records it as `renewed`.

Previously earned local ranks also cost 50% less to rebuild. The discount stops at that track's retained best rank; a batch crossing that boundary prices each rank correctly. New best ranks always use the normal price. This makes resets feel faster without inflating mature production or giving free research progress.

Verified pending advertisements keep their originally disclosed rewards. Native paid ownership remains an ephemeral verified account cache, not exported earned ownership. Resetting must never reroll random rewards, duplicate a first-reset gift or refill Focus.

## Optional active choices

After the first Refit, a shared Focus pool holds three charges and recovers one charge every four simulated hours, including offline time. Area actions offer priority delivery, targeted deposits, focused surveys, rush orders, selected discovery study or voyage advancement. A preview shows the input and result before spending a charge. The same pool serves every area, avoiding six independent cooldowns that require constant patrol.

Automatic play can complete the authored arc. Active choices should provide a useful but bounded daily advantage; they are not required collection actions or reflex games.

Focus applies 25 times the selected operation's speed for 90 seconds. Six replenished daily charges give a theoretical 15% gain for that operation if fully usable. This is not a guaranteed whole-guild income bonus: other operations, input shortages and shared production can dilute it. A controlled mature-network day measured 6.40% more commission work and 6.69% more knowledge. The implementation retains this allocation-dependent benefit rather than claiming the original 10–20% global target was achieved.

## Progressive mobile controls

The mature navigation has three destinations: **Areas**, **Upgrades**, **Guild**. Once a second area opens, the area header reveals Previous, Next and a named picker with the unlocked-area count. Swiping unobstructed scenery uses the same selection action. It skips locked areas, stops at either end, ignores Android edge gestures and cancels vertical, diagonal, interrupted or multitouch gestures. Buttons and the picker provide the complete alternative.

The six tracks keep their authored order inside a bounded scrolling panel. Unlocking a row cannot stretch the viewport. The selected area and its scroll position survive navigation. Details, quantities and active choices use contextual sheets; only the next relevant discovery needs a locked preview. Plans remain available beside Expand at a completed frontier.

## Save migration and native checkpointing

The envelope and state use schema **5**. New expeditions use nested version **3**. Strict migration accepts supported historical versions before normalization. A released version-4 run retains its resources, ranks, caps, choices, buffers, reward clocks and current formulas; migration changes no reward or progress. Its first explicitly confirmed Refit or Charter adopts the new economy. Older earned ceiling permissions must not be silently reduced.

The localStorage key and Android appassets origin remain unchanged. The native checkpoint record remains version 1 and accepts matching game envelope/state versions 1–5, letting the canonical parser perform semantic migration. The first successful schema-5 write keeps the previous valid WebView bytes as the backup. Corrupt, mislabeled and unsupported saves remain protected; failed writes cannot replace the last good save.

`tests/games/fixtures/wayfarers-v4-state.json` is a synthetic reachable save captured from released commit `180f6b02db48f17a641fed8ed930004df7edb296`. Its expected prices, rates and 120-second/eight-hour outcomes come from that exact engine. It contains no account credentials or purchased wallet.

## Pacing and acceptance

Balancing targets are a first purchase in 5–10 seconds, useful opening decisions roughly every 10–30 seconds, and a first worthwhile Refit after 30–45 minutes of continuous reference play. Offline progress counts; these are not session-length requirements. After investing the reset reward, sustained prior output should recover in roughly 20–40% of its original development time.

The longer reference policy is three ten-minute visits per day without purchases or advertisements. The authored arc targets 28–56 days, with all six areas around week three and later discoveries transforming the earlier network. Simulations must make decisions only during those visits or through legitimately earned standing automation. They are balance evidence, not proof of human retention.

Acceptance includes exact batch boundaries and failed transactions, save and reward continuity, reset bootstrap, online/offline equivalence, useful alternative working plans, real late-to-early wallet effects, and rendered controls at 320×740, 390×844 and 915×390 with larger text. The Android release additionally requires matching package and signer, verified public bytes, a real in-app update from the previously installed release, retained settings and save, and offline cold launch.

### Measured balance for versions 0.5.0–0.5.1

The deterministic opening buys its first rank at six seconds and earns its first Refit at 37 minutes 45 seconds. After investing in Pace and enabling the earned standing plan, all initially positive canonical production rates recover after 12 minutes 11 seconds and stay at or above the original values for the following minute: 32.3% of the first run. This measures output recovery separately from merely becoming eligible for another Refit.

| Policy, without advertisements or purchases | Harbor learned, elapsed days | All 27 projects learned, elapsed days |
| --- | ---: | ---: |
| Three ten-minute visits per day, Focus unused | 14.33 | 34.33 |
| Three ten-minute visits per day, earned Focus used | 13.67 | 33.00 |
| Same visits and Focus, with two entire days missed | 14.00 | 33.67 |
| Continuous active decision policy | 12.54 | 30.96 |

These times start at day zero; a day-34 snapshot can therefore record a completion at 33 elapsed days. The continuous policy is an upper-activity comparison, not a suggested play schedule. The short-session policy makes no manual decisions between visits. Earned standing purchases and already funded research continue while away. The missed visits occur on days 13 and 14; there is no streak penalty.

In the no-Focus trace, the middle discoveries arrive as Ruins at day 5.67, Restoration at day 7, Precision at day 9, Attunement at day 11.33 and Harbor at day 14.33. This replaces the earlier single Harbor barrier. The longest later research interval is about 3.67 days after a new 1,000-rank permission creates further productive investment. The authored research arc is finite; completing it does not imply an independently validated years-long content supply or human retention.

Run the manual balance policy outside the fast unit suite:

```powershell
$env:WAYFARERS_SIM_OUTPUT = 'C:/Guild-QA/no-focus.json'
node tests/games/wayfarers-guild-progression-simulation.cjs no-focus
# Other policies: reference, missed, active; opening measures the first Refit and recovery.
```

The output directory must already exist. The runner starts from a new guild and records its policy, decisions, daily checkpoints and final state. It grants no currency and uses no advertisements or premium items. Renderer and behavior fixtures in `tests/games/helpers/wayfarers-progression.cjs` are explicitly funded and must never be used as pacing evidence.
