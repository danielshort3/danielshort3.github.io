# Wayfarers walkthroughs and discoveries

`onboarding.js` owns the discovery inbox; `practice-lessons.js` owns hands-on lesson progress. Core integrates validation, canonical actions, bounded free practice, migration and offline synchronization. The UI owns spotlight geometry, focus, real target inspection, replay and safe exit.

## Why this design

The publisher's [Idle Skilling page](https://store.steampowered.com/app/1048370/Idle_Skilling/) describes starting with three of 21 interconnected skills. [ISEPS' official listing](https://play.google.com/store/apps/details?hl=en&id=com.AppSociety.ISEPS) describes progressively revealed upgrade menus. Those support gradual disclosure; they do not establish an exact tutorial interaction or reward schedule.

[Apple's game onboarding guidance](https://developer.apple.com/app-store/onboarding-for-games/) recommends short instruction at relevant moments, one step at a time, and replayable help. It also suggests considering a skip option; our required first-visit area guides are an explicit product choice, with safe Settings/Back access and saved resume instead of forced spending.

Official MapleStory Idle references include [Hero Journey, Quick Menu and Growth Guide patch notes](https://forum.nexon.com/maplestoryidle/board_view?board=6675&thread=3474001) and [currency explanations and help in the FAQ](https://forum.nexon.com/maplestoryidle/board_view?board=6679&stickyBoard=1&thread=3209008). Their exact forced spotlight behavior and tutorial payouts were not verified. Our action lessons and fixed rewards below are authored decisions, not claimed competitor conventions.

## Saved contract

Schema 6 accepts additive `state.onboarding`. Its original version-1 discovery inbox and historical guide rewards remain intact. Optional `onboarding.practice` version 1 records earned lesson progress, the active lesson, bound items, canonical action proofs, consumed practice supplies, completion rewards and first-open help rewards. Malformed supplied metadata is rejected.

`practice-lessons.js` defines the lessons; `onboarding.js` combines them with discoveries. Core supplies real current descriptors and executes practice transactions. Load `practice-lessons.js` after `upgrade-tiers.js`, then `onboarding.js` before Core in both bundles.

Required first-visit lessons cover all six areas, Cards and Equipment. Later mechanics are taught when relevant: upgrade tiers, expansion, operating plans, bulk purchases, guild improvements, connected projects, Focus, crew, companions, recipes, relics, kits, collection crafting and planning controls. A mature guild does not receive a queue of every available lesson.

Lessons use symbolic targets and one of three modes:

- **Inspect:** open the specified real comparison, slot, objective or review screen. The UI reports success only after the correct content is rendered.
- **Action:** use the real purchase, equip, fusion, scroll or configuration control. Core verifies the exact expected action and its successful state change.
- **Wait:** the mechanic currently has no valid action. It cannot fabricate progress; return when its prerequisite is available.

There is no acknowledgment button that completes an action step. Read-only replay does not mutate progress or grant resources. Consequential systems such as Refit, Charter, reforge, purchases and rewarded ads teach their actual review screens; a tutorial never forces a reset, payment or ad request.

### Actions and persistence

`onboarding-visit {id, intendedAction?}` opens an earned lesson; `onboarding-leave {id}` saves a safe exit. First-use triggers bind the actual control the player chose to the relevant lesson instead of presenting a queue of unrelated features. `onboarding-perform {id, stepId, token, action}` validates the current expected action, applies bounded practice supplies, executes the canonical engine action and records proof atomically. Stale, repeated, wrong-target and no-op requests do not advance. An equivalent ordinary successful action can also satisfy the current lesson.

`onboarding-inspect` accepts only the current inspect target and token. `onboarding-help-open` records the first opening of earned optional help. These UI receipts require the actual target or help content to exist; an intercepted click alone is insufficient.

The app saves action, proof and supplies as one transaction before reporting success. An uncommitted failure restores the complete pre-action state. A committed result with failed final verification remains pending Retry; it must not be spent or awarded a second time. Settings, export and safe exit remain reachable. Import/reset epochs invalidate callbacks from the prior guild.

### Free practice and rewards

A new local track receives one free first-rank purchase through the normal upgrade control. Already-invested tracks teach their existing comparison instead of forcing another purchase. Cards teach equipping, one guaranteed first fusion, a second saved deck and returning to the original deck. The guild supplies the fusion copies without consuming existing duplicates. Equipment teaches equipping and one guaranteed Steady Scroll; existing scroll stock is preserved. An item with all attempt slots used receives a truthful inspection lesson rather than forced reforging.

Practice costs use temporary transaction inputs and restore the original funded wallet balances precisely, preserving outputs and completion rewards. This avoids erasing a small balance when a late-game quote is too large for ordinary floating-point addition to retain it. The temporary inputs cannot be withdrawn or saved independently of the action.

Optional helpful information awards four ordinary coins on its first deliberate opening from Help. Automatic first-use interception alone does not claim this reward. Completing its hands-on exercise gives another eight coins. Both have separate durable receipts. Required area completion retains the fixed grants below; previously claimed historical rewards cannot be claimed again.

| Newly completed area lesson | Ordinary coins |
| --- | ---: |
| Trail | 12 |
| Quarry | 60 |
| Tower | 120 |
| Workshop | 200 |
| Ruins | 360 |
| Harbor | 600 |

Cards and Equipment have no extra completion currency beyond their practice materials. Practice is not a premium-currency faucet. Paid entitlements, ad receipts, loot RNG and pity remain under their existing owners.

### Migration and retention

Released saves keep their economy, ownership and original discovery history. Missing practice metadata adds relevant action lessons; meaningful existing investments are inspected rather than charged again. Refit and Charter retain proof and consumed supplies, including temporarily unavailable mechanics. Testing reset intentionally creates a fresh guild and new lesson receipts.

A reviewed import of an older backup from the same `createdAt` guild keeps the furthest lesson proof and already-consumed supplies/help rewards from the current guild. It restores the chosen backup's economy, not a copy of current inventory. Importing a different guild remains separate. Item bindings must be repaired against the restored inventory so a lesson cannot point at a card absent from that backup.

### Spotlight behavior

`onboarding-ui.js` presents a short coach in the top layer. The required real button and its necessary navigation remain usable; unrelated controls are inert. The coach reanchors after sheets, selection and confirmation screens change. Pointer, keyboard, native Back, small screens, text scaling and reduced motion are part of the rendering contract. A missing target offers recovery; it never automatically completes the step.

## Focused verification

```powershell
node --test tests/games/wayfarers-guild-practice.test.cjs tests/games/wayfarers-guild-onboarding.test.cjs
npm.cmd run test:wayfarers-guild:progression
npm.cmd run test:wayfarers-guild:onboarding:browser
```

Engine checks cover required actions, no-op/stale requests, bounded free practice, optional help, same-guild imports, replay, migration and prestige. Browser/native checks operate the actual highlighted controls and verify resulting ranks, inventory, saves, focus, visibility and safe exit. Trail destination bonuses have a separate [simulation contract](wayfarers-trail-deliveries.md).
