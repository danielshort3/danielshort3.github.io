# Wayfarers walkthroughs and discoveries

`onboarding.js` owns saved guide progress and an ordered discovery inbox. Core integrates its validation, actions, migration, and offline synchronization. The UI owns spotlight geometry, focus, replay, safe exit, and persistence before feedback. Load the module after `upgrade-tiers.js` and before `core.js` in both website and Android bundles.

## Why this design

The publisher's [Idle Skilling page](https://store.steampowered.com/app/1048370/Idle_Skilling/) describes starting with three of 21 interconnected skills. [ISEPS' official listing](https://play.google.com/store/apps/details?hl=en&id=com.AppSociety.ISEPS) describes progressively revealed upgrade menus. Those support gradual disclosure; they do not establish an exact tutorial interaction or reward schedule.

[Apple's game onboarding guidance](https://developer.apple.com/app-store/onboarding-for-games/) recommends short instruction at relevant moments, one step at a time, and replayable help. It also suggests considering a skip option; our required first-visit area guides are an explicit product choice, with safe Settings/Back access and saved resume instead of forced spending.

Official MapleStory Idle references include [Hero Journey, Quick Menu and Growth Guide patch notes](https://forum.nexon.com/maplestoryidle/board_view?board=6675&thread=3474001) and [currency explanations and help in the FAQ](https://forum.nexon.com/maplestoryidle/board_view?board=6679&stickyBoard=1&thread=3209008). Their exact forced spotlight behavior and tutorial payouts were not verified. Our three-step guides and fixed rewards below are authored decisions, not claimed competitor conventions.

## Saved contract

Schema 6 accepts optional additive `state.onboarding`:

```js
{
  version: 1,
  progress: { greenway: 0 }, // sparse earned guide IDs; 0..3
  active: null,             // open, unfinished guide ID
  entries: ['area:greenway'],
  announced: ['area:greenway'],
  read: ['area:greenway'],
  rewardClaims: []
}
```

The six area IDs have required guides with stable `purpose`, `operation`, and `next-step` steps. Cards and Equipment use the same ledger: their guides become required after the player explicitly claims their free starters, when the real deck and inventory controls exist. Explicit Next acknowledges instruction; it never purchases a rank or changes a plan. Missing spotlight targets use the authored descriptive fallback or a safe exit, never automatic completion.

Discovery IDs cover actual areas, major menus, Focus, exact batch entitlements, funded-project completions, ready upgrade tiers and explicitly claimed tiers. Order is first-observed engine order, including offline processing; entries are bounded by the finite authored registry rather than a capped recent-event log. Announced and read are separate. Merely opening a menu or reading a view cannot acknowledge anything.

`Core.getView(state).onboarding` and `.expedition.onboarding` are aliases. They expose `guides`, the current `active` step, `inbox.entries`, `unreadCount`, and one coalesced `notice`. Targets are symbolic keys such as `scene`, `area-upgrades`, `area-plans`, and `area-goal`, never DOM selectors. Guides expose complete steps for read-only replay. Discovery destinations identify the exact area, local track, catalog purchase, or feature.

Actions:

- `onboarding-visit {id}` opens an earned unfinished guide; an area must be selected.
- `onboarding-next {id, stepId}` accepts only the current step. The final step completes and pays once in the same mutation.
- `onboarding-leave {id}` saves the current step for a safe exit.
- `onboarding-announce {ids}` consumes automatic prompting while retaining unread entries.
- `onboarding-read {ids}` explicitly reads existing discoveries.
- `onboarding-open {id}` acknowledges and returns a destination. For a ready tier, it also claims the real entitlement and reads the resulting unlocked notice atomically: one **Unlock & go**, without a second success popup.

Ordinary tier claims outside the inbox produce one truthful unlocked entry. Read-only guide replay has no action, completion mutation, or award.

## Rewards and migration

| First newly discovered guide | Ordinary coins |
| --- | ---: |
| Trail | 12 |
| Quarry | 60 |
| Tower | 120 |
| Workshop | 200 |
| Ruins | 360 |
| Harbor | 600 |

Cards and Equipment guides have no additional grant; their separately claimed starters keep their existing contract. Guide completion never modifies premium balances, inventory, RNG, pity, rates, or frozen rewards. The coin grants are fixed, not production-scaled, and do not buy an upgrade automatically.

Refit and Charter retain guide steps, completion, discoveries and reward claims. Missing metadata in a released save adds incomplete guides for earned areas, but silently reads historical discoveries and baselines their reward claims. Returning players receive teaching on the next visit without historical popup or reward waves. Every existing economic field remains unchanged. Unsupported/malformed supplied metadata is rejected rather than replaced.

Testing reset creates new initial guide state inside the existing new save generation. The persistence/app layer owns reset-generation fencing and invalidates callbacks on import/reset/reinitialization; the engine does not invent another envelope identity. The view's `createdAt:version` identity supports UI invalidation but is not a substitute for that fence. Save a complete final-step transaction before showing reward success. Roll back only an uncommitted failure. If the record committed but its final generation check could not complete, retain that exact state pending Retry; never restore older progress over the committed reward.

## Focused verification

```powershell
node --test tests/games/wayfarers-guild-onboarding.test.cjs
npm.cmd run test:wayfarers-guild:progression
npm.cmd run test:wayfarers-guild:onboarding:browser
```

Tests cover all six guides, no-spend steps, stale/duplicate rejection, save/reload resume, replay, exact once-only grants, atomic tier opening, durable unread notices, offline partition order, old-save neutrality, actual Refit/Charter retention, malformed imports and fresh reset state. Rendered and native checks additionally own spotlight visibility, no modal chains, Settings/export escape, save-failure feedback and old-generation callbacks.
