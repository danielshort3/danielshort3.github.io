# Project Starfall UI Visual Iteration Audit

Initial audit notes for the active UI visual iteration goal.

## References Checked

- `project_starfall_gdd_v0_5.md` sections `0.9 Current Player Guide` and `22. UI / UX Direction`.
- `pages/games/project-starfall.html` for the playable canvas shell.
- `css/games/project-starfall.css` for Starfall-specific DOM styling.
- `js/games/project-starfall/project-starfall-ui.js` for DOM and canvas windows, prompts, and hit regions.
- `test.js` Starfall attunement tests.

## Current Issues Found

- The canvas Attunement Prism ready prompt is still sized around `340px`, which makes current lines, target gear, status, auto controls, and action copy crowd into very narrow columns.
- The canvas attunement prompt tests currently assert the narrow `<= 340px` behavior, so regression coverage needs to be updated to protect a readable wider layout instead.
- The canvas ready prompt uses a fixed three-column comparison layout even when the viewport is constrained, making it prone to tiny side panels and clipped labels.
- The DOM attunement prompt already has wider card styling, but the canvas prompt and default window state lag behind it.
- Attunement prompt copy only partially exposes material state. Prism count is visible through action buttons, but Prism Shard conversion state is easier to find in the separate shard prompt than in the attunement flow.

## Priority Fixes For This Pass

- Widen the canvas attunement prompt for ready gear from a cramped single-card width to a readable two-panel RPG window.
- Add a stacked canvas layout for constrained viewports so the prompt stays inside the canvas instead of forcing unreadably narrow columns.
- Update tests so they prevent the old narrow attunement prompt from returning.
- Add visible Prism Shard conversion context in the attunement ready state where space allows.

## Completed In This Pass

- Increased the default canvas Attunement Prism ready prompt from the old narrow footprint to a wider readable layout.
- Added responsive canvas sizing for empty, ready, comparison, and Line Catalyst attunement prompt states.
- Added a stacked ready-prompt layout for constrained canvases, including a two-row footer so the primary action remains full width.
- Added Prism Shard conversion context to the canvas and DOM attunement ready states.
- Shortened tight-space shard/header labels after browser verification exposed truncation at the 760px desktop prompt width.
- Replaced a UI-side `CUBE_FRAGMENT_MATERIAL_ID` reference with the existing `cubeFragment` material key so focused tests run without relying on an engine-only constant.

## Verification

- `node --check js/games/project-starfall/project-starfall-ui.js`
- `node --check test.js`
- `npm run test:starfall:inventory`
- Served `/games/project-starfall` locally through the existing dev server and verified the route returned `200 OK`.
- Captured and inspected browser-rendered canvas screenshots:
  - `tmp/starfall-ui-screenshots/project-starfall-attunement-desktop.png`
  - `tmp/starfall-ui-screenshots/project-starfall-attunement-constrained-canvas.png`

## Remaining Scope

- This pass prioritized the known extremely narrow attunement window. The broader goal still calls for a full visual sweep of all Project Starfall UI surfaces against the guide and common UX standards.

## Panel Audit Pass

Additional running-game canvas screenshots were captured under `tmp/starfall-ui-screenshots/panel-audit/` for the command menu and these guide-required panels: Character, Equipment, Inventory, Skills, World Map, Quests & Trials, Shop, Upgrade Station, Monster Guide, Keybinds, Session Log, Party, Pet, Settings, and Admin Settings. Constrained screenshots were also captured for Command, Inventory, Skills, World Map, Keybinds, Shop, and Upgrade.

Issues found:

- The Skills canvas window was still sized as a very narrow 280px panel, even though the guide requires skill nodes to show current rank, next-rank changes, role tags, status, and prerequisites. The old 38px rows truncated summaries and made the Level Up controls dominate the available space.
- Other inspected high-impact constrained panels fit their windows and kept primary actions visible in this pass, though the full goal still requires continued inspection of deeper scroll states, hover tooltips, prompts, and panel variants.

Completed in this pass:

- Widened the default Skills canvas window to a readable 520px desktop layout, while preserving constrained viewport clamping.
- Reworked canvas skill rows into larger cards with visible name/rank, current effect summary, role tags, next-rank preview or locked status, and a stable Level Up button.
- Updated focused tests so Skills cannot regress to the old narrow window and 38px rows.
- Regenerated the missing per-item icon set from the Project Starfall item sheets so registered item/card UI assets resolve during the full test gate.

Verification added:

- `tmp/starfall-ui-screenshots/panel-audit/skills-1280x806.png`
- `tmp/starfall-ui-screenshots/panel-audit/skills-480x806.png`
- `node --check js/games/project-starfall/project-starfall-ui.js`
- `node --check test.js`
- `npm run test:starfall:inventory`
- `node build/process-project-starfall-ai-item-icons.js`
- `npm test`

## Command Menu Readability Pass

The command menu screenshot pass showed that the legacy 302px two-column menu could truncate guide-facing labels such as `Monster Guide`, especially on constrained canvases. After the constrained fix, the desktop capture still clipped the same label, so the base menu width was increased as well.

Completed in this pass:

- Kept the command menu right-aligned on desktop while widening it from 302px to 352px.
- Centered the command menu on compact canvases under 700px and allowed it to expand up to 430px while staying inside the canvas.
- Added a test contract so the readable desktop width and compact centered layout do not regress.

Verification added:

- `tmp/starfall-ui-screenshots/panel-audit/command-1280x806.png`
- `tmp/starfall-ui-screenshots/panel-audit/command-480x806.png`
- `node --check js/games/project-starfall/project-starfall-ui.js`
- `node --check test.js`
- `npm run test:starfall:inventory`

## Equipment Grid Fit Pass

The Equipment canvas screenshot showed the bottom gear row colliding with the window frame. The boot slot was partially hidden because the default equipment window was too short for the fixed four-row gear layout, and the gear renderer underreported its content height by omitting the tab/header area.

Completed in this pass:

- Increased the default Equipment canvas window height from 430px to 460px.
- Counted the tab/header area in the gear content height so the panel can scroll correctly if it is clamped in a smaller viewport.
- Added a test contract so Equipment keeps enough body height for the full gear grid while Shop and Upgrade retain their dedicated compact sizing.

Verification added:

- `tmp/starfall-ui-screenshots/panel-audit/equipment-1280x806.png`
- `tmp/starfall-ui-screenshots/panel-audit/equipment-480x806.png`
- `node --check js/games/project-starfall/project-starfall-ui.js`
- `node --check test.js`
- `npm run test:starfall:inventory`

## Character Overview Fit Pass

The Character overview screenshot showed the `Class Mastery` and `Advanced Class` cards colliding with the bottom frame. Their action buttons were partially hidden, so the guide-facing progression links were not cleanly usable.

Completed in this pass:

- Increased the default Character canvas window height from 430px to 500px.
- Tightened the overview responsive thresholds so 480px constrained canvases can still use four stat columns, eight equipment summary cells, and two progression cards.
- Updated `itemSheetAssets(...)` to emit generated standalone item icon paths, matching the current item-icon contract and keeping item art resolvable from catalog data.
- Added tests for the new Character window height and responsive overview thresholds.

Verification added:

- `tmp/starfall-ui-screenshots/panel-audit/character-1280x806.png`
- `tmp/starfall-ui-screenshots/panel-audit/character-480x806.png`
- `node --check js/games/project-starfall/project-starfall-data.js`
- `node --check js/games/project-starfall/project-starfall-ui.js`
- `node --check test.js`
- `npm run test:starfall:inventory`

## Party Panel Fit Pass

The Party canvas screenshot showed the `Party Skill` section crowded against the bottom frame, and the first cooldown empty-state line was clipped after the initial height increase. The guide-facing party controls should show roster slots, command strategy buttons, Party Skill unlock state, cooldowns, and active buffs without requiring a scroll in the default empty-status state.

Completed in this pass:

- Increased the default Party canvas window height to 580px so the full roster, command grid, Party Skill card, cooldown empty state, and buff empty state fit cleanly.
- Added a test contract so the Party window cannot regress to a shorter default while still requiring the Party Skill and status-grid render paths.
- Recaptured desktop and 480px constrained canvases; both report `maxScroll: 0` for the default Party panel state.

Verification added:

- `tmp/starfall-ui-screenshots/panel-audit/partyPanel-1280x806.png`
- `tmp/starfall-ui-screenshots/panel-audit/partyPanel-480x806.png`
- `node --check js/games/project-starfall/project-starfall-ui.js`
- `node --check test.js`

## Settings Panel Fit Pass

The Settings canvas screenshot showed the Display controls colliding with the bottom frame. Frame-rate options, `Reduced FX Off`, and `Reset Settings` were visible only at the clipped edge, which made a primary player settings panel feel cramped and unfinished.

Completed in this pass:

- Increased the default Settings canvas window height from 430px to 490px so Window Size, Audio, Display, frame-rate, Reduced FX, and Reset Settings controls fit in the default view.
- Corrected the Settings canvas content-height return so the panel no longer reports a tiny scrollbar for already-visible content.
- Added a test contract for the taller Settings default, the three Settings content sections, the Reduced FX control, and the corrected content-height return.
- Recaptured desktop and 480px constrained canvases; both report `maxScroll: 0` for the default Settings panel state.

Verification added:

- `tmp/starfall-ui-screenshots/panel-audit/settings-1280x806.png`
- `tmp/starfall-ui-screenshots/panel-audit/settings-480x806.png`
- `node --check js/games/project-starfall/project-starfall-data.js`
- `node --check js/games/project-starfall/project-starfall-ui.js`
- `node --check test.js`
- `npm run test:starfall:inventory`

## Admin Settings Fit Pass

The Admin Settings canvas screenshot showed the `Combat Metrics` section and `Show Panel` control clipped into the bottom frame. A taller first attempt exposed the remaining Asset Preview and reset controls, but pushed `Reset to 1x` into the bottom toast area on constrained canvases.

Completed in this pass:

- Reworked the Admin Settings default layout so `Reset to 1x` lives in the title row instead of the bottom of the scroll body.
- Set the default Admin canvas window to 630px high, which fits XP/Drop sliders, Performance Debug, In-Game Benchmark, Combat Metrics, Asset Preview, and reset controls without colliding with the HUD or status toast.
- Widened the canvas `Copy Debug Report` button slightly so the label is readable in the default desktop panel.
- Updated the item asset mapping so cube, Line Catalyst, slot coupon, prism, and Plinko item ids also resolve to generated standalone icon PNGs rather than sheet-frame URLs.
- Added a test contract for the Admin default size, title-row reset button, wider debug-copy button, section renderers, and corrected content-height return.
- Recaptured desktop and 480px constrained canvases; both report `maxScroll: 0` for the default Admin Settings panel state.

Verification added:

- `tmp/starfall-ui-screenshots/panel-audit/admin-1280x806.png`
- `tmp/starfall-ui-screenshots/panel-audit/admin-480x806.png`
- `node --check js/games/project-starfall/project-starfall-data.js`
- `node --check js/games/project-starfall/project-starfall-ui.js`
- `node --check test.js`
- `npm run test:starfall:inventory`

## Remaining Panels Fit Pass

The remaining desktop panel screenshots showed three concrete issues: Monster Guide clipped the lower detail rows and reported default scroll, Quests & Trials showed a chopped quest card in the first viewport, and unlocked Pet Assist overlapped the gear-rarity control with the loot filter row.

Completed in this pass:

- Increased the Monster Guide default window height so the default monster details, maps/drops lock rows, and calculated drop table lock card fit without panel scrolling.
- Increased the Quests & Trials default height so the first viewport shows complete quest cards while the long quest chain remains scrollable.
- Widened and heightened the Pet canvas window, then moved the gear-rarity row below the loot filter buttons so the unlocked Pet Assist controls no longer overlap.
- Added test contracts for the updated Quests, Monster Guide, and Pet canvas defaults and Pet unlocked-control spacing.
- Recaptured desktop and constrained diagnostic canvases for these panels. Monster Guide and unlocked Pet report `maxScroll: 0`; Quests remains intentionally scrollable for the full quest chain.

Verification added:

- `tmp/starfall-ui-screenshots/panel-audit/monsters-1280x806.png`
- `tmp/starfall-ui-screenshots/panel-audit/quests-1280x806.png`
- `tmp/starfall-ui-screenshots/panel-audit/pet-1280x806.png`
- `tmp/starfall-ui-screenshots/panel-audit/pet-unlocked-1280x806.png`
- `tmp/starfall-ui-screenshots/panel-audit/monsters-480x806-current.png`
- `tmp/starfall-ui-screenshots/panel-audit/quests-480x806-current.png`
- `tmp/starfall-ui-screenshots/panel-audit/pet-unlocked-480x806-current.png`

## Remaining Surface Cleanup Pass

The final screenshot sweep covered start screen, character select, Storage, Cash Shop, Beta Systems, Guide, Worldwright Console command/inventory tabs, item context menu, shard crafting, gear picker attunement, quest prompts, quantity prompts, confirmation prompts, Plinko, and Asset Preview. Earlier prompt captures for shard craft, quest prompt, drop quantity, confirmation prompt, gear picker attunement, Plinko, and Asset Preview were visually clean and did not require code changes in this pass.

Issues found:

- Storage opened under the fallback `Project Starfall` title and its default grid viewport clipped the lower visible slots into the frame.
- Cash Shop and Beta Systems showed partially cut catalog rows in their first viewport.
- Guide showed useful cards but opened too short for the default guide-card rhythm.
- Worldwright Console squeezed its tabs into a narrow row, truncating `Attunement`, and command sample buttons could draw long command labels outside their control bounds.
- The item context menu capture setup was corrected to verify an equipped starter weapon context menu rather than an invalid inventory payload.

Completed in this pass:

- Added a dedicated `Storage` canvas window title and taller/wider Storage default window.
- Increased Cash Shop, Beta Systems, and Guide default canvas windows so their first visible rows read cleanly with scroll affordance for the longer lists.
- Widened Worldwright Console and bounded all canvas button labels through the shared button renderer, while also widening command sample buttons.
- Added tests covering these default sizes, the Storage title, bounded canvas button labels, and the Worldwright command sample width.

Verification added:

- `tmp/starfall-ui-screenshots/remaining-audit-after/start-screen-1280x806.png`
- `tmp/starfall-ui-screenshots/remaining-audit-after/character-select-1280x806.png`
- `tmp/starfall-ui-screenshots/remaining-audit-after/storage-1280x806.png`
- `tmp/starfall-ui-screenshots/remaining-audit-after/cashShop-1280x806.png`
- `tmp/starfall-ui-screenshots/remaining-audit-after/beta-1280x806.png`
- `tmp/starfall-ui-screenshots/remaining-audit-after/guide-1280x806.png`
- `tmp/starfall-ui-screenshots/remaining-audit-after/worldwright-commands-1280x806.png`
- `tmp/starfall-ui-screenshots/remaining-audit-after/worldwright-inventory-1280x806.png`
- `tmp/starfall-ui-screenshots/remaining-audit-after/item-context-1280x806.png`
- `npm test`
