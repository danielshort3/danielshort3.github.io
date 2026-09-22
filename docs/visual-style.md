# Shared visual style

Use this guide for the public site shell, homepage interests, and project, tool,
and game catalog illustrations. Keep approved artwork and each game's deliberate
visual identity. Screenshots, portraits, and playable scenes have their own
composition; they should not be converted into catalog icons.

## Brand identity and color roles

Use a navy-and-white foundation with restrained section accents. The DS mark and
vertical category tabs identify the site; preserve their geometry and the flush
mobile navigation. Keep content surfaces white or pale neutral rather than
giving each section a separate background theme.

The illustrated [brand guide](../documents/brand_guide.pdf) summarizes these rules
and keeps proposed improvements separate. See [its maintenance instructions](brand-guide.md)
for the editable source and the screenshot/PDF generation workflow.

The header uses the approved DS SVG beside the full Daniel Short label: a 38px
desktop image box and 40px mobile box. Preserve the image's aspect ratio with
`object-fit: contain`, prevent shrinking, and keep its border radius at zero so
the artwork is not clipped. Do not apply the old document's universal 120px
digital-logo minimum to header marks or favicons.

Keep header marks flat and stationary, including hover: no drop shadow, glow,
or scale animation. The favicon keeps the complete DS identity, including the
three chart bars inside the D. Use the approved compact favicon source rather
than simplifying the mark. Regenerate the PNG sizes and ICO together with
`npm run build:icons`.

Use the plain personal wordmark and current social card documented in
[the brand asset index](../img/brand/README.md). The analytics descriptor belongs
only to its labeled audience variant. The default social preview is configured
in `content/site/settings.json`; project/tool/game cards retain their own imagery.

| Role | Source / color | Use |
| --- | --- | --- |
| Main brand and Projects | `--brand-signal-blue` / `#005FED` | Global actions, links, Projects rail and primary actions |
| Blue hover / stronger emphasis | `--brand-deep-blue` / `#0145C8` | Hover and interaction states |
| Tools | `--category-tools` / `#087F8C` | Tools rail, divider, active states, primary actions |
| Games | `--category-games` / `#C94B0A` | Games rail and shared page controls |
| About / headings | `--brand-midnight` / `#091F3B` | Identity and main text |
| Contact / supporting text | Graphite `#334155` / slate `#475569` | Quiet navigation and secondary copy |

Projects aliases the main brand blue. Do not add near-identical blues for a new
component. Category accents belong in navigation, dividers, active states, and
primary actions; ordinary cards and secondary controls stay neutral. Use shared
semantic status colors and visible words for success, warning, and error states.
Status text must remain legible on its background without relying on color alone.
Use `--warning-text` for readable amber text on white; `--warning` is an accent
color for graphics or a suitable contrasting fill.

The personal-site brand message is **“Solving everyday problems with data and
thoughtful tools.”** Its authored `brandTagline` lives in
`content/audiences/personal.json` and flows through the generated audience config
to the closed homepage. About and page metadata use the same plain, personal
voice. Describe concrete work without repeating the tagline on every page.

Mobile demo launch previews should show the current working interface. Capture
the live layout, keep black drawing canvases and current controls, and use neutral
preview status for recorded examples. Do not present fixture data as live model
results or a screenshot as proof of current AWS connectivity. Keep these captures
separate from the approved illustrated catalog icons.

## Shared surfaces and controls

Use the semantic tokens in `css/variables.css` for new shared UI and when
updating an existing component:

| Role | Token | Treatment |
| --- | --- | --- |
| Outer site frame | `--radius-frame` | 12px external corners |
| Dialog | `--radius-dialog` | 12px corners, 1px neutral border, restrained shadow |
| Content card / tool panel | `--radius-card` | 10px corners, 1px neutral border, no resting shadow |
| Button / input / select | `--radius-control` | 8px corners |
| Status / tag | `--radius-pill` | Fully rounded when it identifies a state or category |
| Normal action | `--control-height`, `--control-padding`, `--control-font-size` | At least 44px tall, 10px by 14px padding, 14px text |

Use a 6px category-colored perimeter for open homepage tabs and a 4px perimeter
for libraries and detail pages. Keep the 2px masthead divider. Internal
boundaries use `--surface-border`; prefer spacing or one divider to another
enclosing card. Joined rails, editor headers, tab underlines, and flush mobile
edges remain square internally. The enclosing surface owns the external curve.

Use a filled category-colored button for the primary action, a neutral outlined
button for secondary actions, and a simple text action for tertiary choices.
Shared actions change color on hover without jumping or gaining a large shadow.
Preserve visible keyboard focus and space for its outline inside scroll areas.
Deliberately dark game canvases and game-specific controls retain their identity.

Mastheads and their content share the same horizontal gutter. Apply the gap
below a masthead divider once. A compact mobile account action may sit beside
Back when there is room; otherwise it aligns to the same left edge as the title.
Library titles use 16px text and descriptions 14px on both desktop and phone;
let cards grow or reduce columns instead of shrinking their descriptions.

## Two related visual families

**Interface glyphs** explain navigation and controls. Use simple outlined SVGs,
`currentColor`, rounded caps and joins, and approximately 1.8–2px strokes on a
24px canvas. Keep them flat, single-color, and free of decorative shadows.

**Content illustrations** identify an interest, project, tool, or game. Use one
clear silhouette with crisp navy edges, rounded geometric forms, and a small
amount of dimensional detail. Avoid embedding labels, letters, numbers, or UI
screenshots inside the illustration. Their adjacent text supplies the name.

## Palette, depth, and composition

- Use midnight navy `#091F3B`, signal blue `#005FED`, deep blue `#0145C8`, white,
  and mist `#EEF2F7`. A restrained copper `#D97706` accent may identify one detail.
  Category accents remain part of the surrounding navigation and controls.
- Keep outline weight visually consistent with the existing tool icons. Large
  shapes should still be recognizable at 48–64px; omit intricate decorative lines.
- Use gentle upper-left lighting and shading within the object. Avoid heavy
  ground shadows, glossy effects, colored glows, and dramatic gradients.
- Prefer a frontal view or a slight three-quarter perspective. Combine related
  objects into one compact mark rather than scattering several tiny symbols.
- Match apparent visual weight beside the existing artwork, not just canvas
  dimensions. A thin network diagram and a solid family frame need comparable
  presence without stretching or cropping either subject.
- Export real alpha transparency when possible. Never paint a checkerboard to
  suggest transparency. Existing white-backed project, game, and About icons
  use scoped multiply blending against the pale well; do not apply this treatment
  to photographs or screenshots.

## Presentation in the site

The shared tokens in `css/variables.css` define illustration wells:

| Token | Treatment |
| --- | --- |
| `--illustration-surface` | Pale `#F8FAFC` background |
| `--illustration-border` | 1px `#DCE3EC` outline |
| `--illustration-radius` | 10px corner radius |
| `--illustration-fill` | Contained image at 82% of the well |
| `--illustration-fill-compact` | Contained image at 78% in compact layouts |

The homepage, libraries, and About interests share this treatment. The well's
dimensions follow its context: a featured homepage row may have a larger well
than a dense library item. Keep the established card geometry; consistency does
not require making every box the same size. Do not enlarge or crop source images
to eliminate their intentional whitespace.

For new assets, compare against the approved artwork at the actual rendered size
on desktop and a 390px-wide phone. Check silhouette clarity, visual weight, edge
padding, and that white-backed art blends without a visible rectangular patch.
Also verify the 320px layout and confirm labels remain readable without overflow.

Existing prompt references:
[tools](tool-icon-generation-prompts.md),
[projects](project-library-icon-prompts.md), and
[games](game-library-icon-prompts.md).
