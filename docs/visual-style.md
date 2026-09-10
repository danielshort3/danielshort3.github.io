# Shared visual style

Use this guide for the public site shell, homepage interests, and project, tool,
and game catalog illustrations. Keep approved artwork and each game's deliberate
visual identity. Screenshots, portraits, and playable scenes have their own
composition; they should not be converted into catalog icons.

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
