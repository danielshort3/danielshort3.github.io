# Social preview font

`Inter-Latin.ttf` is a decompressed copy of the site's existing `css/fonts/Inter-Latin.woff2` variable font. Sharp's text renderer needs the TrueType form; passing WOFF2 can silently select a fallback font.

The build reads this checked-in font directly and does not need Python or a font download. To update it when the site's font changes, use FontTools with Brotli support:

```python
from fontTools.ttLib import TTFont

font = TTFont("css/fonts/Inter-Latin.woff2")
font.flavor = None
font.save("build/fonts/Inter-Latin.ttf")
```
