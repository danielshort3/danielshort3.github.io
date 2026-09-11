'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const sharp = require('sharp');
const icons = require('../../js/common/catalog-icons');
const { versionedImageUrl } = require('../../build/lib/versioned-image-url');
const root = path.resolve(__dirname, '../..');

async function main() {
  const image = '<img src="/img/projects/icons/babynames.png?v=abc&amp;size=2#icon" alt="" width="256" height="256" loading="lazy">';
  const rendered = icons.render(image);
  assert(rendered.includes('type="image/webp" srcset="/img/projects/icons/babynames.webp?v=abc&amp;size=2#icon"'));
  assert(rendered.endsWith(`${image}</picture>`), 'native PNG fallback retains all semantic and dimension attributes');
  assert.equal(icons.render(rendered), rendered, 'already wrapped markup is unchanged');
  for (const source of ['https://outside.test/img/tools/icons/text-compare.png', '/img/projects/babynames.png', 'img/tools/icons/../other.png']) {
    assert.equal(icons.webpSource(source), '', 'only explicit catalog PNGs use generated variants');
  }
  const context = { window: {}, module: { exports: {} } };
  vm.runInNewContext(fs.readFileSync(path.join(root, 'js/common/catalog-icons.js'), 'utf8'), context);
  assert.equal(typeof context.window.SiteCatalogIcons.create, 'function', 'bundling as CommonJS still exposes the browser API');

  const operations = [];
  const document = { createElement: (tag) => ({ tag, append(...nodes) { operations.push(`append:${tag}:${nodes.map((node) => node.tag).join(',')}`); } }) };
  const node = { tag: 'img', ownerDocument: document, setAttribute(name, value) { operations.push(`${name}:${value}`); } };
  const picture = icons.create(node, '/img/tools/icons/text-compare.png?v=abc');
  assert.equal(picture.tag, 'picture');
  assert.equal(operations[0], 'append:picture:source,img');
  assert.equal(operations[1], 'src:/img/tools/icons/text-compare.png?v=abc', 'picture source exists before assigning a potentially eager fallback');

  let count = 0;
  let originalBytes = 0;
  let optimizedBytes = 0;
  for (const directory of ['img/projects/icons', 'img/tools/icons', 'img/games/icons']) {
    for (const file of fs.readdirSync(path.join(root, directory)).filter((name) => /\.png$/i.test(name))) {
      const relative = `${directory}/${file}`;
      const variant = relative.replace(/\.png$/i, '.webp');
      const original = fs.readFileSync(path.join(root, relative));
      const optimized = fs.readFileSync(path.join(root, variant));
      const [before, after] = await Promise.all([sharp(original).metadata(), sharp(optimized).metadata()]);
      assert.equal(after.width, before.width, `${relative} width`);
      assert.equal(after.height, before.height, `${relative} height`);
      assert.equal(after.hasAlpha, before.hasAlpha, `${relative} alpha channel`);
      if (before.hasAlpha) {
        const [beforeAlpha, afterAlpha] = await Promise.all([
          sharp(original).extractChannel('alpha').raw().toBuffer(), sharp(optimized).extractChannel('alpha').raw().toBuffer()
        ]);
        assert(beforeAlpha.equals(afterAlpha), `${relative} keeps exact transparency`);
      }
      assert(optimized.length < original.length, `${relative} candidate must save bytes`);
      assert(icons.render(`<img src="${versionedImageUrl(relative)}" alt="">`).includes(`${variant}?v=`));
      count += 1;
      originalBytes += original.length;
      optimizedBytes += optimized.length;
    }
  }
  assert(count > 30, 'project, tool and game catalogs are represented');
  console.log(`Catalog icons: ${count} native fallbacks/dimensions/alpha verified; ${originalBytes} -> ${optimizedBytes} bytes (${((1 - optimizedBytes / originalBytes) * 100).toFixed(1)}% smaller).`);
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
