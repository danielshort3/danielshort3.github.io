'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { createLocalServer, buildRedirectLocation } = require('../../build/dev');

const root = path.resolve(__dirname, '../..');
const read = (file) => fs.readFileSync(path.join(root, file), 'utf8');

async function run() {
  const aliases = [];
  // Authored canonical tags are independent of the redirects being tested.
  const walk = (directory) => fs.readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const filename = path.join(directory, entry.name);
    return entry.isDirectory() ? walk(filename) : entry.name.endsWith('.html') ? [filename] : [];
  });
  for (const filename of walk(path.join(root, 'pages'))) {
    const file = path.relative(root, filename).replace(/\\/g, '/');
    const canonical = /<link\b[^>]*rel="canonical"[^>]*href="([^"]+)"/i.exec(read(file))?.[1];
    if (!canonical?.startsWith('https://www.danielshort.me/')) continue;
    const url = new URL(canonical);
    const destination = `${url.pathname}${url.search}`;
    aliases.push({ source: `/${file.slice(0, -5)}`, destination });
    aliases.push({ source: `/${file}`, destination });
  }
  assert(aliases.length >= 120, 'Cover both aliases for every current canonical source document, including Android and Copilot privacy.');
  const currentAliases = aliases.length;
  for (const audience of ['analytics', 'data-science', 'tourism']) {
    for (const prefix of [`/professional/${audience}`, `/pages/professional/${audience}`]) {
      for (const suffix of ['', '.html']) {
        for (const page of ['contact', 'portfolio', 'search']) aliases.push({ source: `${prefix}/${page}${suffix}`, destination: `/${page}` });
        aliases.push({ source: `${prefix}/portfolio/website${suffix}`, destination: '/portfolio/website' });
      }
    }
  }
  for (const page of ['analytics', 'data-science', 'tourism', 'resume', 'resume-pdf', 'resume-analytics', 'resume-analytics-pdf', 'resume-data-science', 'resume-data-science-pdf', 'resume-tourism', 'resume-tourism-pdf', 'destination-analytics', 'contributions']) {
    for (const suffix of ['', '.html']) aliases.push({ source: `/pages/${page}${suffix}`, destination: '/' });
  }

  const envDir = fs.mkdtempSync(path.join(os.tmpdir(), 'site-legacy-route-env-'));
  const server = createLocalServer({ envDir });
  let checks = 0;
  try {
    await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
    const origin = `http://127.0.0.1:${server.address().port}`;
    for (const { source, destination } of aliases) {
      const response = await fetch(`${origin}${source}`, { redirect: 'manual' });
      assert.equal(response.status, 308, `${source} must permanently redirect.`);
      assert.equal(response.headers.get('location'), destination, `${source} must reach its canonical route in one redirect.`);
      checks += 2;

      const query = 'input=a%20b%2Bc%26d&tag=one&tag=two&code=fixture%2Bcode&state=fixture%2Fstate';
      const destinationUrl = new URL(destination, origin);
      const expectedQuery = `?${query}${destinationUrl.search ? `&${destinationUrl.search.slice(1)}` : ''}`;
      const shared = await fetch(`${origin}${source}?${query}`, { redirect: 'manual' });
      assert.equal(shared.headers.get('location'), `${destinationUrl.pathname}${expectedQuery}`,
        `${source} must preserve encoded/repeated and OAuth inputs while appending missing canonical defaults.`);
      const sharedFinal = await fetch(new URL(shared.headers.get('location'), origin));
      assert.equal(sharedFinal.status, 200, `${source} with shared inputs must resolve successfully.`);
      assert.equal(new URL(sharedFinal.url).search, expectedQuery, `${source} must retain raw query encoding through the redirect.`);
      await sharedFinal.text();
      checks += 3;

      if (source.includes('/professional/')) {
        const overrideQuery = 'audience=personal&input=override%20value&tag=one&tag=two';
        const override = await fetch(`${origin}${source}?${overrideQuery}`, { redirect: 'manual' });
        assert.equal(override.headers.get('location'), `${destinationUrl.pathname}?${overrideQuery}`,
          'Retired professional links preserve explicit incoming audience and encoded inputs while targeting the single canonical route.');
        const overrideFinal = await fetch(new URL(override.headers.get('location'), origin));
        assert.equal(overrideFinal.status, 200);
        const overrideCanonical = /<link\b[^>]*rel="canonical"[^>]*href="([^"]+)"/i.exec(await overrideFinal.text())?.[1];
        assert.equal(new URL(overrideCanonical).search, '', 'An explicit personal audience must reach the personal document.');
        checks += 3;
      }
    }
    for (const destination of new Set(aliases.map((alias) => alias.destination))) {
      const response = await fetch(`${origin}${destination}`, { redirect: 'manual' });
      assert.equal(response.status, 200, `${destination} must resolve after the legacy redirect.`);
      const html = await response.text();
      const canonical = /<link\b[^>]*rel="canonical"[^>]*href="([^"]+)"/i.exec(html)?.[1];
      const canonicalUrl = new URL(canonical);
      assert.equal(`${canonicalUrl.pathname}${canonicalUrl.search}`, destination, `${destination} must retain its canonical tag and audience.`);
      checks += 2;
    }

    const sharedInputs = [
      { source: '/pages/text-compare.html', destination: '/tools/text-compare', query: 'input=left%2Bright%26value%3D1&tag=one&tag=two' },
      { source: '/pages/qr-code-generator', destination: '/tools/qr-code-generator', query: 'data=https%3A%2F%2Fexample.com%2F%3Fq%3Dhello%20world%26next%3D1&session=shared-test' },
      { source: '/pages/tools-dashboard.html', destination: '/tools/dashboard', query: 'code=test%2Bauthorization%2Fcode&state=return%3Dtool%26nonce%3D123' },
      { source: '/pages/demos/digit-generator-demo.html', destination: '/digit-generator-demo', query: 'digit=7&seed=42' }
    ];
    for (const { source, destination, query } of sharedInputs) {
      const response = await fetch(`${origin}${source}?${query}`, { redirect: 'manual' });
      assert.equal(response.status, 308, `${source} must redirect with shared inputs.`);
      assert.equal(response.headers.get('location'), `${destination}?${query}`, `${source} must retain the query and its encoding.`);
      const followed = await fetch(new URL(response.headers.get('location'), origin));
      assert.equal(followed.status, 200, `${source} shared inputs must reach the canonical document.`);
      assert.equal(new URL(followed.url).search, `?${query}`, `${source} inputs must survive the complete redirect flow.`);
      checks += 4;
    }

    const requestUrl = new URL(`${origin}/old?mode=guest&tag=one&tag=two&input=a%2Bb%26c`);
    const merged = new URL(buildRedirectLocation('/new?mode=owner#result', requestUrl), origin);
    assert.equal(merged.searchParams.get('mode'), 'owner', 'Configured destination parameters take precedence.');
    assert.deepEqual(merged.searchParams.getAll('tag'), ['one', 'two'], 'Repeated incoming parameters remain available.');
    assert.equal(merged.searchParams.get('input'), 'a+b&c', 'Encoded input content must not be reinterpreted as parameters.');
    assert.equal(merged.hash, '#result', 'Destination fragments remain intact.');
    assert.equal(buildRedirectLocation('https://example.com/new', requestUrl), `https://example.com/new${requestUrl.search}`, 'External redirects retain incoming inputs.');
    assert.equal(buildRedirectLocation('/new', new URL(`${origin}/old`)), '/new', 'Empty queries must not add a question mark.');
    assert.equal(buildRedirectLocation('/new?audience=analytics', new URL(`${origin}/old?input=a%20b%2Bc&tag=one&tag=two`)),
      '/new?input=a%20b%2Bc&tag=one&tag=two&audience=analytics', 'Canonical defaults must append without rewriting raw encoded query bytes.');
    assert.equal(buildRedirectLocation('/new?mode=owner', new URL(`${origin}/old?m%6fde=guest&input=a%20b%2Bc&mode=other`)),
      '/new?input=a%20b%2Bc&mode=owner', 'Configured destination keys replace every encoded/repeated occurrence without altering other values.');
    checks += 8;
  } finally {
    server.closeAllConnections();
    await new Promise((resolve) => server.close(resolve));
    fs.rmdirSync(envDir);
  }
  console.log(`Covered ${currentAliases / 2} current canonical documents and ${aliases.length - currentAliases} retired URL forms.`);
  return checks;
}

module.exports = run;
if (require.main === module) {
  run().then((checks) => console.log(`Legacy page redirects: ${checks} checks passed.`)).catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}
