'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const directory = path.join(__dirname, '../../build/gtm');
const exported = JSON.parse(fs.readFileSync(path.join(directory, 'GTM-MX6DNH8L-activity.json'), 'utf8'));
const container = exported.containerVersion;
const parameter = (entity, key) => entity.parameter?.find((entry) => entry.key === key);
const variables = new Map(container.variable.map((variable) => [variable.name, variable]));
const triggers = new Map(container.trigger.map((trigger) => [trigger.triggerId, trigger]));

function resolveValue(value, data) {
  const variableName = /^\{\{(.+)\}\}$/.exec(value)?.[1];
  if (!variableName) return value;
  if (variableName === 'Event') return data.event;
  const variable = variables.get(variableName);
  assert(variable, `Missing referenced variable ${variableName}`);
  assert.strictEqual(variable.type, 'v', `${variableName} must preserve the site's data-layer value`);
  assert.strictEqual(parameter(variable, 'setDefaultValue').value, 'false', 'Do not invent defaults for absent event fields');
  return data[parameter(variable, 'name').value];
}

function readSettings(entity, key, data) {
  const settings = {};
  for (const entry of parameter(entity, key)?.list || []) {
    const name = entry.map.find((field) => field.key === 'parameter').value;
    const value = resolveValue(entry.map.find((field) => field.key === 'parameterValue').value, data);
    assert(!Object.hasOwn(settings, name), `Duplicate setting ${name}`);
    if (value !== undefined) settings[name] = value;
  }
  return settings;
}

function dispatch(data) {
  return container.tag.filter((tag) => tag.type === 'gaawe').flatMap((tag) => {
    const fires = tag.firingTriggerId.some((id) => {
      const trigger = triggers.get(id);
      assert(trigger, `Missing trigger ${id}`);
      const filter = trigger.customEventFilter[0];
      assert.strictEqual(filter.type, 'MATCH_REGEX');
      const expression = filter.parameter.find((entry) => entry.key === 'arg1').value;
      return new RegExp(expression).test(data.event);
    });
    if (!fires) return [];
    const settingsName = parameter(tag, 'eventSettingsVariable').value.slice(2, -2);
    return [{
      name: resolveValue(parameter(tag, 'eventName').value, data),
      parameters: readSettings(variables.get(settingsName), 'eventSettingsTable', data)
    }];
  });
}

const googleTags = container.tag.filter((tag) => tag.type === 'googtag');
assert.strictEqual(googleTags.length, 1, 'Only one initial Google tag should send the page-load view');
assert.strictEqual(parameter(googleTags[0], 'tagId').value, 'G-0VL37MQ62P');
assert.deepStrictEqual(googleTags[0].firingTriggerId, ['2147479573'], 'Keep the existing initialization trigger');
assert(!parameter(googleTags[0], 'configSettingsTable').list.some((entry) =>
  entry.map.some((field) => field.key === 'parameter' && field.value === 'send_page_view')),
'Initial page-view collection must remain enabled');

for (const event of ['gtm.js', 'gtm.historyChange', 'page_view', 'virtual_page_view_extra', 'contact_form_success_extra']) {
  assert.deepStrictEqual(dispatch({ event }), [], `${event} must not create duplicate or partial-match event hits`);
}
for (const event of ['directory_depth_reached', 'tool_run_error', 'game_session_start', 'game_milestone']) {
  const hits = dispatch({ event });
  assert.strictEqual(hits.length, 1, `${event} must reach exactly one GA4 tag`);
  assert.strictEqual(hits[0].name, event);
}
assert.strictEqual(dispatch({ event: 'contact_form_success' })[0].name, 'generate_lead');

const page = {
  event: 'virtual_page_view',
  page_location: 'https://www.danielshort.me/portfolio',
  page_title: 'Portfolio | Daniel Short',
  page_referrer: 'https://www.danielshort.me/',
  page_id: 'portfolio',
  audience: 'analytics',
  activity_detail: 'previous-tool-activity',
  q: 'private search text',
  analytics_debug: true,
  traffic_type: 'internal'
};
const hits = dispatch(page);
assert.strictEqual(hits.length, 1, 'Completed route changes must emit exactly one hit');
assert.deepStrictEqual(hits[0], {
  name: 'page_view',
  parameters: {
    page_location: page.page_location,
    page_title: page.page_title,
    page_referrer: page.page_referrer,
    page_id: page.page_id,
    audience: page.audience,
    debug_mode: true,
    traffic_type: 'internal'
  }
}, 'Virtual views must carry current route context without stale activity or free-form search data');

for (const data of [page, { ...page, analytics_debug: undefined, traffic_type: undefined }]) {
  const expected = data.analytics_debug ? { debug_mode: true, traffic_type: 'internal' } : {};
  assert.deepStrictEqual(readSettings(googleTags[0], 'configSettingsTable', data), {
    page_id: data.page_id,
    audience: data.audience,
    ...expected
  },
    'Initial page views must classify QA and ordinary visitors consistently');
  for (const event of ['virtual_page_view', 'tool_run_error', 'game_milestone']) {
    const params = dispatch({ ...data, event })[0].parameters;
    assert.strictEqual(params.debug_mode, expected.debug_mode);
    assert.strictEqual(params.traffic_type, expected.traffic_type);
    assert.strictEqual(Object.hasOwn(params, 'debug_mode'), Boolean(data.analytics_debug),
      'Ordinary traffic must omit debug_mode entirely, rather than send false');
  }
}

for (const [field, entities] of [['tagId', container.tag], ['triggerId', container.trigger], ['variableId', container.variable]]) {
  assert.strictEqual(new Set(entities.map((entity) => entity[field])).size, entities.length, `Duplicate ${field}`);
}

let generated;
vm.runInNewContext(fs.readFileSync(path.join(directory, 'generate-activity-container.js'), 'utf8'), {
  __dirname: directory,
  require(name) {
    if (name === 'fs') return { writeFileSync(outputPath, content) { generated = JSON.parse(content); } };
    if (name === 'path') return path;
    throw new Error(`Unexpected generator dependency ${name}`);
  },
  process: { stdout: { write() {} } }
});
assert.deepStrictEqual(generated.containerVersion, container, 'Regenerate the import whenever container source changes');

console.log('GTM container routing, page context, QA classification, and export consistency checks passed.');
