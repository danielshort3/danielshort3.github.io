(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.WayfarersIcons = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const ATLAS = 'img/wayfarers-guild/ui-icons.png?v=542f9ea5e843';
  const NAMES = ['coins','ore','herbs','provisions','knowledge','maps','notes','crests','starshards','boot','mine','tools','forge','compass','observatory','trail','kitchen','crate','caravan','chest','golden-pickaxe','surveyors-lens','living-crucible','recipe','banner-amber','banner-moon','automation','mastery','miners','meal','backpack','time','equipment','alloys','quill','supplies'];
  const ALIASES = {
    boots:'boot',preparation:'backpack',guild:'crests',crew:'miners',research:'knowledge',journal:'notes',shop:'chest',settings:'automation',garden:'herbs',forage:'herbs',foragers:'herbs',gardeners:'herbs',cooks:'kitchen',scholars:'knowledge',surveyors:'maps',scouts:'compass',mentors:'quill',cartography:'maps',maproom:'maps',hall:'crests',watchtower:'observatory',refit:'notes',charter:'crests',challenge:'equipment',common:'crate',uncommon:'crate',rare:'chest',epic:'surveyors-lens',legendary:'golden-pickaxe',relic:'surveyors-lens',ad:'caravan',gift:'chest',artisan:'tools',scholar:'knowledge',mining:'miners',travel:'boot',alloy:'alloys',survey:'maps','gear-tools':'tools','gear-boots':'boot','gear-instruments':'observatory','auto-work':'automation','auto-forge':'automation','auto-route':'automation','efficient-smelting':'forge','field-notes':'notes','balanced-meals':'provisions','ore-conversion':'alloys','map-survey':'maps','smart-reserve':'crate','specialist-training':'miners','frontier-compass':'compass','meal-none':'provisions','meal-travel':'meal','meal-study':'knowledge','meal-mining':'miners','careful-salvage':'crate','relic-lore':'surveyors-lens','duplicate-study':'recipe',balanced:'compass'
  };
  const escape = value => String(value || '').replace(/[&<>"']/g, character => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[character]));
  Object.assign(ALIASES, { study:'knowledge',shipment:'supplies',surge:'time','prepare-mining':'miners','prepare-travel':'boot','use-kit':'backpack',industry:'tools',expedition:'boot',scholarship:'knowledge',frontier:'compass',discovery:'maps',pace:'boot',supply:'supplies',insight:'knowledge',foundations:'crests',curriculum:'knowledge',waystones:'compass',operations:'automation',routes:'maps','light-pack':'backpack','old-tools':'equipment','quiet-company':'quill','relic-found':'chest','caravan-reward':'caravan' });
  const PORTRAITS = { scout:3,prospector:4,naturalist:10,quartermaster:9,'crew-scholar':8,fox:14,owl:15,tortoise:16 };
  const COLLECTION_ART = ["trail-courier","trail-cartographer","trail-stag","quarry-mole","quarry-hauler","quarry-salamander","tower-scribe","tower-signalist","tower-astronomer","workshop-tinker","workshop-smith","workshop-clockwork","ruins-delver","ruins-restorer","ruins-oracle","harbor-deckhand","harbor-navigator","harbor-leviathan","quarry-pick","clockwork-wrench","survey-hood","captain-hat","porter-coat","scholar-robe","trail-boots","deck-boots"];
  const TRACK_ART = {greenway:['boots','porters','scouts','caravans','waystations','railways'],quarry:['picks','carts','furnace','geology','recovery','deepworks'],watchtower:['beacon','signals','crew','optics','forecasting','relay-grid'],workshop:['assembly','toolmaking','metallurgy','mechanisms','precision','replication'],ruins:['delving','archaeology','recovery-teams','restoration','attunement','resonance'],harbor:['shipbuilding','seamanship','stowage','contracts','navigation','fleet-command']};
  ALIASES.collection = 'banner-moon';
  Object.assign(ALIASES, { 'frost-compass': 'compass', 'archive-quill': 'quill', 'wayfarer-standard': 'banner-amber' });
  Object.assign(ALIASES, { 'refit-pace':'boot','refit-supply':'supplies','refit-insight':'knowledge','legacy-foundations':'crests','legacy-curriculum':'knowledge','legacy-waystones':'compass','standing-orders':'automation','reserve-policy':'crate','purchase-queue':'recipe','purchase-goals':'coins' });
  function markup(id, options) {
    const settings = options || {};
    if (id === 'lock') return '<svg class="wg-icon wg-lock-icon ' + escape(settings.className) + '" viewBox="0 0 24 24" ' + (settings.label ? 'role="img" aria-label="' + escape(settings.label) + '"' : 'aria-hidden="true"') + '><path fill="none" stroke="currentColor" stroke-width="3" d="M7 11V7a5 5 0 0 1 10 0v4"/><path fill="currentColor" d="M4 10h16v12H4z"/><path fill="#102b46" d="M11 14h2v5h-2z"/></svg>';
    const name = ALIASES[id] || id;
    const found = NAMES.indexOf(name);
    const index = found < 0 ? NAMES.indexOf('crate') : found;
    const accessibility = settings.label ? 'role="img" aria-label="' + escape(settings.label) + '"' : 'aria-hidden="true"';
    if (/^station-[a-z0-9-]+$/.test(id)) return '<span class="wg-icon wg-skill-art ' + escape(settings.className) + '" ' + accessibility + ' data-icon="' + escape(id) + '" style="display:inline-block;width:1.75em;height:1.75em;flex-shrink:0;background-image:url(&quot;img/wayfarers-guild/' + escape(id) + '.webp&quot;);background-size:contain;background-position:center;background-repeat:no-repeat;image-rendering:pixelated"></span>';
    if (/^(skill|track)-[a-z0-9-]+$/.test(id)) return '<span class="wg-icon wg-skill-art ' + escape(settings.className) + '" ' + accessibility + ' data-icon="' + escape(id) + '" style="display:inline-block;width:1.75em;height:1.75em;flex-shrink:0;background-image:url(&quot;img/wayfarers-guild/c-' + escape(id) + '.webp&quot;);background-size:contain;background-position:center;background-repeat:no-repeat;image-rendering:pixelated"></span>';
    if (COLLECTION_ART.includes(id) || id === 'cards') {
      const artId = id === 'cards' ? 'trail-courier' : id;
      return '<span class="wg-icon wg-collection-art ' + (id === 'cards' ? 'wg-card-symbol ' : '') + escape(settings.className) + '" ' + accessibility + ' data-icon="' + escape(id) + '" style="display:inline-block;width:1.75em;height:1.75em;flex-shrink:0;background-image:url(&quot;img/wayfarers-guild/' + artId + '.png?v=collections-1&quot;);background-size:contain;background-position:center;background-repeat:no-repeat;image-rendering:pixelated' + (id === 'cards' ? ';border:1px solid currentColor;border-radius:3px' : '') + '"></span>';
    }
    if (Object.prototype.hasOwnProperty.call(PORTRAITS,id)) {
      const frame=PORTRAITS[id];
      return '<span class="wg-icon wg-portrait ' + escape(settings.className) + '" ' + accessibility + ' data-icon="' + escape(id) + '" style="display:inline-block;width:1.75em;height:1.75em;flex-shrink:0;background-image:url(&quot;img/wayfarers-guild/actors.png?v=51f333248522&quot;);background-size:400% 500%;background-position:' + (frame%4*100/3) + '% ' + (Math.floor(frame/4)*25) + '%;background-repeat:no-repeat;image-rendering:pixelated"></span>';
    }
    return '<span class="wg-icon ' + escape(settings.className) + '" ' + accessibility + ' data-icon="' + escape(id) + '" style="display:inline-block;width:1.75em;height:1.75em;flex-shrink:0;background-image:url(&quot;' + ATLAS + '&quot;);background-size:600% 600%;background-position:' + (index % 6 * 20) + '% ' + (Math.floor(index / 6) * 20) + '%;background-repeat:no-repeat;image-rendering:pixelated"></span>';
  }
  return { ATLAS, NAMES, ALIASES, COLLECTION_ART, TRACK_ART, markup };
});
