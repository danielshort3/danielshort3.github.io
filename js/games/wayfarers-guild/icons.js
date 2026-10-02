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
  ALIASES.collection = 'banner-moon';
  Object.assign(ALIASES, { 'frost-compass': 'compass', 'archive-quill': 'quill', 'wayfarer-standard': 'banner-amber' });
  function markup(id, options) {
    const settings = options || {};
    const name = ALIASES[id] || id;
    const found = NAMES.indexOf(name);
    const index = found < 0 ? NAMES.indexOf('crate') : found;
    const accessibility = settings.label ? 'role="img" aria-label="' + escape(settings.label) + '"' : 'aria-hidden="true"';
    if (Object.prototype.hasOwnProperty.call(PORTRAITS,id)) {
      const frame=PORTRAITS[id];
      return '<span class="wg-icon wg-portrait ' + escape(settings.className) + '" ' + accessibility + ' data-icon="' + escape(id) + '" style="display:inline-block;width:1.75em;height:1.75em;flex-shrink:0;background-image:url(&quot;img/wayfarers-guild/actors.png?v=51f333248522&quot;);background-size:400% 500%;background-position:' + (frame%4*100/3) + '% ' + (Math.floor(frame/4)*25) + '%;background-repeat:no-repeat;image-rendering:pixelated"></span>';
    }
    return '<span class="wg-icon ' + escape(settings.className) + '" ' + accessibility + ' data-icon="' + escape(id) + '" style="display:inline-block;width:1.75em;height:1.75em;flex-shrink:0;background-image:url(&quot;' + ATLAS + '&quot;);background-size:600% 600%;background-position:' + (index % 6 * 20) + '% ' + (Math.floor(index / 6) * 20) + '%;background-repeat:no-repeat;image-rendering:pixelated"></span>';
  }
  return { ATLAS, NAMES, ALIASES, markup };
});
