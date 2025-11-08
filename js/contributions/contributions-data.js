/* Public-facing docs organized by category
   Update/extend this array – only the data lives here!
   ─────────────────────────────────────────────────────────── */

const MONTH_LOOKUP = {
  january: { quarter: 'Q1' },
  february: { quarter: 'Q1' },
  march: { quarter: 'Q1' },
  april: { quarter: 'Q2' },
  may: { quarter: 'Q2' },
  june: { quarter: 'Q2' },
  july: { quarter: 'Q3' },
  august: { quarter: 'Q3' },
  september: { quarter: 'Q3' },
  october: { quarter: 'Q4' },
  november: { quarter: 'Q4' },
  december: { quarter: 'Q4' }
};

function inferYear(text = ''){
  const match = text.match(/(20\d{2})/);
  return match ? match[1] : 'Earlier';
}

function inferQuarter(text = ''){
  const lower = text.toLowerCase();
  for (const [name, meta] of Object.entries(MONTH_LOOKUP)){
    if (lower.includes(name)) return meta.quarter;
  }
  return null;
}

function annotateItems(items = [], defaults = {}){
  return items.map(item => {
    const entry = { ...defaults, ...item };
    const combined = `${entry.title ?? ''} ${entry.role ?? ''}`;
    entry.year = entry.year ?? inferYear(combined);
    entry.quarter = entry.quarter ?? inferQuarter(entry.title ?? '');
    if (!entry.focus && defaults.focus) entry.focus = defaults.focus;
    return entry;
  });
}

function yearRange(items){
  const numericYears = items
    .map(item => parseInt(item.year, 10))
    .filter(Number.isFinite)
    .sort((a, b) => a - b);
  if (!numericYears.length) return '';
  const first = numericYears[0];
  const last = numericYears[numericYears.length - 1];
  return first === last ? `${first}` : `${first}–${last}`;
}

function formatNavSummary(items, noun = 'contribution'){
  const count = items.length;
  const label = `${noun}${count === 1 ? '' : 's'}`;
  const range = yearRange(items);
  return range ? `${count} ${label} · ${range}` : `${count} ${label}`;
}

function annotateSection(section){
  const { metaDefaults = {}, navNoun = 'contribution', ...rest } = section;
  const items = annotateItems(rest.items, metaDefaults);
  return {
    ...rest,
    navSummary: rest.navSummary || formatNavSummary(items, navNoun),
    items
  };
}

const RAW_CONTRIBUTIONS = [
  {
    id: 'flagship-reports',
    type: 'reports',
    shortTitle: 'Reports',
    heroSummary: 'Budget, economic outlook, and accomplishment docs for leadership.',
    heading:'Flagship Reports & Economic Outlook',
    desc   :'Flagship reports where I lead modeling, revenue tracking, and narrative data summaries that inform Visit Grand Junction, city finance, and statewide partners.',
    impact : {
      title  : 'Budget & statewide planning insights',
      summary: 'Synthesized lodging tax, economic outlook, and visitation trends to guide capital planning and Visit Grand Junction strategy.',
      metrics: [
        { label:'Pages contributed', value:'30+' },
        { label:'Data sources', value:'STR, VisaVue, city finance' },
        { label:'Span', value:'2024–2025' }
      ]
    },
    download: {
      label: 'Download report index (ZIP)',
      description: 'Link index + key talking points',
      file: 'documents/contributions/downloads/flagship-reports-pack.zip',
      includeLinkIndex: true,
      summaryNotes: [
        'Colorado Business Economic Outlook · revenue + visitation modeling',
        'City budget pages · tourism KPIs and scenario planning',
        'Annual accomplishment report · analytics narrative'
      ]
    },
    navNoun: 'report',
    metaDefaults: { focus: 'Budget & economic planning' },
    items  : [
      { title:'Colorado Business Economic Outlook · 2025',
        role :'Contributions: Pages 122–129 - Data aggregation and analysis',
        link :'https://www.colorado.edu/business/brd/colorado-business-economic-outlook-forum',
        pdf  :'documents/Leeds_2025_Colorado_Business_Economic_Outlook.pdf' 
      },
      { title:'Grand Junction City Budget · 2025',
        role :'Contributions: Pages 199–214 - Data aggregation and analysis',
        link :'https://www.gjcity.org/DocumentCenter/View/14655',
        pdf  :'documents/GJ_2025_Budget.pdf' 
      },
      { title:'Visit Grand Junction Annual Accomplishments · 2024',
        role :'Contributions: Data aggregation and analysis',
        link :'https://visitgj.com/4hcADGt',
        pdf  :'documents/VGJ_2024_Accomplishments.pdf' 
      }
    ]
  },

  {
    id: 'council-briefings',
    type: 'council',
    shortTitle: 'Council Briefings',
    heroSummary: 'Standing analyst briefings covering pacing, lodging tax, and campaign results.',
    heading: 'City Council Briefings',
    desc   : 'Visit Grand Junction sections delivered to City Council with lodging tax pacing, forecast signals, and marketing performance insights.',
    impact : {
      title  : 'Standing City Council analyst',
      summary: 'Briefed council leadership on pacing, campaign impact, and strategic risks across tourism and economic development.',
      metrics: [
        { label:'Briefings delivered', value:'30+' },
        { label:'Cadence', value:'Bi-weekly & ad-hoc' },
        { label:'Focus', value:'Lodging tax · campaign ROI' }
      ]
    },
    download: {
      label: 'Download briefing index (ZIP)',
      description: 'Link index + recap notes',
      file: 'documents/contributions/downloads/council-briefings-pack.zip',
      includeLinkIndex: true,
      summaryNotes: [
        'Includes Visit Grand Junction talking points used in council meetings',
        'Each entry references tax, pacing, or campaign analytics delivered live'
      ]
    },
    navNoun: 'briefing',
    metaDefaults: { focus: 'Tourism pacing & tax revenue' },
    items  : [
      { title:'October 20, 2025 Council Briefing',
        role :'Contributions: Visit Grand Junction section',
        link :'https://ccbrief.my.canva.site/city-council-briefing-oct-20-2025' 
      },
      { title:'October 7, 2025 Council Briefing',
        role :'Contributions: Visit Grand Junction section',
        link :'https://ccbrief.my.canva.site/council-briefing-oct-07-2025' 
      },
      { title:'September 22, 2025 Council Briefing',
        role :'Contributions: Visit Grand Junction section',
        link :'https://ccbrief.my.canva.site/city-council-briefing-sept-22-2025' 
      },
      { title:'June 6, 2025 Council Briefing',
        role :'Contributions: Visit Grand Junction section',
        link :'https://ccbrief.my.canva.site/city-council-briefing-june-6-2025' 
      },
      { title:'May 30, 2025 Council Briefing',
        role :'Contributions: Visit Grand Junction section',
        link :'https://ccbrief.my.canva.site/city-council-briefing-may-30-2025' 
      },
      { title:'May 9, 2025 Council Briefing',
        role :'Contributions: Visit Grand Junction section',
        link :'https://ccbrief.my.canva.site/city-council-briefing-may-9-2025' 
      },
      { title:'April 18, 2025 Council Briefing',
        role :'Contributions: Visit Grand Junction section',
        link :'https://ccbrief.my.canva.site/city-council-briefing-april-18-2025' 
      },
      { title:'April 4, 2025 Council Briefing',
        role :'Contributions: Visit Grand Junction section',
        link :'https://ccbrief.my.canva.site/city-council-briefing-april-4-2025' 
      },
      { title:'March 21, 2025 Council Briefing',
        role :'Contributions: Visit Grand Junction section',
        link :'https://ccbrief.my.canva.site/city-council-briefing-march-21-2025' 
      },
      { title:'March 7, 2025 Council Briefing',
        role :'Contributions: Visit Grand Junction section',
        link :'https://ccbrief.my.canva.site/city-council-briefing-march-7' 
      },
      { title:'February 21, 2025 Council Briefing',
        role :'Contributions: Visit Grand Junction section',
        link :'https://ccbrief.my.canva.site/city-council-briefing-february-21' 
      },
      { title:'February 7, 2025 Council Briefing',
        role :'Contributions: Visit Grand Junction section',
        link :'https://ccbrief.my.canva.site/city-council-briefing-february-7' 
      },
      { title:'January 17, 2025 Council Briefing',
        role :'Contributions: Visit Grand Junction section',
        link :'https://ccbrief.my.canva.site/city-council-briefing-january-17' 
      },
      { title:'December 20, 2024 Council Briefing',
        role :'Contributions: Visit Grand Junction section',
        link :'https://ccbrief.my.canva.site/city-council-briefing-december-20' 
      },
      { title:'November 25, 2024 Council Briefing',
        role :'Contributions: Visit Grand Junction section',
        link :'https://ccbrief.my.canva.site/city-council-briefing-november-25' 
      },
      { title:'November 12, 2024 Council Briefing',
        role :'Contributions: Visit Grand Junction section',
        link :'https://ccbrief.my.canva.site/city-council-briefing-november-12' 
      },
      { title:'October 21, 2024 Council Briefing',
        role :'Contributions: Visit Grand Junction section',
        link :'https://ccbrief.my.canva.site/city-council-briefing-october-21' 
      },
      { title:'October 14, 2024 Council Briefing',
        role :'Contributions: Visit Grand Junction section',
        link :'https://ccbrief.my.canva.site/city-council-briefing-october-14' 
      },
      { title:'September 23, 2024 Council Briefing',
        role :'Contributions: Visit Grand Junction section',
        link :'https://ccbrief.my.canva.site/city-council-briefing-september-23' 
      },
      { title:'September 16, 2024 Council Briefing',
        role :'Contributions: Visit Grand Junction section',
        link :'https://ccbrief.my.canva.site/city-council-briefing-september-13' 
      },
      { title:'August 26, 2024 Council Briefing',
        role :'Contributions: Visit Grand Junction section',
        link :'https://ccbrief.my.canva.site/city-council-briefing-august-26' 
      },
      { title:'August 12, 2024 Council Briefing',
        role :'Contributions: Visit Grand Junction section',
        link :'https://www.gjcity.org/DocumentCenter/View/13425/City-Council-Briefing-August-12-2024' 
      },
      { title:'July 22, 2024 Council Briefing',
        role :'Contributions: Visit Grand Junction section',
        link :'https://ccbrief.my.canva.site/city-council-briefing-july-22' 
      },
      { title:'July 8, 2024 Council Briefing',
        role :'Contributions: Visit Grand Junction section',
        link :'https://ccbrief.my.canva.site/city-council-briefing-july-8' 
      },
      { title:'June 10, 2024 Council Briefing',
        role :'Contributions: Visit Grand Junction section',
        link :'https://ccbrief.my.canva.site/city-council-briefing-june-10' 
      },
      { title:'May 17, 2024 Council Briefing',
        role :'Contributions: Visit Grand Junction section',
        link :'https://ccbrief.my.canva.site/city-council-briefing-may-17' 
      },
      { title:'May 3, 2024 Council Briefing',
        role :'Contributions: Visit Grand Junction section',
        link :'https://ccbrief.my.canva.site/city-council-briefing-may-3' 
      },
      { title:'April 19, 2024 Council Briefing',
        role :'Contributions: Visit Grand Junction section',
        link :'https://ccbrief.my.canva.site/april19' 
      },
      { title:'April 5, 2024 Council Briefing',
        role :'Contributions: Visit Grand Junction section',
        link :'https://new.express.adobe.com/webpage/6y0JHWFDI5biO' 
      },
      { title:'March 22, 2024 Council Briefing',
        role :'Contributions: Visit Grand Junction section',
        link :'https://new.express.adobe.com/webpage/y51YVfnRLHb8d' 
      },
      { title:'March 8, 2024 Council Briefing',
        role :'Contributions: Visit Grand Junction section',
        link :'https://new.express.adobe.com/webpage/gvosfJu0N5ywV' 
      }
    ]
  },

  {
    id: 'stakeholder-enews',
    type: 'newsletter',
    shortTitle: 'Stakeholder eNews',
    heroSummary: 'Industry newsletters with pacing dashboards, lodging tax, and automation work.',
    heading:'Visit Grand Junction Stakeholder eNewsletters',
    desc   :'Industry newsletters where I publish analytics dashboards, automation-driven pacing, and tourism insights for partners.',
    impact : {
      title  : 'Stakeholder analytics publisher',
      summary: 'Delivered recurring Industry Data Reports, pacing dashboards, and lodging insights that inform hotels and attractions.',
      metrics: [
        { label:'Stakeholder reach', value:'Hospitality & civic partners' },
        { label:'Newsletter cadence', value:'Monthly/quarterly' },
        { label:'Focus', value:'Automation · pacing · conversions' }
      ]
    },
    download: {
      label: 'Download newsletter index (ZIP)',
      description: 'Link index + highlight reel',
      file: 'documents/contributions/downloads/stakeholder-enews-pack.zip',
      includeLinkIndex: true,
      summaryNotes: [
        'Industry Data Report automation and pacing dashboards',
        'Revenue, lodging, and website performance features'
      ]
    },
    navNoun: 'newsletter',
    metaDefaults: { focus: 'Stakeholder analytics' },
    items  : [
      { title:'Stakeholder eNewsletter · October 2025',
        role :'Contributions: Industry Data Report',
        link :'https://us4.campaign-archive.com/?e=18b7bff0b8&u=d69163b71ce34ec42d130a6a4&id=665ac728f5' 
      },
      { title:'Stakeholder eNewsletter · September 2025',
        role :'Contributions: Industry Data Report',
        link :'https://us4.campaign-archive.com/?e=18b7bff0b8&u=d69163b71ce34ec42d130a6a4&id=d94612ebc7' 
      },
      { title:'Stakeholder eNewsletter · April 2025',
        role :'Contributions: Industry Data Report, Elizabeth\'s GJ Communiqué, Lodging Cannibalization Part 2',
        link :'https://us4.campaign-archive.com/?e=18b7bff0b8&u=d69163b71ce34ec42d130a6a4&id=16d85d0a8e' 
      },
      { title:'Stakeholder eNewsletter · March 2025',
        role :'Contributions: Industry Data Report, Elizabeth\'s GJ Communiqué, Lodging Cannibalization',
        link :'https://us4.campaign-archive.com/?e=18b7bff0b8&u=d69163b71ce34ec42d130a6a4&id=76eef051b3' 
      },
      { title:'Stakeholder eNewsletter · February 2025',
        role :'Contributions: Industry Data Report',
        link :'https://us4.campaign-archive.com/?e=ab674ce95e&u=d69163b71ce34ec42d130a6a4&id=1590378e4c' 
      },
      { title:'Stakeholder eNewsletter · January 2025',
        role :'Contributions: Lodging Tax Revenue Report',
        link :'https://us4.campaign-archive.com/?e=ab674ce95e&u=d69163b71ce34ec42d130a6a4&id=1590378e4c' 
      },
      { title:'Stakeholder eNewsletter · December 2024',
        role :'Contributions: Lodging Tax Revenue Report, Website Performance Data',
        link :'https://us4.campaign-archive.com/?e=f5ce3bd589&u=d69163b71ce34ec42d130a6a4&id=87fdbebaf6' 
      },
      { title:'Stakeholder eNewsletter · November 2024',
        role :'Contributions: Lodging Tax Revenue Report',
        link :'https://us4.campaign-archive.com/?e=f5ce3bd589&u=d69163b71ce34ec42d130a6a4&id=bcba2760e8' 
      },
      { title:'Stakeholder eNewsletter · October 2024',
        role :'Contributions: Lodging Tax Revenue Report, Join the Adventure!',
        link :'https://us4.campaign-archive.com/?e=f5ce3bd589&u=d69163b71ce34ec42d130a6a4&id=2d7388ee76' 
      },
      { title:'Stakeholder eNewsletter · September 2024',
        role :'Contributions: Lodging Tax Revenue Report',
        link :'https://us4.campaign-archive.com/?e=f5ce3bd589&u=d69163b71ce34ec42d130a6a4&id=f9eec9e21f' 
      },
      { title:'Stakeholder eNewsletter · August 2024',
        role :'Contributions: Lodging Tax Revenue Report, Elizabeth\'s GJ Communiqué, Grand Junction Hotel Performance Update',
        link :'https://us4.campaign-archive.com/?e=f5ce3bd589&u=d69163b71ce34ec42d130a6a4&id=f9eec9e21f' 
      },
      { title:'Stakeholder eNewsletter · July 2024',
        role :'Contributions: Lodging Tax Revenue Report, Elizabeth\'s GJ Communiqué',
        link :'https://us4.campaign-archive.com/?e=18b7bff0b8&u=d69163b71ce34ec42d130a6a4&id=19ae68609c' 
      },
      { title:'Stakeholder eNewsletter · June 2024',
        role :'Contributions: Lodging Tax Revenue Report, Elizabeth\'s GJ Communiqué',
        link :'https://us4.campaign-archive.com/?u=d69163b71ce34ec42d130a6a4&id=71c73cdd2b' 
      },
      { title:'Stakeholder eNewsletter · May 2024',
        role :'Contributions: Lodging Tax Revenue Report',
        link :'https://us4.campaign-archive.com/?u=d69163b71ce34ec42d130a6a4&id=71c73cdd2b' 
      },
      { title:'Stakeholder eNewsletter · April 2024',
        role :'Contributions: Lodging Tax Revenue Report',
        link :'https://us4.campaign-archive.com/?e=f5ce3bd589&u=d69163b71ce34ec42d130a6a4&id=cd0cc7edcd' 
      }
    ]
  }
];

const contributions = RAW_CONTRIBUTIONS.map(annotateSection);

if (typeof module !== 'undefined' && module.exports) {
  module.exports = contributions;
}

if (typeof window !== 'undefined') {
  window.contributions = contributions;
}
