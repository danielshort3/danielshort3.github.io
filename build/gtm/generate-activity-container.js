const fs = require('fs');
const path = require('path');

const ACCOUNT_ID = '6221119580';
const CONTAINER_ID = '179683921';
const PUBLIC_ID = 'GTM-MX6DNH8L';
const MEASUREMENT_ID = 'G-0VL37MQ62P';

const tagGroups = [
  {
    tagId: '6',
    triggerId: '5',
    name: 'Home Explore Select',
    trigger: '^home_explore_select$',
    eventName: 'home_explore_select'
  },
  {
    tagId: '29',
    triggerId: '15',
    name: 'Audience & Navigation',
    trigger: '^(portfolio_audience_select|hero_cta_click|nav_link_click)$'
  },
  {
    tagId: '30',
    triggerId: '16',
    name: 'Directory Behavior',
    trigger: '^(directory_filter_apply|directory_search|directory_depth_reached|select_content|project_filter_select)$'
  },
  {
    tagId: '31',
    triggerId: '17',
    name: 'Portfolio Depth',
    trigger: '^(case_study_engaged|project_view|multi_project_view|modal_close)$'
  },
  {
    tagId: '32',
    triggerId: '18',
    name: 'Career Intent',
    trigger: '^(resume_cta_click|email_cta_click)$'
  },
  {
    tagId: '33',
    triggerId: '19',
    name: 'Tool Activation',
    trigger: '^(tool_run_start|tool_run_complete|tool_run_error)$'
  },
  {
    tagId: '34',
    triggerId: '20',
    name: 'Tool Value',
    trigger: '^(tool_output_export|tool_session_save)$'
  },
  {
    tagId: '35',
    triggerId: '21',
    name: 'Game Activation',
    trigger: '^game_session_start$'
  },
  {
    tagId: '36',
    triggerId: '22',
    name: 'Contact Intent',
    trigger: '^(contact_intent|contact_modal_open|contact_modal_close|contact_form_validation_error|contact_card_click)$'
  },
  {
    tagId: '37',
    triggerId: '23',
    name: 'Generate Lead',
    trigger: '^contact_form_success$',
    eventName: 'generate_lead'
  },
  {
    tagId: '38',
    triggerId: '24',
    name: 'Chatbot Activation',
    trigger: '^(chatbot_launcher_opened|chatbot_starter_prompts_hidden|chatbot_nudge_shown|chatbot_nudge_opened|chatbot_nudge_dismissed|chatbot_reset|chatbot_question_submit)$'
  },
  {
    tagId: '39',
    triggerId: '25',
    name: 'Chatbot Outcomes',
    trigger: '^(chatbot_response_success|chatbot_response_error|chatbot_response_stopped|chatbot_rate_limited|chatbot_link_click)$'
  },
  {
    tagId: '40',
    triggerId: '26',
    name: 'Site Search',
    trigger: '^(site_search|site_search_result_click)$'
  },
  {
    tagId: '41',
    triggerId: '27',
    name: 'Content & Contributions',
    trigger: '^(see_more_toggle|scroll_depth|contrib_doc_click|contrib_timeline_toggle)$'
  },
  {
    tagId: '42',
    triggerId: '28',
    name: 'Reliability',
    trigger: '^(client_error|contact_form_error)$'
  }
];

const dataLayerVariables = [
  ['7', 'activity_label'],
  ['9', 'activity_detail'],
  ['10', 'activity_value'],
  ['11', 'activity_state'],
  ['12', 'activity_category'],
  ['13', 'page_id'],
  ['14', 'audience']
];

const fingerprint = (offset) => String(1783739000000 + offset);

const makeTag = (group, index) => ({
  accountId: ACCOUNT_ID,
  containerId: CONTAINER_ID,
  tagId: group.tagId,
  name: `GA4 - Event - ${group.name}`,
  type: 'gaawe',
  parameter: [
    {
      type: 'BOOLEAN',
      key: 'sendEcommerceData',
      value: 'false'
    },
    {
      type: 'TEMPLATE',
      key: 'eventName',
      value: group.eventName || '{{Event}}'
    },
    {
      type: 'TEMPLATE',
      key: 'measurementIdOverride',
      value: MEASUREMENT_ID
    },
    {
      type: 'TEMPLATE',
      key: 'eventSettingsVariable',
      value: '{{EVS - Site activity parameters}}'
    }
  ],
  fingerprint: fingerprint(100 + index),
  firingTriggerId: [group.triggerId],
  tagFiringOption: 'ONCE_PER_EVENT',
  monitoringMetadata: { type: 'MAP' },
  consentSettings: { consentStatus: 'NOT_SET' }
});

const makeTrigger = (group, index) => ({
  accountId: ACCOUNT_ID,
  containerId: CONTAINER_ID,
  triggerId: group.triggerId,
  name: `CE - ${group.name}`,
  type: 'CUSTOM_EVENT',
  customEventFilter: [
    {
      type: 'MATCH_REGEX',
      parameter: [
        {
          type: 'TEMPLATE',
          key: 'arg0',
          value: '{{_event}}'
        },
        {
          type: 'TEMPLATE',
          key: 'arg1',
          value: group.trigger
        }
      ]
    }
  ],
  fingerprint: fingerprint(200 + index)
});

const makeDataLayerVariable = ([variableId, dataLayerName], index) => ({
  accountId: ACCOUNT_ID,
  containerId: CONTAINER_ID,
  variableId,
  name: `DLV - ${dataLayerName}`,
  type: 'v',
  parameter: [
    {
      type: 'INTEGER',
      key: 'dataLayerVersion',
      value: '2'
    },
    {
      type: 'BOOLEAN',
      key: 'setDefaultValue',
      value: 'false'
    },
    {
      type: 'TEMPLATE',
      key: 'name',
      value: dataLayerName
    }
  ],
  fingerprint: fingerprint(300 + index),
  formatValue: {}
});

const eventSettingsVariable = {
  accountId: ACCOUNT_ID,
  containerId: CONTAINER_ID,
  variableId: '8',
  name: 'EVS - Site activity parameters',
  type: 'gtes',
  parameter: [
    {
      type: 'LIST',
      key: 'eventSettingsTable',
      list: dataLayerVariables.map(([, name]) => ({
        type: 'MAP',
        map: [
          {
            type: 'TEMPLATE',
            key: 'parameter',
            value: name
          },
          {
            type: 'TEMPLATE',
            key: 'parameterValue',
            value: `{{DLV - ${name}}}`
          }
        ]
      }))
    }
  ],
  fingerprint: fingerprint(400)
};

const containerVersion = {
  path: `accounts/${ACCOUNT_ID}/containers/${CONTAINER_ID}/versions/0`,
  accountId: ACCOUNT_ID,
  containerId: CONTAINER_ID,
  containerVersionId: '0',
  container: {
    path: `accounts/${ACCOUNT_ID}/containers/${CONTAINER_ID}`,
    accountId: ACCOUNT_ID,
    containerId: CONTAINER_ID,
    name: 'www.danielshort.me',
    publicId: PUBLIC_ID,
    usageContext: ['WEB'],
    fingerprint: '1709933027544',
    tagManagerUrl: `https://tagmanager.google.com/#/container/accounts/${ACCOUNT_ID}/containers/${CONTAINER_ID}/workspaces?apiLink=container`,
    features: {
      supportUserPermissions: true,
      supportEnvironments: true,
      supportWorkspaces: true,
      supportGtagConfigs: false,
      supportBuiltInVariables: true,
      supportClients: false,
      supportFolders: true,
      supportTags: true,
      supportTemplates: true,
      supportTriggers: true,
      supportVariables: true,
      supportVersions: true,
      supportZones: true,
      supportTransformations: false
    },
    tagIds: [PUBLIC_ID]
  },
  tag: [
    {
      accountId: ACCOUNT_ID,
      containerId: CONTAINER_ID,
      tagId: '3',
      name: 'GA4 - Google tag - All pages',
      type: 'googtag',
      parameter: [
        {
          type: 'TEMPLATE',
          key: 'tagId',
          value: MEASUREMENT_ID
        }
      ],
      fingerprint: '1783730399914',
      firingTriggerId: ['2147479573'],
      tagFiringOption: 'ONCE_PER_EVENT',
      monitoringMetadata: { type: 'MAP' },
      consentSettings: { consentStatus: 'NOT_SET' }
    },
    ...tagGroups.map(makeTag)
  ],
  trigger: tagGroups.map(makeTrigger),
  variable: [
    ...dataLayerVariables.map(makeDataLayerVariable),
    eventSettingsVariable
  ],
  builtInVariable: [
    { accountId: ACCOUNT_ID, containerId: CONTAINER_ID, type: 'PAGE_URL', name: 'Page URL' },
    { accountId: ACCOUNT_ID, containerId: CONTAINER_ID, type: 'PAGE_HOSTNAME', name: 'Page Hostname' },
    { accountId: ACCOUNT_ID, containerId: CONTAINER_ID, type: 'PAGE_PATH', name: 'Page Path' },
    { accountId: ACCOUNT_ID, containerId: CONTAINER_ID, type: 'REFERRER', name: 'Referrer' },
    { accountId: ACCOUNT_ID, containerId: CONTAINER_ID, type: 'EVENT', name: 'Event' }
  ],
  fingerprint: fingerprint(500),
  tagManagerUrl: `https://tagmanager.google.com/#/versions/accounts/${ACCOUNT_ID}/containers/${CONTAINER_ID}/versions/0?apiLink=version`
};

const output = {
  exportFormatVersion: 2,
  exportTime: new Date().toISOString().replace('T', ' ').replace(/\.\d{3}Z$/, ''),
  containerVersion
};

const outputPath = path.join(__dirname, `${PUBLIC_ID}-activity.json`);
fs.writeFileSync(outputPath, `${JSON.stringify(output, null, 2)}\n`, 'utf8');
process.stdout.write(`${outputPath}\n`);
