(() => {
  'use strict';

  const getDefaultRedirect = () => {
    try {
      const origin = (window.location && window.location.origin) ? String(window.location.origin) : '';
      if (!origin || origin === 'null') return 'https://danielshort.me/tools/dashboard';
      return `${origin}/tools/dashboard`;
    } catch {
      return 'https://danielshort.me/tools/dashboard';
    }
  };

  window.TOOLS_AUTH_CONFIG = window.TOOLS_AUTH_CONFIG || {
    cognitoDomain: 'job-tracker-auth-886623862678.auth.us-east-2.amazoncognito.com',
    cognitoClientId: '78oo663obb0t28u63u9bqn00o9',
    cognitoRedirect: getDefaultRedirect(),
    cognitoScopes: 'openid email profile'
  };
})();
