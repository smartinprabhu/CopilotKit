import { Cookies } from 'react-cookie';

const cookies = new Cookies();

type TokenObj = {
  access_token: string;
  refresh_token: string;
  expires_in: number;
};

type UserObj = {
  refresh_token: string;
};

type Credentials = {
  client_key: string;
  client_secret: string;
  account_id: string;
};

type MicrosoftData = {
  client_id: string;
};

type OktaData = {
  client_id: string;
  web_discovery_uri: string;
  web_redirect_uri: string;
};

type AuthServiceType = {
  setToken: (tokenObj: TokenObj) => void;
  setCompany: (count: number) => void;
  setUid: (userObj: UserObj) => void;
  getAccessToken: () => string | undefined;
  getRefreshToken: () => string | undefined;
  getSessionExpiry: () => string | undefined;
  getGatewaySession: () => string | false;
  clearToken: () => void;
  setServerError: () => void;
  getServerError: () => string | undefined;
  clearServerError: () => void;
  setServerErrorCount: (count: string) => void;
  getServerErrorCount: () => string | undefined;
  getUid: () => string | undefined;
  setMicrosoftUid: (userObj: UserObj) => void;
  getMicrosoftUid: () => string | undefined;
  setMicrosoftCredentials: (data: MicrosoftData) => void;
  getMicrosoftCredentials: () => string | undefined;
  setOktaCredentials: (data: OktaData) => void;
  getOktaCredentials: () => string | null;
  setCredentials: (data: Credentials) => void;
  clearCredentials: () => void;
  getCredentials: () => {
    client_id: string | undefined;
    client_secret: string | undefined;
    account_id: string | undefined;
  };
  setGlobalTower: (data: { code: string }) => void;
  getGlobalTower: () => string | undefined;
  getCompanyId: () => number | undefined;
  removeGlobalTower: () => void;
};

const AuthService: AuthServiceType = {
  setToken(tokenObj) {
    const accessTokenExp = new Date();
    const refreshTokenExp = new Date();
    accessTokenExp.setTime(accessTokenExp.getTime() + tokenObj.expires_in * 1000);
    refreshTokenExp.setTime(refreshTokenExp.getTime() + 28800 * 1000);

    cookies.set('accessToken', tokenObj.access_token, {
      path: '/',
      secure: true,
      sameSite: 'strict',
      expires: accessTokenExp,
    });
    cookies.set('refresh_token', tokenObj.refresh_token, {
      path: '/',
      secure: true,
      sameSite: 'strict',
      expires: refreshTokenExp,
    });
  },

  setUid(userObj) {
    const uidExp = new Date();
    uidExp.setTime(uidExp.getTime() + 28800 * 1000);
    cookies.set('uid', userObj.refresh_token, { path: '/', expires: uidExp });
  },

   setCompany(id) {
    const refreshTokenExp = new Date();
    refreshTokenExp.setTime(refreshTokenExp.getTime() + 28800 * 1000);

    cookies.set('company_id', id, {
      path: '/',
      secure: true,
      sameSite: 'strict',
      expires: refreshTokenExp,
    });
  },

  getAccessToken: () => cookies.get('accessToken'),
  getRefreshToken: () => cookies.get('refresh_token'),
  getSessionExpiry: () => cookies.get('sessionExpiry'),

   getCompanyId: () => cookies.get('company_id'),

  getGatewaySession: () => {
    const session = cookies.get('Session');
    return session !== 'undefined' ? session : false;
  },

  clearToken() {
    [
      'accessToken',
      'refresh_token',
      'server_error_count',
      'uid',
      'microsoft_uid',
      'session_id',
      'Session',
      'sessionExpiry',
      'company_id',
      'microsoft_client_id',
    ].forEach((key) => cookies.remove(key, { path: '/' }));
    localStorage.removeItem('issuer');
    localStorage.removeItem('redirect_uri');
    localStorage.removeItem('okta_client_id');
    localStorage.removeItem('TimeZone');
    sessionStorage.clear();
  },

  setServerError: () => cookies.set('server_error', 'true'),
  getServerError: () => cookies.get('server_error'),
  clearServerError: () => cookies.remove('server_error'),
  setServerErrorCount: (count) => cookies.set('server_error_count', count),
  getServerErrorCount: () => cookies.get('server_error_count'),
  getUid: () => cookies.get('uid'),

  setMicrosoftUid(userObj) {
    const exp = new Date();
    exp.setTime(exp.getTime() + 28800 * 1000);
    cookies.set('microsoft_uid', userObj.refresh_token, { path: '/', expires: exp });
  },

  getMicrosoftUid: () => cookies.get('microsoft_uid'),

  setMicrosoftCredentials(data) {
    cookies.set('microsoft_client_id', data.client_id);
  },

  getMicrosoftCredentials: () => cookies.get('microsoft_client_id'),

  setOktaCredentials(data) {
    localStorage.setItem('okta_client_id', data.client_id);
    localStorage.setItem('issuer', data.web_discovery_uri);
    localStorage.setItem('redirect_uri', data.web_redirect_uri);
  },

  getOktaCredentials: () => localStorage.getItem('okta_client_id'),

  setCredentials(data) {
    cookies.set('client_id', data.client_key, { secure: true, sameSite: 'strict' });
    cookies.set('client_secret', data.client_secret, { secure: true, sameSite: 'strict' });
    cookies.set('accountId', data.account_id, { secure: true, sameSite: 'strict' });
  },

  clearCredentials() {
    localStorage.removeItem('api-url');
    cookies.remove('client_id');
    cookies.remove('client_secret');
    cookies.remove('accountId');
  },

  getCredentials: () => ({
    client_id: cookies.get('client_id'),
    client_secret: cookies.get('client_secret'),
    account_id: cookies.get('accountId'),
  }),

  setGlobalTower(data) {
    cookies.set('global_tower', data.code);
  },

  getGlobalTower: () => cookies.get('global_tower'),

  removeGlobalTower: () => cookies.remove('global_tower'),
};

export default AuthService;
