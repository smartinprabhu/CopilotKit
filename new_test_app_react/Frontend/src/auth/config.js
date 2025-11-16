import { env } from '../../env.js';

const ConfigData = {
    API_URL: env.VITE_API_URL && env.MODE !== 'development' ? env.VITE_API_URL : 'http://localhost:8081',
};
export default ConfigData;