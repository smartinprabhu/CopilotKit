// Access VITE_API_URL from process.env
const VITE_API_URL = import.meta.env.VITE_API_URL;
if (!VITE_API_URL) {
  throw new Error('VITE_API_URL environment variable is not defined');
}
 
const API_CONFIG = {
  baseUrls: [VITE_API_URL.replace(/\/$/, '')], // Use environment variable and remove trailing slash
  endpoints: {
    forecast: "forecast",
    analyze_forecasts: "analyze_forecasts",
  },
  healthcheck: "healthcheck",
};
 
export default API_CONFIG;