import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";
import { componentTagger } from "lovable-tagger";
import replace from '@rollup/plugin-replace';

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  server: {
    host: true, // Allow access from all hosts
    port: 3000, // Keep the port as 3000
    allowedHosts: ["all"], // Allow all hosts
    proxy: {
      // Proxy API requests to backend server to avoid CORS issues
      '/api': {
        target: 'http://localhost:8081',
        changeOrigin: true,
        secure: false,
        rewrite: (path) => path.replace(/^\/api/, '/api'),
      },
      '/api/v2': {
        target: 'http://localhost:8081',
        changeOrigin: true,
        secure: false,
        rewrite: (path) => path.replace(/^\/api\/v2/, '/api/v2'),
      },
    },
  },
   build: {
    rollupOptions: {
      plugins: [
        replace({
          'import.meta.env.VITE_API_URL': process.env.REACT_APP_API_URL,
        }),
      ],
    },
  },
  plugins: [
    react(),
    mode === 'development' && componentTagger(),
  ].filter(Boolean),
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
}));
