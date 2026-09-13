import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'node:path';

export default defineConfig({
  plugins: [react()],
  resolve: { alias: { '@': path.resolve(__dirname, './src') } },
  server: {
    port: 5173,
    proxy: {
      // Dev proxy so the dashboard talks to the API on one origin and CORS
      // never becomes a demo-day surprise.
      '/api': { target: 'http://localhost:8000', changeOrigin: true },
      '/ws': { target: 'ws://localhost:8000', ws: true },
    },
  },
  build: { outDir: 'dist', sourcemap: true, target: 'es2022' },
  test: { environment: 'jsdom', globals: true },
});
