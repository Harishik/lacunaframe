import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
// Override when another local service already uses 8000, e.g. LACUNA_API_PORT=8010.
const apiPort = process.env.LACUNA_API_PORT ?? '8000';
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    // Research artifacts and captured logs must not reload an active experiment.
    watch: { ignored: ['**/docs/**', '**/examples/**', '**/test-results/**', '**/playwright-report/**', '**/backend/**', '**/assets/**', '**/.venv/**'] },
    proxy: { '/api': { target: `http://127.0.0.1:${apiPort}`, changeOrigin: true } },
  },
});
