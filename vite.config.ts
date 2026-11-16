import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    // Research artifacts and captured logs must not reload an active experiment.
    watch: { ignored: ['**/docs/**', '**/examples/**', '**/test-results/**', '**/playwright-report/**', '**/backend/**', '**/assets/**', '**/.venv/**'] },
    proxy: { '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true } },
  },
});
