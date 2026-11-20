import { defineConfig } from '@playwright/test';
// LACUNA_API_PORT / LACUNA_WEB_PORT avoid reusing an unrelated server on a default port.
const apiPort = process.env.LACUNA_API_PORT ?? '8000';
const webPort = process.env.LACUNA_WEB_PORT ?? '5173';
export default defineConfig({
  testDir: 'tests', timeout: 120000, workers: 1,
  use: { baseURL: `http://127.0.0.1:${webPort}`, headless: true, trace: 'retain-on-failure' },
  webServer: [
    { command: `.venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port ${apiPort}`, url: `http://127.0.0.1:${apiPort}/api/demos`, reuseExistingServer: true, timeout: 120000 },
    { command: `npm run dev -- --port ${webPort} --strictPort`, url: `http://127.0.0.1:${webPort}`, reuseExistingServer: true, timeout: 30000 },
  ],
});
