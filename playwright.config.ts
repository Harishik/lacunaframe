import { defineConfig } from '@playwright/test';
export default defineConfig({
  testDir: 'tests', timeout: 120000, workers: 1,
  use: { baseURL: 'http://127.0.0.1:5173', headless: true, trace: 'retain-on-failure' },
  webServer: [
    { command: '.venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000', url: 'http://127.0.0.1:8000/api/demos', reuseExistingServer: true, timeout: 120000 },
    { command: 'npm run dev -- --port 5173 --strictPort', url: 'http://127.0.0.1:5173', reuseExistingServer: true, timeout: 30000 },
  ],
});
