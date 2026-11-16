import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

test('workspace, evaluation and project accessibility', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'Black swan', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Create damaged clip' }).click();
  await page.getByRole('button', { name: 'Run 3 methods' }).click();
  await expect(page.getByText('RECONSTRUCTION COMPLETE')).toBeVisible({ timeout: 90000 });
  for (const name of ['Workspace', 'Evaluation', 'The project']) {
    await page.getByRole('button', { name, exact: true }).click();
    const results = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze();
    expect.soft(results.violations.map(v => ({ id: v.id, help: v.help, nodes: v.nodes.map(n => ({ target: n.target, summary: n.failureSummary })) }))).toEqual([]);
  }
});
