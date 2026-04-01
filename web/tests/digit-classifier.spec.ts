import { test, expect } from '@playwright/test';

test.describe('Digit Classifier', () => {
	test('page loads and model initializes without error', async ({ page }) => {
		const errors: string[] = [];
		page.on('console', (msg) => {
			if (msg.type() === 'error') {
				errors.push(msg.text());
			}
		});

		await page.goto('/ml-digits-claude/');
		await expect(page.getByRole('heading', { name: 'Digit Classifier' })).toBeVisible();

		// Wait for the model to load (loading message disappears)
		await expect(page.getByText('Loading model...')).not.toBeVisible({ timeout: 30000 });

		// Verify no ONNX loading errors were reported
		const onnxErrors = errors.filter(
			(e) => e.includes('Failed to load model') || e.includes('MountedFiles')
		);
		expect(onnxErrors).toHaveLength(0);
	});

	test('drawing on canvas triggers a prediction', async ({ page }) => {
		await page.goto('/ml-digits-claude/');

		// Wait for model to load
		await expect(page.getByText('Loading model...')).not.toBeVisible({ timeout: 30000 });

		const canvas = page.locator('canvas');
		await expect(canvas).toBeVisible();

		// Draw a digit on the canvas
		const box = await canvas.boundingBox();
		if (!box) throw new Error('Canvas bounding box not found');

		const cx = box.x + box.width / 2;
		const cy = box.y + box.height / 2;

		await page.mouse.move(cx, cy);
		await page.mouse.down();
		await page.mouse.move(cx, cy - 40, { steps: 10 });
		await page.mouse.move(cx, cy + 40, { steps: 10 });
		await page.mouse.up();

		// After drawing, at least one probability bar should have a non-zero label visible
		await expect(page.locator('.bar-label').first()).toBeVisible({ timeout: 10000 });
	});

	test('clear button resets the prediction', async ({ page }) => {
		await page.goto('/ml-digits-claude/');

		// Wait for model to load
		await expect(page.getByText('Loading model...')).not.toBeVisible({ timeout: 30000 });

		const canvas = page.locator('canvas');
		const box = await canvas.boundingBox();
		if (!box) throw new Error('Canvas bounding box not found');

		// Draw something
		const cx = box.x + box.width / 2;
		const cy = box.y + box.height / 2;
		await page.mouse.move(cx, cy);
		await page.mouse.down();
		await page.mouse.move(cx, cy - 50, { steps: 10 });
		await page.mouse.move(cx, cy + 50, { steps: 10 });
		await page.mouse.up();

		// Wait for prediction
		await expect(page.locator('.bar-label').first()).toBeVisible({ timeout: 10000 });

		// Click clear
		await page.getByRole('button', { name: 'Clear' }).click();

		// Bar labels should disappear after clearing
		await expect(page.locator('.bar-label').first()).not.toBeVisible();
	});
});
