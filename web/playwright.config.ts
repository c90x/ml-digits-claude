import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
	testDir: './tests',
	timeout: 60000,
	use: {
		baseURL: 'http://localhost:4173'
	},
	webServer: {
		command: 'npm run preview -- --port 4173',
		url: 'http://localhost:4173/ml-digits-claude/',
		reuseExistingServer: !process.env.CI,
		stdout: 'pipe',
		timeout: 60000
	},
	projects: [
		{
			name: 'chromium',
			use: { ...devices['Desktop Chrome'] }
		}
	]
});
