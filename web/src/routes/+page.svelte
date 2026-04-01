<script lang="ts">
	import { onMount } from 'svelte';
	import * as ort from 'onnxruntime-web';

	let canvas: HTMLCanvasElement;
	let ctx: CanvasRenderingContext2D;
	let isDrawing = false;
	let session: ort.InferenceSession | null = null;
	let probabilities: number[] = Array(10).fill(0);
	let predictedDigit: number = -1;
	let modelLoaded = false;
	let hasDrawn = false;

	const CANVAS_SIZE = 280;
	const LINE_WIDTH = 20;

	onMount(async () => {
		ctx = canvas.getContext('2d')!;
		ctx.fillStyle = 'white';
		ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
		ctx.lineCap = 'round';
		ctx.lineJoin = 'round';
		ctx.strokeStyle = 'black';
		ctx.lineWidth = LINE_WIDTH;

		try {
			ort.env.wasm.wasmPaths = '/onnx/';
			session = await ort.InferenceSession.create('/model/digit_classifier.onnx');
			modelLoaded = true;
		} catch (e) {
			console.error('Failed to load model:', e);
		}
	});

	function getPos(e: MouseEvent | TouchEvent): { x: number; y: number } {
		const rect = canvas.getBoundingClientRect();
		const scaleX = CANVAS_SIZE / rect.width;
		const scaleY = CANVAS_SIZE / rect.height;
		if ('touches' in e) {
			return {
				x: (e.touches[0].clientX - rect.left) * scaleX,
				y: (e.touches[0].clientY - rect.top) * scaleY
			};
		}
		return {
			x: (e.clientX - rect.left) * scaleX,
			y: (e.clientY - rect.top) * scaleY
		};
	}

	function startDraw(e: MouseEvent | TouchEvent) {
		e.preventDefault();
		isDrawing = true;
		hasDrawn = true;
		const pos = getPos(e);
		ctx.beginPath();
		ctx.moveTo(pos.x, pos.y);
	}

	function draw(e: MouseEvent | TouchEvent) {
		if (!isDrawing) return;
		e.preventDefault();
		const pos = getPos(e);
		ctx.lineTo(pos.x, pos.y);
		ctx.stroke();
		ctx.beginPath();
		ctx.moveTo(pos.x, pos.y);
	}

	function endDraw() {
		if (!isDrawing) return;
		isDrawing = false;
		ctx.beginPath();
		predict();
	}

	async function predict() {
		if (!session || !hasDrawn) return;

		// Create a temporary canvas to resize to 28x28
		const tempCanvas = document.createElement('canvas');
		tempCanvas.width = 28;
		tempCanvas.height = 28;
		const tempCtx = tempCanvas.getContext('2d')!;

		// Draw the original canvas scaled down
		tempCtx.drawImage(canvas, 0, 0, 28, 28);

		// Get pixel data
		const imageData = tempCtx.getImageData(0, 0, 28, 28);
		const input = new Float32Array(1 * 1 * 28 * 28);

		for (let i = 0; i < 28 * 28; i++) {
			// Convert to grayscale and invert (black digit on white bg -> white digit on black bg)
			const r = imageData.data[i * 4];
			const g = imageData.data[i * 4 + 1];
			const b = imageData.data[i * 4 + 2];
			const gray = (r + g + b) / 3;
			// Invert: white bg (255) -> 0, black stroke (0) -> 1
			input[i] = (255 - gray) / 255.0;
		}

		const tensor = new ort.Tensor('float32', input, [1, 1, 28, 28]);
		const results = await session.run({ input: tensor });
		const output = results['output'].data as Float32Array;

		// Apply softmax
		const maxVal = Math.max(...Array.from(output));
		const exps = Array.from(output).map((v) => Math.exp(v - maxVal));
		const sumExps = exps.reduce((a, b) => a + b, 0);
		probabilities = exps.map((v) => v / sumExps);

		predictedDigit = probabilities.indexOf(Math.max(...probabilities));
	}

	function clearCanvas() {
		ctx.fillStyle = 'white';
		ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
		probabilities = Array(10).fill(0);
		predictedDigit = -1;
		hasDrawn = false;
	}
</script>

<svelte:head>
	<title>ML Digits Classifier</title>
</svelte:head>

<main>
	<h1>Digit Classifier</h1>
	<p class="subtitle">Draw a digit and the neural network will classify it in real-time</p>

	<div class="container">
		<div class="canvas-section">
			<h2>Draw a digit here</h2>
			<div class="canvas-wrapper">
				<canvas
					bind:this={canvas}
					width={CANVAS_SIZE}
					height={CANVAS_SIZE}
					onmousedown={startDraw}
					onmousemove={draw}
					onmouseup={endDraw}
					onmouseleave={endDraw}
					ontouchstart={startDraw}
					ontouchmove={draw}
					ontouchend={endDraw}
				></canvas>
			</div>
			<button onclick={clearCanvas}>Clear</button>
			{#if !modelLoaded}
				<p class="loading">Loading model...</p>
			{/if}
		</div>

		<div class="chart-section">
			<h2>Probability result</h2>
			<div class="chart">
				{#each probabilities as prob, i}
					<div class="bar-group">
						<div class="bar-container">
							<div
								class="bar"
								class:predicted={i === predictedDigit && hasDrawn}
								style="height: {prob * 100}%"
							>
								{#if prob > 0.01 && hasDrawn}
									<span class="bar-label">{prob.toFixed(2)}</span>
								{/if}
							</div>
						</div>
						<span class="digit-label">{i}</span>
					</div>
				{/each}
			</div>
		</div>
	</div>

	<footer>
		<p>The model runs entirely in the browser using ONNX Runtime Web.</p>
	</footer>
</main>

<style>
	:global(body) {
		margin: 0;
		padding: 0;
		background: #0a0a0a;
		color: #e0e0e0;
		font-family:
			-apple-system,
			BlinkMacSystemFont,
			'Segoe UI',
			Roboto,
			sans-serif;
		min-height: 100vh;
	}

	main {
		max-width: 900px;
		margin: 0 auto;
		padding: 2rem 1rem;
		text-align: center;
	}

	h1 {
		font-size: 2.2rem;
		font-weight: 700;
		margin-bottom: 0.3rem;
		color: #ffffff;
	}

	.subtitle {
		color: #888;
		margin-bottom: 2.5rem;
		font-size: 0.95rem;
	}

	.container {
		display: flex;
		gap: 3rem;
		justify-content: center;
		align-items: flex-start;
		flex-wrap: wrap;
		background: #141414;
		border: 1px solid #2a2a2a;
		border-radius: 16px;
		padding: 2.5rem;
	}

	h2 {
		font-size: 1rem;
		font-weight: 600;
		margin-bottom: 1rem;
		color: #ccc;
	}

	.canvas-section {
		display: flex;
		flex-direction: column;
		align-items: center;
	}

	.canvas-wrapper {
		border: 2px solid #333;
		border-radius: 12px;
		overflow: hidden;
		cursor: crosshair;
		touch-action: none;
	}

	canvas {
		display: block;
		width: 260px;
		height: 260px;
		background: white;
		border-radius: 10px;
	}

	button {
		margin-top: 1rem;
		padding: 0.6rem 2.5rem;
		background: #f97316;
		color: white;
		border: none;
		border-radius: 8px;
		font-size: 0.95rem;
		font-weight: 600;
		cursor: pointer;
		transition: background 0.2s;
	}

	button:hover {
		background: #ea580c;
	}

	.loading {
		color: #f97316;
		font-size: 0.85rem;
		margin-top: 0.5rem;
	}

	.chart-section {
		display: flex;
		flex-direction: column;
		align-items: center;
	}

	.chart {
		display: flex;
		gap: 6px;
		align-items: flex-end;
		height: 260px;
	}

	.bar-group {
		display: flex;
		flex-direction: column;
		align-items: center;
		width: 32px;
	}

	.bar-container {
		width: 100%;
		height: 230px;
		display: flex;
		align-items: flex-end;
		justify-content: center;
	}

	.bar {
		width: 100%;
		background: #333;
		border-radius: 4px 4px 0 0;
		transition:
			height 0.2s ease,
			background 0.2s ease;
		position: relative;
		min-height: 2px;
	}

	.bar.predicted {
		background: #f97316;
	}

	.bar-label {
		position: absolute;
		top: -20px;
		left: 50%;
		transform: translateX(-50%);
		font-size: 0.7rem;
		color: #f97316;
		font-weight: 600;
		white-space: nowrap;
	}

	.digit-label {
		margin-top: 6px;
		font-size: 0.85rem;
		color: #999;
		font-weight: 500;
	}

	footer {
		margin-top: 2rem;
		color: #555;
		font-size: 0.85rem;
	}

	@media (max-width: 680px) {
		.container {
			flex-direction: column;
			align-items: center;
			padding: 1.5rem;
		}
	}
</style>
