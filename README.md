# ML Digits Classifier

A handwritten digit classifier (0-9) using a CNN built with PyTorch. Includes a Svelte web interface for real-time inference in the browser via ONNX Runtime Web.

<a href="https://colab.research.google.com/github/c90x/ml-digits-claude/blob/main/model/digit_classifier.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Project Structure

```
model/          # Python ML project (uv)
├── digit_classifier.ipynb   # Training notebook
├── pyproject.toml
└── ...
web/            # Svelte web interface
├── src/
│   └── routes/
│       └── +page.svelte     # Drawing canvas + probability bars
├── static/
│   ├── model/               # ONNX model
│   └── onnx/                # ONNX Runtime WASM files
└── ...
```

## Model

- **Architecture**: CNN with 3 convolutional blocks, batch normalization, and dropout
- **Parameters**: ~438K
- **Test Accuracy**: ~99.5%
- **Export**: ONNX format for browser inference

## Getting Started

### Model Training

```bash
cd model
uv sync
uv run jupyter notebook digit_classifier.ipynb
```

### Web Interface

```bash
cd web
npm install
npm run dev
```

The web app lets you draw a digit on a canvas and see the classification probabilities in real-time, powered entirely by in-browser inference using ONNX Runtime Web.

## Tech Stack

- **ML**: Python, PyTorch, Jupyter
- **Web**: Svelte, SvelteKit, ONNX Runtime Web
- **Package Management**: uv (Python), npm (JavaScript)
