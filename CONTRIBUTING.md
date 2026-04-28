# Contributing

Thanks for your interest in improving Aura Score Analyzer.

This repository mixes research-style experimentation, a Streamlit app, and a Colab workflow, so the best contributions are the ones that keep the project understandable, reproducible, and safe to demo.

## Ways To Contribute

- Fix bugs in the Streamlit or Colab experience
- Improve model-loading reliability across TensorFlow and legacy Keras exports
- Improve documentation, setup instructions, and deployment notes
- Refine the UI while preserving the current project behavior
- Improve evaluation, logging, or explainability outputs
- Propose better dataset governance or provenance documentation

## Development Setup

Use Python 3.12 for the smoothest local experience.

```bash
git clone https://github.com/mhirez/Aura-Score-Analyzer.git
cd Aura-Score-Analyzer
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
streamlit run streamlit_app.py
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Project Principles

- Preserve explainability. The transparent multi-model score is a core project goal.
- Keep Colab and Streamlit behavior aligned when changing inference logic.
- Avoid breaking path assumptions for model ZIP files unless the documentation is updated too.
- Be careful with large binaries and datasets. Document them clearly and avoid unnecessary duplication.
- Do not overstate model accuracy or intended use.

## Before Opening A Pull Request

- Make sure the app still starts with `streamlit run streamlit_app.py`
- Keep README instructions consistent with code changes
- Add or update docs when changing model behavior, dataset handling, or deployment
- Keep the repository professional: clear names, small focused commits, and readable explanations

## Pull Request Checklist

- Explain what changed and why
- Include screenshots for visible UI changes when possible
- Mention whether the Colab notebook, Streamlit app, or both were updated
- Call out any changes to model files, dependencies, or dataset assumptions

## Reporting Issues

When opening an issue, please include:

- What you were trying to do
- What environment you used
- The exact error message
- Whether the problem happened in Colab, Streamlit, or both

## Data And Model Assets

Code contributions are covered by the repository license. Dataset files, example images, and model artifacts may carry separate provenance or usage constraints. See [DATASET.md](DATASET.md) and [MODEL_CARD.md](MODEL_CARD.md) before redistributing or reusing those assets.
