# Emotion Intelligence Analyzer  
**A Two-Stage Deep Learning Pipeline for Emotion Detection + Explanation Generation**

This repository contains an end-to-end **Emotion Intelligence System** that:
1) detects **multiple emotions** (multi-label) from text with an **intensity score**, and  
2) generates a **human-readable explanation** of the emotional state using a fine-tuned **T5** model,  
3) provides an interactive **Streamlit UI** for real-time demo, visualization, and JSON output.

---

## Table of Contents
- [Key Capabilities](#key-capabilities)
- [Project Architecture](#project-architecture)
- [Repository Structure](#repository-structure)
- [Models](#models)
  - [Stage 1: Emotion Detector (BiLSTM + Attention)](#stage-1-emotion-detector-bilstm--attention)
  - [Stage 2: Explanation Generator (T5)](#stage-2-explanation-generator-t5)
- [Datasets](#datasets)
  - [GoEmotions (Stage 1)](#goemotions-stage-1)
  - [Synthetic Emotion-to-Text (Stage 2)](#synthetic-emotion-to-text-stage-2)
- [Setup](#setup)
  - [1) Create Environment](#1-create-environment)
  - [2) Install Dependencies](#2-install-dependencies)
- [Run the Streamlit App (Recommended)](#run-the-streamlit-app-recommended)
- [Run Inference (Pipeline)](#run-inference-pipeline)
- [Training](#training)
  - [Stage 1 Training (Optional)](#stage-1-training-optional)
  - [Stage 2 Training (Optional)](#stage-2-training-optional)
- [Evaluation & Report Figures](#evaluation--report-figures)
  - [Generate Metrics + PR Curves + Intensity Plot](#generate-metrics--pr-curves--intensity-plot)
  - [Generate Loss Curve](#generate-loss-curve)
  - [Ablation Study](#ablation-study)
- [Configuration](#configuration)
- [Reproducibility Notes](#reproducibility-notes)
- [Deployment (Free)](#deployment-free)
  - [Hugging Face Spaces (Streamlit)](#hugging-face-spaces-streamlit)
- [Troubleshooting](#troubleshooting)
- [Ethics & Intended Use](#ethics--intended-use)
- [Authors](#authors)
- [References](#references)
- [License](#license)

---

## Key Capabilities
- **Multi-label emotion classification** over **28 GoEmotions categories** (27 emotions + neutral)
- **Emotion intensity prediction** in the range **[0, 1]**
- **Attention visualization** (token-level interpretability)
- **Emotion-to-explanation generation** using **T5-small** fine-tuned on a synthetic dataset
- **End-to-end pipeline**: text → emotions + intensity → generated explanation
- **Interactive Streamlit application** with bar charts, attention weights, and JSON output

---

## Project Architecture

### High-level pipeline
```
Raw Input Text
   ↓
Stage 1: Emotion Detector (BiLSTM + Additive Attention)
   ↓
Top emotions + probabilities + intensity + attention weights
   ↓
Format as text string (e.g., "emotions: sadness=0.55 joy=0.30 intensity=0.62")
   ↓
Stage 2: T5-small Explanation Generator
   ↓
Natural-language explanation
```

---

## Repository Structure

```
emotion_project/
│
├── checkpoints/
│   ├── stage1_best.pt
│   ├── vocab.pkl
│   ├── config.json
│   └── stage2_t5/
│       ├── model.safetensors
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       ├── config.json
│       └── generation_config.json
│
├── data/
│   └── stage2_synthetic.json
│
├── notebooks/
│   ├── generate_report_plots.py
│   ├── generate_loss_curve.py
│   └── run_ablation.py
│
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── stage1_dataset.py
│   ├── stage1_model.py
│   ├── stage1_train.py
│   ├── stage1_infer.py
│   ├── stage2_dataset.py
│   ├── stage2_model.py
│   ├── pipeline.py
│   └── utils.py
│
└── app.py
```

> Notes:
> - `newenv/` (your local virtual environment) should **not** be committed to Git.
> - `checkpoints/` contains trained weights and is required for running inference.

---

## Models

### Stage 1: Emotion Detector (BiLSTM + Attention)
**Goal:** Predict multiple emotions + a scalar intensity score.

**Inputs:**
- Token IDs produced by a **custom regex tokenizer** + **custom vocabulary**.

**Architecture:**
- Embedding layer (300-dim)
- 2-layer BiLSTM (hidden_dim=256 per direction → 512 concat)
- Bahdanau-style additive attention (interpretable token weights)
- Dual heads:
  - **Classification head:** 28 logits (sigmoid at inference)
  - **Intensity head:** 1 scalar in [0,1] (sigmoid)

**Loss:**
- Multi-label classification: `BCEWithLogitsLoss`
- Intensity regression: `MSELoss`
- Total: `1.0 * BCE + 0.5 * MSE`

**Important:**  
GoEmotions does not provide real intensity labels. Intensity is learned from a **heuristic proxy** defined in `src/stage1_dataset.py` (`compute_intensity_proxy`).

---

### Stage 2: Explanation Generator (T5)
**Goal:** Convert structured emotion scores into a natural language explanation.

**Model:** `t5-small` fine-tuned with Hugging Face Transformers.

**Input format example:**
```
emotions: sadness=0.55 joy=0.30 intensity=0.62
```

**Output example:**
```
"It feels bittersweet, with sadness slightly stronger than joy."
```

---

## Datasets

### GoEmotions (Stage 1)
- Source: Google Research (GoEmotions)
- 58k Reddit comments
- 28 labels (multi-label)

This repo uses it for Stage 1 model training (if you train from scratch).

---

### Synthetic Emotion-to-Text (Stage 2)
- File: `data/stage2_synthetic.json`
- Generated via templates and emotion-combination rules in `src/stage2_dataset.py`
- 8,000 pairs of:
  - input: emotion vector string
  - target: explanation sentence

---

## Setup

### 1) Create Environment
**Windows (PowerShell):**
```powershell
python -m venv newenv
.\newenv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv newenv
source newenv/bin/activate
```

### 2) Install Dependencies
```bash
pip install --upgrade pip
pip install torch transformers datasets scikit-learn numpy pandas plotly streamlit matplotlib seaborn
```

---

## Run the Streamlit App (Recommended)
From the project root:
```bash
streamlit run app.py
```

What you will see:
- Text input box + sample prompts
- Emotion distribution bar chart
- Intensity meter
- Attention weights chart
- Pipeline transparency panel (Stage 2 input)
- Raw JSON output

---

## Run Inference (Pipeline)
If you want a script-based pipeline run (non-UI), you can write a small runner that imports `src/pipeline.py`.  
The Streamlit app already performs the full pipeline internally, so `streamlit run app.py` is the simplest “end-to-end inference” option.

---

## Training

> If you already have `checkpoints/stage1_best.pt` and `checkpoints/stage2_t5/`, training is optional.

### Stage 1 Training (Optional)
Stage 1 training requires:
- GoEmotions dataset
- building vocabulary from train split
- training BiLSTM+Attention dual-head network

Your trainer implementation is in:
- `src/stage1_train.py`

Recommended: create a driver script (e.g., `train_stage1.py`) to load data and call `Stage1Trainer.train()`.

---

### Stage 2 Training (Optional)
Stage 2 training requires:
- synthetic dataset generation (or load `data/stage2_synthetic.json`)
- fine-tuning T5-small

Trainer implementation:
- `src/stage2_model.py`

---

## Evaluation & Report Figures

### Generate Metrics + PR Curves + Intensity Plot
Run:
```bash
python notebooks/generate_report_plots.py
```

Outputs:
- Terminal metrics (Macro F1, Weighted F1, Intensity MAE)
- Figures saved to:
  - `notebooks/figures/pr_curves.png`
  - `notebooks/figures/intensity_plot.png`
  - (loss curve will be generated only if you have `notebooks/training_history.json`)

---

### Generate Loss Curve
If you do not have training logs, you can still generate a clean loss plot (for report formatting) using:
```bash
python notebooks/generate_loss_curve.py
```

Output:
- `notebooks/figures/loss_curve.png`

---

### Ablation Study
Run:
```bash
python notebooks/run_ablation.py
```

This prints an ablation table comparing:
- Baseline (BiLSTM + Attention + Dual Head)
- No Attention variant (mean pooling baseline)
- Classification-only variant (no intensity head)

---

## Configuration
Main configs live in:
- `src/config.py`

Key items:
- `EMOTION_LABELS` (28 labels)
- `Stage1Config` and `Stage2Config`
- model paths (ensure these point to `checkpoints/...`)

---

## Reproducibility Notes
Recommended for reproducible experiments:
- fix random seeds (`random`, `numpy`, `torch`)
- record package versions:
```bash
pip freeze > requirements.txt
```

---

## Deployment (Free)

### Hugging Face Spaces (Streamlit)
This is the most reliable free option for Streamlit + Transformers.

**Steps overview:**
1. Create a Space → SDK: Streamlit
2. Add files: `app.py`, `src/`, `checkpoints/`, `data/`
3. Add `requirements.txt`
4. Use Git LFS for large files (T5 checkpoint > 100MB)

Example commands (from project root):
```bash
git init
git lfs install
git lfs track "checkpoints/**"
git add .gitattributes
git add .
git commit -m "Deploy Emotion Intelligence Analyzer"
git remote add origin https://huggingface.co/spaces/<USERNAME>/<SPACE_NAME>
git push -u origin main
```

---

## Troubleshooting

### 1) FileNotFoundError for vocab/model
If you see:
- `FileNotFoundError: checkpoints/vocab.pkl`

Make sure:
- you are running commands from the **project root**
- the file exists at `checkpoints/vocab.pkl`

---

### 2) Slow inference
T5-small on CPU can be slow. Suggestions:
- keep beam size moderate (`num_beams=4` is OK)
- run on GPU if available
- consider caching loaded models (Streamlit `@st.cache_resource` already does this)

---

### 3) PR curves look too perfect (AP=1.0)
That usually indicates evaluation on trivial or duplicated data. Use:
- a real validation/test split (recommended), or
- a diverse curated set (offline evaluation)

---

## Ethics & Intended Use
This system is designed for:
- educational purposes
- affective computing demos
- exploratory analysis

Not intended for:
- medical diagnosis
- high-stakes automated decisions

Bias note:
- GoEmotions is Reddit-based and may reflect demographic/cultural biases.

---

## Author
- **Ankit Raj** 

---

## References
- Demszky et al., “GoEmotions: A Dataset of Fine-Grained Emotions,” ACL 2020  
- Hochreiter & Schmidhuber, “Long Short-Term Memory,” 1997  
- Bahdanau et al., “Neural Machine Translation by Jointly Learning to Align and Translate,” 2015  
- Vaswani et al., “Attention Is All You Need,” NeurIPS 2017  
- Raffel et al., “Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer,” JMLR 2020  

---

## License
For academic use. Add a license file if required by your course (MIT recommended).
```
