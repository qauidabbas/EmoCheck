"""
============================================================
  EMOTION INTELLIGENCE ANALYZER — Streamlit App
============================================================
  Run with:
      streamlit run app.py
============================================================
"""

import os
import re
import json
import pickle
import warnings
from typing import List, Dict, Any, Tuple
from collections import OrderedDict
from transformers import AutoTokenizer

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers import T5Tokenizer
warnings.filterwarnings("ignore")


# ============================================================
# PATHS — These point to YOUR existing files
# DO NOT CHANGE these if your folder structure matches
# ============================================================

STAGE1_MODEL_PATH = "checkpoints/stage1_best.pt"
VOCAB_PATH        = "checkpoints/vocab.pkl"
CONFIG_PATH       = "checkpoints/config.json"
STAGE2_MODEL_DIR  = "checkpoints/stage2_t5"


# ============================================================
# EMOTION LABELS (same 28 labels from training)
# ============================================================

EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral"
]

NUM_EMOTIONS = len(EMOTION_LABELS)

EMOTION_EMOJIS = {
    "admiration": "🌟", "amusement": "😄", "anger": "😠",
    "annoyance": "😤",  "approval": "👍",  "caring": "🤗",
    "confusion": "😕",  "curiosity": "🤔", "desire": "😍",
    "disappointment": "😞", "disapproval": "👎", "disgust": "🤢",
    "embarrassment": "😳",  "excitement": "🎉",  "fear": "😨",
    "gratitude": "🙏",  "grief": "😭",     "joy": "😊",
    "love": "❤️",       "nervousness": "😰", "optimism": "🌈",
    "pride": "🏆",      "realization": "💡", "relief": "😌",
    "remorse": "😔",    "sadness": "😢",    "surprise": "😲",
    "neutral": "😐",
}

EMOTION_COLORS = {
    "admiration": "#2ecc71", "amusement": "#2ecc71", "approval": "#2ecc71",
    "caring": "#2ecc71",     "excitement": "#2ecc71", "gratitude": "#2ecc71",
    "joy": "#2ecc71",        "love": "#e74c3c",       "optimism": "#2ecc71",
    "pride": "#2ecc71",      "relief": "#2ecc71",     "desire": "#f39c12",
    "surprise": "#3498db",   "realization": "#3498db", "curiosity": "#3498db",
    "confusion": "#3498db",  "nervousness": "#f39c12",
    "anger": "#e74c3c",      "annoyance": "#e74c3c",  "disappointment": "#e74c3c",
    "disapproval": "#e74c3c","disgust": "#e74c3c",    "embarrassment": "#f39c12",
    "fear": "#e74c3c",       "grief": "#e74c3c",      "remorse": "#e74c3c",
    "sadness": "#e74c3c",    "neutral": "#95a5a6",
}


# ============================================================
# MODEL CLASSES (exact same architecture as your training code)
# ============================================================

class SimpleTokenizer:
    """Same tokenizer from your src/preprocessing.py"""
    _pattern = re.compile(r"[a-z]+|[0-9]+|[^\s\w]")

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return SimpleTokenizer._pattern.findall(text.lower())


class Vocabulary:
    """Same vocabulary from your src/preprocessing.py"""
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    PAD_IDX = 0
    UNK_IDX = 1

    def __init__(self, freq_threshold: int = 2):
        self.freq_threshold = freq_threshold
        self.itos = {0: self.PAD_TOKEN, 1: self.UNK_TOKEN}
        self.stoi = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}

    def numericalize(self, text: str) -> List[int]:
        tokens = SimpleTokenizer.tokenize(text)
        return [self.stoi.get(t, self.UNK_IDX) for t in tokens]

    def tokens_from_text(self, text: str) -> List[str]:
        return SimpleTokenizer.tokenize(text)

    def __len__(self):
        return len(self.stoi)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        vocab = cls(freq_threshold=data["freq_threshold"])
        vocab.stoi = data["stoi"]
        vocab.itos = data["itos"]
        return vocab


class AdditiveAttention(nn.Module):
    """Same attention from your src/stage1_model.py"""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_out, mask=None):
        energy = torch.tanh(self.W(lstm_out))
        scores = self.v(energy).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)
        return context, weights


class EmotionDetector(nn.Module):
    """Same model from your src/stage1_model.py"""
    def __init__(self, vocab_size, embed_dim, hidden_dim,
                 num_classes, num_layers, dropout, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        bidir_dim = hidden_dim * 2
        self.attention = AdditiveAttention(bidir_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(bidir_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.intensity_head = nn.Sequential(
            nn.Linear(bidir_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_ids, lengths):
        batch_size, seq_len = input_ids.size()
        mask = (input_ids != 0).float()
        embedded = self.dropout(self.embedding(input_ids))

        packed = pack_padded_sequence(
            embedded, lengths.cpu().clamp(min=1),
            batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(
            lstm_out, batch_first=True, total_length=seq_len
        )
        lstm_out = self.dropout(lstm_out)

        context, attn_weights = self.attention(lstm_out, mask)
        logits = self.classifier(context)
        intensity = self.intensity_head(context).squeeze(-1)
        return logits, intensity, attn_weights


# ============================================================
# MODEL LOADING — Cached so models load only ONCE
# ============================================================

@st.cache_resource
def load_all_models():
    """
    Load all models at once. Cached by Streamlit.
    This runs only on first page load, then stays in memory.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- Load Vocabulary ----------
    vocab = Vocabulary.load(VOCAB_PATH)

    # ---------- Load Config ----------
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    s1_cfg = config.get("stage1", {})

    # ---------- Load Stage 1 Model ----------
    s1_model = EmotionDetector(
        vocab_size=len(vocab),
        embed_dim=int(s1_cfg.get("embed_dim", 300)),
        hidden_dim=int(s1_cfg.get("hidden_dim", 256)),
        num_classes=int(s1_cfg.get("num_classes", 28)),
        num_layers=int(s1_cfg.get("num_lstm_layers", 2)),
        dropout=float(s1_cfg.get("dropout", 0.3)),
        pad_idx=0,
    )

    checkpoint = torch.load(STAGE1_MODEL_PATH, map_location=device)
    s1_model.load_state_dict(checkpoint["model_state_dict"])
    s1_model.to(device)
    s1_model.eval()

    # ---------- Load Stage 2 Model ----------
    if os.path.exists(STAGE2_MODEL_DIR):
        s2_tokenizer = AutoTokenizer.from_pretrained(STAGE2_MODEL_DIR, local_files_only=True)
        s2_model = T5ForConditionalGeneration.from_pretrained(STAGE2_MODEL_DIR, local_files_only=True)
    else:
        s2_tokenizer = AutoTokenizer.from_pretrained("t5-small")
        s2_model = T5ForConditionalGeneration.from_pretrained("t5-small")
    s2_model.to(device)
    s2_model.eval()

    return vocab, s1_model, s2_model, s2_tokenizer, device


# ============================================================
# INFERENCE FUNCTIONS
# ============================================================

def run_stage1(text, model, vocab, device, max_len=128):
    """
    Stage 1: Text → Emotion probabilities + Intensity + Attention
    """
    tokens = vocab.tokens_from_text(text)
    ids = vocab.numericalize(text)[:max_len]
    length = max(len(ids), 1)

    ids_tensor = torch.tensor([ids], dtype=torch.long).to(device)
    len_tensor = torch.tensor([length], dtype=torch.long).to(device)

    with torch.no_grad():
        logits, intensity, attn_weights = model(ids_tensor, len_tensor)

    probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    intensity_val = intensity.item()
    attn = attn_weights.squeeze(0).cpu().numpy()[:length]

    # All 28 scores
    all_scores = {}
    for i in range(NUM_EMOTIONS):
        all_scores[EMOTION_LABELS[i]] = round(float(probs[i]), 4)

    # Sorted
    sorted_emotions = sorted(all_scores.items(), key=lambda x: -x[1])

    # Attention per word
    display_tokens = tokens[:length]
    attn_display = attn[:len(display_tokens)]
    word_attention = [
        {"word": w, "weight": round(float(a), 4)}
        for w, a in zip(display_tokens, attn_display)
    ]

    return {
        "all_scores": all_scores,
        "sorted_emotions": sorted_emotions,
        "intensity": round(intensity_val, 4),
        "attention": word_attention,
        "tokens": display_tokens,
        "attn_values": attn_display.tolist(),
    }


def format_for_stage2(emotion_result, top_k=5, threshold=0.05):
    """
    Convert Stage 1 output → text string for Stage 2 input
    Example: 'emotions: sadness=0.55 joy=0.45 intensity=0.52'
    """
    intensity = emotion_result["intensity"]
    relevant = [
        (emo, score) for emo, score in emotion_result["sorted_emotions"]
        if score >= threshold
    ][:top_k]

    if not relevant:
        return f"emotions: neutral=0.90 intensity={intensity:.2f}"

    parts = [f"{emo}={score:.2f}" for emo, score in relevant]
    return f"emotions: {' '.join(parts)} intensity={intensity:.2f}"


def run_stage2(input_text, model, tokenizer, device):
    """
    Stage 2: Emotion string → Natural language explanation
    """
    input_enc = tokenizer(
        input_text, return_tensors="pt",
        max_length=128, truncation=True
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_enc["input_ids"],
            attention_mask=input_enc["attention_mask"],
            max_length=100,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


# ============================================================
# CHART FUNCTIONS
# ============================================================

def make_bar_chart(sorted_emotions, top_k=8):
    """Horizontal bar chart of top emotions."""
    top = sorted_emotions[:top_k]
    names = [e[0] for e in reversed(top)]
    values = [e[1] * 100 for e in reversed(top)]
    colors = [EMOTION_COLORS.get(e[0], "#3498db") for e in reversed(top)]
    emojis_list = [EMOTION_EMOJIS.get(e[0], "•") for e in reversed(top)]
    labels = [f"{emj} {name}" for emj, name in zip(emojis_list, names)]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker=dict(color=colors, line=dict(width=1, color="#2c3e50")),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(size=13),
    ))

    fig.update_layout(
        xaxis=dict(title="Confidence (%)", range=[0, max(values) * 1.3 + 5]),
        yaxis=dict(tickfont=dict(size=13)),
        height=max(300, top_k * 50),
        margin=dict(l=10, r=40, t=20, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def make_attention_chart(tokens, weights):
    """Bar chart showing attention weight per word."""
    fig = go.Figure(go.Bar(
        x=tokens,
        y=weights,
        marker=dict(
            color=weights,
            colorscale="YlOrRd",
            line=dict(width=1, color="#2c3e50"),
        ),
        text=[f"{w:.3f}" for w in weights],
        textposition="outside",
        textfont=dict(size=10),
    ))

    fig.update_layout(
        xaxis=dict(title="Words", tickangle=45, tickfont=dict(size=11)),
        yaxis=dict(title="Attention Weight"),
        height=320,
        margin=dict(l=40, r=20, t=20, b=80),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ============================================================
# CUSTOM CSS STYLING
# ============================================================

CUSTOM_CSS = """
<style>
    .main-title {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #7f8c8d;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .explanation-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        font-size: 1.15rem;
        line-height: 1.6;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.35);
    }
    .intensity-box {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #e74c3c;
    }
    .pipeline-box {
        background: #2c3e50;
        color: #2ecc71;
        border-radius: 8px;
        padding: 0.7rem 1rem;
        font-family: monospace;
        font-size: 0.82rem;
        margin: 0.5rem 0;
    }
    footer {visibility: hidden;}
</style>
"""


# ============================================================
# MAIN STREAMLIT APP
# ============================================================

def main():

    # ── Page Setup ──
    st.set_page_config(
        page_title="Emotion Intelligence Analyzer",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("## ⚙️ Settings")

        top_k = st.slider("Top-K emotions to show", 3, 15, 7)
        threshold = st.slider("Min confidence (%)", 1, 30, 5) / 100.0
        show_attention = st.checkbox("Show attention weights", value=True)
        show_all_scores = st.checkbox("Show all 28 scores", value=False)

        st.markdown("---")
        st.markdown("## 📝 Try These Examples")

        samples = {
            "😊 Bittersweet":
                "I watched the empty room and smiled, though something felt misplaced.",
            "🎉 Pure Joy":
                "I feel absolutely amazing today! Everything is going perfectly!",
            "😢 Sadness":
                "Nothing matters anymore. Everything is the same.",
            "😠 Frustrated":
                "I don't understand why they would do that to me.",
            "❤️ Love + Fear":
                "I love spending time with her, but I'm scared it won't last.",
            "🎯 Achievement":
                "She finally got the promotion after years of hard work!",
            "😰 Nervous":
                "I'm so nervous about the interview tomorrow.",
            "🙏 Grateful":
                "Thank you so much for helping me, I really appreciate it.",
            "😲 Shocked":
                "I can't believe this happened. I'm shocked and confused.",
            "🌅 Nostalgic":
                "The sunset was beautiful but it reminded me of her.",
        }

        if "sample_text" not in st.session_state:
            st.session_state.sample_text = ""

        for label, text in samples.items():
            if st.button(label, use_container_width=True):
                st.session_state.sample_text = text

        st.markdown("---")
        st.markdown("## ℹ️ About")
        st.markdown("""
        **Stage 1:** Custom BiLSTM + Attention
        trained on GoEmotions (28 emotions)

        **Stage 2:** T5-small fine-tuned
        (transfer learning) for explanation
        """)

        device_name = "GPU 🚀" if torch.cuda.is_available() else "CPU 💻"
        st.caption(f"Running on: {device_name}")

    # ── Main Title ──
    st.markdown(
        '<h1 class="main-title">🧠 Emotion Intelligence Analyzer</h1>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="subtitle">Detect mixed emotions and generate '
        'human-like emotional explanations</p>',
        unsafe_allow_html=True
    )

    # ── Check If Model Files Exist ──
    missing_files = []
    if not os.path.exists(STAGE1_MODEL_PATH):
        missing_files.append(f"Stage 1 model: {STAGE1_MODEL_PATH}")
    if not os.path.exists(VOCAB_PATH):
        missing_files.append(f"Vocabulary: {VOCAB_PATH}")
    if not os.path.exists(CONFIG_PATH):
        missing_files.append(f"Config: {CONFIG_PATH}")
    if not os.path.exists(STAGE2_MODEL_DIR):
        missing_files.append(f"Stage 2 model: {STAGE2_MODEL_DIR}")

    if missing_files:
        st.error("❌ **Model files not found!**")
        for m in missing_files:
            st.code(m)
        st.info("Make sure you run `streamlit run app.py` from your project root folder.")
        st.stop()

    # ── Load Models (only once) ──
    try:
        with st.spinner("🔄 Loading models (first time only)..."):
            vocab, s1_model, s2_model, s2_tokenizer, device = load_all_models()
        st.sidebar.success(f"✅ Models loaded | Vocab: {len(vocab)} words")
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.exception(e)
        st.stop()

    # ── Text Input ──
    default_text = st.session_state.get("sample_text", "")

    text_input = st.text_area(
        "✏️ Enter text to analyze:",
        value=default_text,
        height=100,
        placeholder="Type a sentence... e.g., 'I smiled, but the room felt empty.'",
        max_chars=500,
    )

    # ── Analyze Button ──
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        analyze_clicked = st.button(
            "🔍 Analyze",
            type="primary",
            use_container_width=True
        )

    # ── Run When Button Clicked ──
    if analyze_clicked:

        # Validate
        if not text_input or text_input.strip() == "":
            st.warning("⚠️ Please enter some text first.")
            st.stop()

        clean_text = text_input.strip()

        # ── Run Full Pipeline ──
        with st.spinner("🧠 Analyzing emotions..."):

            # Stage 1
            emotion_result = run_stage1(clean_text, s1_model, vocab, device)

            # Format for Stage 2
            s2_input = format_for_stage2(emotion_result, top_k=top_k, threshold=threshold)

            # Stage 2
            explanation = run_stage2(s2_input, s2_model, s2_tokenizer, device)

        # ────────────────────────────────────────
        #   DISPLAY RESULTS
        # ────────────────────────────────────────

        st.markdown("---")

        # ── Row 1: Key Metrics ──
        filtered = [(e, s) for e, s in emotion_result["sorted_emotions"] if s >= threshold]
        dominant = filtered[0] if filtered else ("neutral", 0.0)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(
                "🏆 Dominant Emotion",
                f"{EMOTION_EMOJIS.get(dominant[0], '')} {dominant[0].title()}",
                f"{dominant[1]*100:.1f}%"
            )
        with m2:
            st.metric("🔥 Intensity", f"{emotion_result['intensity']*100:.0f}%")
        with m3:
            num_active = len(filtered)
            st.metric(
                "🎭 Active Emotions",
                f"{num_active}",
                "mixed" if num_active >= 3 else "focused"
            )

        # ── Row 2: Generated Explanation (MAIN OUTPUT) ──
        st.markdown("")
        st.markdown(
            f'<div class="explanation-box">'
            f'💬 <strong>Emotional Insight:</strong><br><br>'
            f'"{explanation}"'
            f'</div>',
            unsafe_allow_html=True
        )

        # ── Row 3: Emotion Bar Chart ──
        st.markdown("### 📊 Emotion Distribution")
        st.plotly_chart(
            make_bar_chart(emotion_result["sorted_emotions"], top_k=top_k),
            use_container_width=True
        )

        # ── Row 4: Top Emotions with Progress Bars ──
        st.markdown("### 🎭 Top Emotions Breakdown")

        top_to_show = [(e, s) for e, s in emotion_result["sorted_emotions"]
                       if s >= threshold][:top_k]

        if top_to_show:
            for emotion, score in top_to_show:
                emoji = EMOTION_EMOJIS.get(emotion, "•")
                pct = score * 100

                c1, c2, c3 = st.columns([2, 5, 1])
                with c1:
                    st.markdown(f"**{emoji} {emotion.title()}**")
                with c2:
                    st.progress(min(score, 1.0))
                with c3:
                    st.markdown(f"**{pct:.1f}%**")
        else:
            st.info("No emotions above threshold.")

        # ── Row 5: Intensity ──
        st.markdown("### 🔥 Emotional Intensity")

        intensity = emotion_result["intensity"]
        if intensity < 0.25:
            int_label = "Low 🟢"
        elif intensity < 0.50:
            int_label = "Moderate 🟡"
        elif intensity < 0.75:
            int_label = "Strong 🟠"
        else:
            int_label = "Very Strong 🔴"

        st.markdown(
            f'<div class="intensity-box">'
            f'Intensity: <strong>{intensity*100:.0f}%</strong> ({int_label})'
            f'</div>',
            unsafe_allow_html=True
        )
        st.progress(min(intensity, 1.0))

        # ── Row 6: Attention Weights ──
        if show_attention and emotion_result["tokens"]:
            st.markdown("### 🔍 Attention Weights")
            st.caption("Which words the model focused on")
            st.plotly_chart(
                make_attention_chart(
                    emotion_result["tokens"],
                    emotion_result["attn_values"]
                ),
                use_container_width=True
            )

        # ── Row 7: Stage 2 Input Preview ──
        with st.expander("🔗 What Stage 2 received (pipeline transparency)"):
            st.markdown(
                f'<div class="pipeline-box">{s2_input}</div>',
                unsafe_allow_html=True
            )

        # ── Row 8: All 28 Scores ──
        if show_all_scores:
            with st.expander("📋 All 28 Emotion Scores"):
                scores_df = pd.DataFrame([
                    {
                        "Emotion": f"{EMOTION_EMOJIS.get(e, '')} {e}",
                        "Score": round(s, 4),
                        "Percentage": f"{s*100:.2f}%"
                    }
                    for e, s in emotion_result["sorted_emotions"]
                ])
                st.dataframe(scores_df, use_container_width=True, hide_index=True)

        # ── Row 9: JSON Output ──
        with st.expander("🧾 Raw JSON Output"):
            output_json = {
                "input_text": clean_text,
                "predicted_emotions": {
                    e: round(s, 4) for e, s in emotion_result["sorted_emotions"]
                    if s >= threshold
                },
                "intensity": emotion_result["intensity"],
                "explanation": explanation,
                "stage2_input": s2_input,
            }
            st.json(output_json)

        # Clear sample text
        st.session_state.sample_text = ""

    # ── Footer ──
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#95a5a6; font-size:0.8rem;'>"
        "🧠 Stage 1: Custom BiLSTM+Attention | Stage 2: T5 (Transfer Learning)"
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()