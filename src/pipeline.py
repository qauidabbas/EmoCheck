
"""
Full end-to-end pipeline: Text → Emotion Detection → Explanation Generation.
"""
from collections import OrderedDict
from typing import Dict, List, Any

from .stage1_infer import Stage1Predictor
from .stage2_model import Stage2T5Trainer


def format_emotion_for_stage2(stage1_output: Dict[str, Any],
                               top_k: int = 5,
                               threshold: float = 0.05) -> str:
    """
    Convert Stage 1 structured output into a text string for Stage 2.
    Example: "emotions: sadness=0.55 joy=0.45 confusion=0.30 intensity=0.52"
    """
    top_emotions = stage1_output["top_emotions"]
    intensity = stage1_output["intensity"]

    relevant = [e for e in top_emotions if e["confidence"] >= threshold][:top_k]

    if not relevant:
        return f"emotions: neutral=0.90 intensity={intensity:.2f}"

    parts = [f"{e['emotion']}={e['confidence']:.2f}" for e in relevant]
    return f"emotions: {' '.join(parts)} intensity={intensity:.2f}"


class EmotionIntelligencePipeline:
    """
    Complete two-stage pipeline:
      Text → Stage 1 (emotion detection) → Stage 2 (explanation generation)
    """
    def __init__(self, stage1_predictor: Stage1Predictor,
                 stage2_generator: Stage2T5Trainer,
                 top_k: int = 5,
                 threshold: float = 0.05):
        self.stage1 = stage1_predictor
        self.stage2 = stage2_generator
        self.top_k = top_k
        self.threshold = threshold

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Full pipeline inference.
        Input: raw text string
        Output: emotions + intensity + natural language explanation
        """
        # Stage 1: detect emotions
        s1_output = self.stage1.predict(text, top_k=self.top_k)

        # Format for Stage 2
        s2_input = format_emotion_for_stage2(
            s1_output, top_k=self.top_k, threshold=self.threshold
        )

        # Stage 2: generate explanation
        explanation = self.stage2.generate(s2_input)

        # Build clean output
        predicted_emotions = OrderedDict()
        for e in s1_output["top_emotions"]:
            if e["confidence"] >= self.threshold:
                predicted_emotions[e["emotion"]] = e["confidence"]

        return {
            "input_text": text,
            "predicted_emotions": dict(predicted_emotions),
            "top_emotions": s1_output["top_emotions"],
            "all_emotion_scores": s1_output["all_emotion_scores"],
            "intensity": s1_output["intensity"],
            "attention_weights": s1_output["attention_weights"],
            "stage2_input": s2_input,
            "explanation": explanation,
        }

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Run full pipeline on a list of texts."""
        return [self.analyze(t) for t in texts]
