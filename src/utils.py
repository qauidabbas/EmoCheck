
"""
Visualization and utility functions.
"""
from typing import Dict, List, Any


def print_result_pretty(result: Dict[str, Any]):
    """Pretty-print a full pipeline result."""
    print(f"\n{'═' * 60}")
    print(f"  INPUT: \"{result['input_text']}\"")
    print(f"{'─' * 60}")

    print(f"\n  📊 DETECTED EMOTIONS:")
    for emo, score in result["predicted_emotions"].items():
        bar = "█" * int(score * 30)
        print(f"     {emo:18s} {score:.3f} {bar}")

    print(f"\n  🔥 INTENSITY: {result['intensity']:.3f}")

    attn = result.get("attention_weights", [])
    if attn:
        top_words = sorted(attn, key=lambda x: x["weight"], reverse=True)[:6]
        words_str = ", ".join(f"{w['word']}({w['weight']:.3f})" for w in top_words)
        print(f"\n  🔍 KEY WORDS: {words_str}")

    print(f"\n  🔗 STAGE 2 INPUT: {result.get('stage2_input', 'N/A')}")
    print(f"\n  💬 EXPLANATION: \"{result['explanation']}\"")
    print(f"{'═' * 60}")


def print_stage1_result(result: Dict[str, Any]):
    """Pretty-print Stage 1 output only."""
    print(f"\nInput: \"{result['text']}\"")
    print(f"  Top emotions:")
    for e in result["top_emotions"][:5]:
        bar = "█" * int(e["confidence"] * 30)
        print(f"    {e['emotion']:18s} {e['confidence']:.3f} {bar}")
    print(f"  Intensity: {result['intensity']:.3f}")
    top_attn = sorted(result["attention_weights"],
                      key=lambda x: x["weight"], reverse=True)[:5]
    if top_attn:
        print(f"  Key words: {', '.join(w['word'] for w in top_attn)}")
