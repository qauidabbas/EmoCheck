
"""
Stage 2 Synthetic Dataset Generator.
Creates (emotion_vector_string, natural_sentence) pairs
for training the emotion-to-text generator.
"""
import json
import random
from typing import List, Dict, Tuple

from .config import EMOTION_LABELS


class SyntheticEmotionDatasetGenerator:
    """
    Generates synthetic training data for Stage 2.
    Covers: pure emotions, mixed emotions, dominant+secondary,
    varying intensity, neutral states, ambiguous states.
    """

    COMPOUND_WORDS = {
        ("joy", "sadness"):       ["bittersweet", "wistful", "nostalgic"],
        ("sadness", "joy"):       ["bittersweet", "melancholic yet warm", "tinged with nostalgia"],
        ("love", "fear"):         ["vulnerable", "anxiously attached", "lovingly cautious"],
        ("anger", "sadness"):     ["resentful", "wounded", "hurt and frustrated"],
        ("excitement", "nervousness"): ["thrillingly anxious", "exhilarated", "on edge with anticipation"],
        ("relief", "sadness"):    ["bittersweet relief", "heavy-hearted relief", "a weight lifted but wounds remain"],
        ("joy", "fear"):          ["cautiously happy", "hopeful yet wary", "joyful but on guard"],
        ("confusion", "disappointment"): ["bewildered and let down", "lost and disappointed", "disoriented"],
        ("gratitude", "sadness"): ["thankful yet sorrowful", "grateful through tears", "bittersweet gratitude"],
        ("pride", "fear"):        ["proudly nervous", "scared but accomplished", "anxiously proud"],
        ("curiosity", "fear"):    ["cautiously curious", "intrigued but wary", "nervously fascinated"],
        ("love", "sadness"):      ["longing", "heartsick", "lovingly sorrowful"],
        ("anger", "fear"):        ["defensively aggressive", "enraged yet scared", "lashing out from fear"],
        ("amusement", "embarrassment"): ["sheepishly amused", "laughing at oneself", "awkwardly funny"],
        ("joy", "surprise"):      ["delightfully surprised", "pleasantly shocked", "happily caught off guard"],
        ("sadness", "anger"):     ["bitter", "grievously angry", "sorrowfully furious"],
    }

    INTENSITY_DESCRIPTORS = {
        (0.0, 0.2): ["very subtle", "barely noticeable", "faint", "mild"],
        (0.2, 0.4): ["gentle", "soft", "moderate", "quiet"],
        (0.4, 0.6): ["moderate", "noticeable", "clear", "present"],
        (0.6, 0.8): ["strong", "pronounced", "significant", "deep"],
        (0.8, 1.0): ["intense", "overwhelming", "powerful", "very strong"],
    }

    SINGLE_TEMPLATES = [
        "The dominant feeling is {e1}, experienced with {int_desc} intensity.",
        "There is a {int_desc} sense of {e1}.",
        "The emotion here is clearly {e1}, felt {int_desc}ly.",
        "A {int_desc} wave of {e1} is present.",
        "{e1} is the primary emotion, and it feels {int_desc}.",
        "The overall mood is {e1}, with a {int_desc} presence.",
        "One can feel {e1} quite {int_desc}ly here.",
        "This carries a {int_desc} feeling of {e1}.",
        "The tone is unmistakably {e1}, at a {int_desc} level.",
        "A {int_desc} undercurrent of {e1} runs through this.",
    ]

    TWO_DOMINANT_TEMPLATES = [
        "It feels {compound}, with {e1} slightly stronger than {e2}.",
        "The feeling leans toward {e1}, but {e2} is also clearly present.",
        "{e1} is the primary emotion, accompanied by {e2}.",
        "There is a blend of {e1} and {e2}, with {e1} taking the lead.",
        "The mood is mainly {e1}, with undertones of {e2}.",
        "A mix of {e1} and {e2} is felt, though {e1} dominates.",
        "It is {compound}, mostly {e1} with some {e2} underneath.",
        "The heart of the feeling is {e1}, softened by a touch of {e2}.",
        "One feels {e1} most strongly, but {e2} lingers in the background.",
        "{e1} dominates the emotional landscape, with {e2} adding nuance.",
    ]

    TWO_BALANCED_TEMPLATES = [
        "There is an equal mix of {e1} and {e2}.",
        "Both {e1} and {e2} are present in similar measure.",
        "{e1} and {e2} coexist in a balanced way.",
        "The feeling is split between {e1} and {e2}.",
        "It is hard to tell if {e1} or {e2} is stronger; both are present.",
        "A tension between {e1} and {e2} defines the emotional tone.",
        "Neither {e1} nor {e2} dominates; they are equally felt.",
        "The mood is a delicate balance of {e1} and {e2}.",
    ]

    MULTI_TEMPLATES = [
        "A complex mix of {e1}, {e2}, and a touch of {e3}.",
        "The emotions are layered: primarily {e1}, with {e2} and hints of {e3}.",
        "There is {e1} at the core, surrounded by {e2} and a whisper of {e3}.",
        "It feels {compound}, blending {e1}, {e2}, and traces of {e3}.",
        "Multiple emotions are at play: {e1}, {e2}, and {e3}, creating a complex emotional landscape.",
        "The dominant emotion is {e1}, but {e2} and {e3} add depth.",
        "One senses {e1} most strongly, accompanied by {e2} and a hint of {e3}.",
        "The feeling is multifaceted: {e1} leads, with {e2} and {e3} following.",
    ]

    NEUTRAL_TEMPLATES = [
        "The emotional tone is relatively neutral.",
        "There is no strong emotion detected; the mood is calm and even.",
        "The feeling is flat and neutral, without much emotional charge.",
        "Very little emotional content is present here.",
        "The tone is composed and emotionally neutral.",
        "No particular emotion stands out; it feels balanced and calm.",
    ]

    def __init__(self, num_samples: int = 8000):
        self.num_samples = num_samples
        self.data = []

    def _get_intensity_desc(self, intensity: float) -> str:
        for (lo, hi), descs in self.INTENSITY_DESCRIPTORS.items():
            if lo <= intensity < hi:
                return random.choice(descs)
        return random.choice(self.INTENSITY_DESCRIPTORS[(0.4, 0.6)])

    def _get_compound(self, e1: str, e2: str) -> str:
        key = (e1, e2)
        if key in self.COMPOUND_WORDS:
            return random.choice(self.COMPOUND_WORDS[key])
        key_rev = (e2, e1)
        if key_rev in self.COMPOUND_WORDS:
            return random.choice(self.COMPOUND_WORDS[key_rev])
        return f"a blend of {e1} and {e2}"

    def _random_emotion(self, exclude=None) -> str:
        pool = [e for e in EMOTION_LABELS if e != "neutral" and (exclude is None or e not in exclude)]
        return random.choice(pool)

    def _make_input_string(self, emotion_scores: Dict[str, float], intensity: float) -> str:
        parts = [f"{e}={v:.2f}" for e, v in sorted(emotion_scores.items(), key=lambda x: -x[1])]
        return f"emotions: {' '.join(parts)} intensity={intensity:.2f}"

    def _generate_single(self) -> Tuple[str, str]:
        e1 = self._random_emotion()
        score = round(random.uniform(0.4, 0.95), 2)
        intensity = round(random.uniform(0.2, 0.9), 2)
        int_desc = self._get_intensity_desc(intensity)
        template = random.choice(self.SINGLE_TEMPLATES)
        sentence = template.format(e1=e1, int_desc=int_desc)
        emotion_scores = {e1: score}
        for _ in range(random.randint(0, 2)):
            noise_e = self._random_emotion(exclude=[e1])
            emotion_scores[noise_e] = round(random.uniform(0.01, 0.10), 2)
        input_str = self._make_input_string(emotion_scores, intensity)
        return input_str, sentence

    def _generate_two_dominant(self) -> Tuple[str, str]:
        e1 = self._random_emotion()
        e2 = self._random_emotion(exclude=[e1])
        s1 = round(random.uniform(0.45, 0.90), 2)
        s2 = round(random.uniform(0.20, max(0.21, s1 - 0.05)), 2)
        intensity = round(random.uniform(0.3, 0.85), 2)
        compound = self._get_compound(e1, e2)
        template = random.choice(self.TWO_DOMINANT_TEMPLATES)
        sentence = template.format(e1=e1, e2=e2, compound=compound)
        emotion_scores = {e1: s1, e2: s2}
        input_str = self._make_input_string(emotion_scores, intensity)
        return input_str, sentence

    def _generate_two_balanced(self) -> Tuple[str, str]:
        e1 = self._random_emotion()
        e2 = self._random_emotion(exclude=[e1])
        base = round(random.uniform(0.35, 0.70), 2)
        s1 = base
        s2 = round(base + random.uniform(-0.05, 0.05), 2)
        intensity = round(random.uniform(0.3, 0.8), 2)
        template = random.choice(self.TWO_BALANCED_TEMPLATES)
        sentence = template.format(e1=e1, e2=e2)
        emotion_scores = {e1: s1, e2: s2}
        input_str = self._make_input_string(emotion_scores, intensity)
        return input_str, sentence

    def _generate_multi(self) -> Tuple[str, str]:
        e1 = self._random_emotion()
        e2 = self._random_emotion(exclude=[e1])
        e3 = self._random_emotion(exclude=[e1, e2])
        s1 = round(random.uniform(0.45, 0.85), 2)
        s2 = round(random.uniform(0.20, max(0.21, s1 - 0.05)), 2)
        s3 = round(random.uniform(0.05, max(0.06, s2 - 0.02)), 2)
        intensity = round(random.uniform(0.4, 0.9), 2)
        compound = self._get_compound(e1, e2)
        template = random.choice(self.MULTI_TEMPLATES)
        sentence = template.format(e1=e1, e2=e2, e3=e3, compound=compound)
        emotion_scores = {e1: s1, e2: s2, e3: s3}
        input_str = self._make_input_string(emotion_scores, intensity)
        return input_str, sentence

    def _generate_neutral(self) -> Tuple[str, str]:
        intensity = round(random.uniform(0.02, 0.15), 2)
        sentence = random.choice(self.NEUTRAL_TEMPLATES)
        emotion_scores = {"neutral": round(random.uniform(0.7, 0.95), 2)}
        input_str = self._make_input_string(emotion_scores, intensity)
        return input_str, sentence

    def generate(self) -> List[Dict[str, str]]:
        """Generate the full synthetic dataset."""
        generators = [
            (self._generate_single,       0.25),
            (self._generate_two_dominant,  0.25),
            (self._generate_two_balanced,  0.10),
            (self._generate_multi,         0.30),
            (self._generate_neutral,       0.10),
        ]
        data = []
        for gen_fn, proportion in generators:
            count = int(self.num_samples * proportion)
            for _ in range(count):
                inp, out = gen_fn()
                data.append({"input": inp, "target": out})
        random.shuffle(data)
        self.data = data
        print(f"  Generated {len(data)} synthetic training pairs for Stage 2.")
        return data

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.data, f, indent=2)
        print(f"  Saved synthetic dataset to {path}")

    @staticmethod
    def load(path: str) -> List[Dict[str, str]]:
        with open(path, "r") as f:
            return json.load(f)
