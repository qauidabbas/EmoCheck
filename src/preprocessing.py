
"""
Tokenizer and Vocabulary for Stage 1.
"""
import re
import pickle
from collections import Counter
from typing import List, Dict


class SimpleTokenizer:
    """
    Rule-based tokenizer: lowercase, split on words/numbers/punctuation.
    We avoid subword tokenizers here to keep Stage 1 fully custom.
    """
    _pattern = re.compile(r"[a-z]+|[0-9]+|[^\s\w]")

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return SimpleTokenizer._pattern.findall(text.lower())


class Vocabulary:
    """
    Word-to-index mapping built from a corpus.
    Special tokens: <PAD>=0, <UNK>=1
    """
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    PAD_IDX = 0
    UNK_IDX = 1

    def __init__(self, freq_threshold: int = 2):
        self.freq_threshold = freq_threshold
        self.itos: Dict[int, str] = {0: self.PAD_TOKEN, 1: self.UNK_TOKEN}
        self.stoi: Dict[str, int] = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}

    def build(self, texts: List[str]):
        """Build vocabulary from list of raw text strings."""
        freq = Counter()
        for text in texts:
            tokens = SimpleTokenizer.tokenize(text)
            freq.update(tokens)

        idx = len(self.itos)
        for word, count in freq.most_common():
            if count >= self.freq_threshold and word not in self.stoi:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

        print(f"  Vocabulary built: {len(self.stoi)} words (min_freq={self.freq_threshold})")

    def numericalize(self, text: str) -> List[int]:
        """Convert text to list of token indices."""
        tokens = SimpleTokenizer.tokenize(text)
        return [self.stoi.get(t, self.UNK_IDX) for t in tokens]

    def tokens_from_text(self, text: str) -> List[str]:
        """Get tokens without converting to indices."""
        return SimpleTokenizer.tokenize(text)

    def __len__(self):
        return len(self.stoi)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({
                "stoi": self.stoi,
                "itos": self.itos,
                "freq_threshold": self.freq_threshold
            }, f)
        print(f"  Vocabulary saved to {path}")

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        with open(path, "rb") as f:
            data = pickle.load(f)
        vocab = cls(freq_threshold=data["freq_threshold"])
        vocab.stoi = data["stoi"]
        vocab.itos = data["itos"]
        print(f"  Vocabulary loaded from {path}: {len(vocab)} words")
        return vocab
