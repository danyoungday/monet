"""
Utilities used throughout the experiment.
"""
import base64
from io import BytesIO
import random

import json
import numpy as np
from PIL import Image


class History():
    """
    Simple history manager for storing a run.
    NOTE: The generation field is 0 if no examples were used to generate the image. Otherwise, it's the max of the
    examples used to generate the image + 1.
    """
    def __init__(self):
        self.entries = []

    def add_entry(self,
                  parents: list[int],
                  code: str,
                  base64_img: str,
                  novelty_score: int,
                  rationale: str,
                  gen: int):
        """
        Simply adds an entry to the history.
        """
        self.entries.append({
            "entry_id": len(self.entries),
            "parents": parents,
            "code": code,
            "base64_img": base64_img,
            "novelty_score": novelty_score,
            "rationale": rationale,
            "gen": gen
        })

    def save_history(self, filename: str):
        """
        Save history as jsonl.
        """
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.entries, f, indent=4)

    def load_history(self, filename: str) -> bool:
        """
        Load a jsonl into history.
        Validates it's in the right format. Returns true if it is, false if we failed.
        """
        with open(filename, "r", encoding="utf-8") as f:
            entries = json.load(f)
            for entry in entries:
                if not all(key in entry for key in ["code", "base64_img", "novelty_score", "rationale", "gen"]):
                    return False
            self.entries = entries
        return True

    def sample(self, n: int, prop: bool) -> list[dict]:
        """
        Samples n entries from history. If there aren't n, returns all entries.
        If prop is true, samples proportionally to novelty score.
        """
        if prop and len(self.entries) > 0:
            total_score = sum([entry["novelty_score"] for entry in self.entries])
            if total_score == 0:
                weights = [1 / len(self.entries) for _ in self.entries]
            else:
                weights = [entry["novelty_score"] / total_score for entry in self.entries]
            return np.random.choice(self.entries, p=weights, replace=False, size=min(n, len(self.entries))).tolist()

        return random.sample(self.entries, min(n, len(self.entries)))


def encode_image(img: Image) -> str:
    """
    Encodes a PIL Image to a base64 string.
    """
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def decode_image(base64_str: str) -> Image:
    """
    Decodes a base64 string to a PIL Image.
    """
    image_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(image_data))
