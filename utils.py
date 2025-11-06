"""
Utilities used throughout the experiment.
"""
import base64
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import random

import numpy as np
import pandas as pd
from PIL import Image


class History():
    """
    Simple history manager for storing a run.
    NOTE: The generation field is 0 if no examples were used to generate the image. Otherwise, it's the max of the
    examples used to generate the image + 1.
    """
    def __init__(self):
        self.entries = []

    def add_entry(
            self,
            entry_id: int,
            gen: int,
            parents: list[int],
            genotype: str,
            phenotype: str,
            novelty_score: float):
        """
        Adds an entry to the history.
        """
        self.entries.append({
            "entry_id": entry_id,
            "gen": gen,
            "parents": parents,
            "genotype": genotype,
            "phenotype": phenotype,
            "novelty_score": novelty_score
        })

    def save_history(self, filename: str):
        """
        Save history as pandas dataframe.
        """
        history_df = pd.DataFrame(self.entries)
        history_df.to_csv(filename, index=False)

    def load_history(self, filename: str):
        """
        Load a csv into History.
        TODO: We need to be able to parse the parents
        """
        history_df = pd.read_csv(filename)
        self.entries = []
        for _, row in history_df.iterrows():
            entry = {
                "entry_id": row["entry_id"],
                "gen": row["gen"],
                "parents": row["parents"],
                "genotype": row["genotype"],
                "phenotype": row["phenotype"],
                "novelty_score": row["novelty_score"]
            }
            self.entries.append(entry)

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

def run_fn_parallel(fn, all_kwargs: list, max_workers: int):
    """
    Runs a function in parallel over the given inputs.
    """
    outputs = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(fn, **kwargs) for kwargs in all_kwargs]
        for fut in futures:
            output = fut.result()
            outputs.append(output)
    return outputs

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
