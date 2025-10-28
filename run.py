"""
Main runner script that executes our experiment
"""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import random

import numpy as np
from tqdm import tqdm

from draw import CodingAgent
from evaluation import ImageNoveltyAgent


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


def single_iteration(subject: str, history: History, n_shot: int) -> dict | None:
    """
    A single workload iteration: generate an image and evaluate its novelty.
    Can be used as a unit of work in parallel execution.
    Returns the result dict if successful or None if generation/execution failed.
    """
    coding_agent = CodingAgent(model="gpt-5-mini", temperature=1.0, log=True)
    novelty_agent = ImageNoveltyAgent(model="gpt-5-mini", temperature=1.0, log_name="novelty")

    # Get some examples from the history
    examples = history.sample(n_shot, prop=False)

    # Generate code and an image from the coding agent
    code, base64_str = coding_agent.generate(subject, [ex["code"] for ex in examples])
    if base64_str is None:
        return None

    # Evaluate novelty score of generated image
    novelty_output, score = novelty_agent.evaluate(subject, base64_str, examples)

    # Returned generation is the max of the examples + 1
    return {
        "parents": [ex["entry_id"] for ex in examples],
        "code": code,
        "base64_img": base64_str,
        "novelty_score": score,
        "rationale": novelty_output,
        "gen": max([ex["gen"] for ex in examples], default=-1) + 1
    }


def basic_loop(subject: str, iters: int, n_shot: int, history_path: str):
    """
    A basic, non-parallel loop for running the experiment.
    """
    history = History()
    for _ in tqdm(range(iters)):
        results = single_iteration(subject, history, n_shot)
        if results is not None and results["novelty_score"] > 5:
            history.add_entry(**results)
        history.save_history(history_path)
    return history


def parallel_loop(subject: str, iters: int, n_shot: int, n_workers: int, history_path: str) -> History:
    """
    Runs the experiment in parallel.
    Run n_workers iters times prompted with n_shot examples and save to history_path.
    """
    history = History()

    for _ in tqdm(range(iters)):
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            # Set up jobs and hold results in to_add
            futures = [ex.submit(single_iteration, subject, history, n_shot) for _ in range(n_workers)]
            to_add = []
            for fut in as_completed(futures):
                result = fut.result()
                if result is not None and (result["novelty_score"] > 0 or len(history.entries) < n_shot):
                    to_add.append(result)

            # Tag entries with an id and add them to history
            for entry in to_add:
                history.add_entry(**entry)

            # Flush history to disk
            history.save_history(history_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save_path", type=str, required=True, help="Path to save history to")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    parallel_loop(
        "Cat",
        iters=10,
        n_shot=5,
        n_workers=10,
        history_path=args.save_path
    )
