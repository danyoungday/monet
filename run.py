from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import random

from tqdm import tqdm

from draw import CodingAgent
from evaluation import ImageNoveltyAgent


class History():
    def __init__(self):
        self.entries = []

    def add_entry(self, code: str, base64_img: str, novelty_score: int, rationale: str, gen: int):
        self.entries.append({
            "code": code,
            "base64_img": base64_img,
            "novelty_score": novelty_score,
            "rationale": rationale,
            "gen": gen
        })

    def save_history(self, filename: str):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.entries, f, indent=4)

    def load_history(self, filename: str):
        with open(filename, "r", encoding="utf-8") as f:
            self.entries = json.load(f)

    def sample(self, n: int) -> list[dict]:
        return random.sample(self.entries, min(n, len(self.entries)))


def single_iteration(history: History, n_shot: int) -> dict | None:
    coding_agent = CodingAgent(model="gpt-5-mini", temperature=1.0, log=True)
    novelty_agent = ImageNoveltyAgent(model="gpt-5-mini", temperature=1.0, log_name="novelty")

    # Get some examples from the history
    examples = history.sample(n_shot)

    # Generate code and an image from the coding agent
    code, base64_str = coding_agent.generate("Draw a cat.", [ex["code"] for ex in examples])
    if base64_str is None:
        return None

    # Evaluate novelty score of generated image
    novelty_output, score = novelty_agent.evaluate(base64_str, [ex["base64_img"] for ex in examples])

    # Returned generation is the max of the examples + 1
    return {
        "code": code,
        "base64_img": base64_str,
        "novelty_score": score,
        "rationale": novelty_output,
        "gen": max([ex["gen"] for ex in examples], default=-1) + 1
    }


def basic_loop(iters: int, n_shot: int, history_path: str):
    history = History()
    for _ in tqdm(range(iters)):
        results = single_iteration(history, n_shot)
        if results is not None and results["novelty_score"] > 5:
            history.add_entry(**results)
        history.save_history(history_path)
    return history


def parallel_loop(iters: int, n_shot: int, n_workers: int, history_path: str) -> History:

    history = History()

    for _ in tqdm(range(iters)):
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(single_iteration, history, n_shot) for _ in range(n_workers)]
            for fut in as_completed(futures):
                to_add = []
                result = fut.result()
                if result is not None and (result["novelty_score"] > 5 or len(history.entries) < n_shot):
                    to_add.append(result)

                for entry in to_add:
                    history.add_entry(**entry)

            history.save_history(history_path)


if __name__ == "__main__":
    parallel_loop(iters=10, n_shot=3, n_workers=10, history_path="parallel_history.jsonl")
