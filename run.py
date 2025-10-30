"""
Main runner script that executes our experiment
"""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

import numpy as np
from tqdm import tqdm

from draw import CodingAgent
# from novelty.direct import ImageNoveltyAgent
from novelty.embedding import EmbeddingNoveltyEvaluator
from utils import History


def single_iteration(subject: str,
                     history: History,
                     coding_agent: CodingAgent,
                     novelty_scorer: EmbeddingNoveltyEvaluator,
                     n_shot: int) -> dict | None:
    """
    A single workload iteration: generate an image and evaluate its novelty.
    Can be used as a unit of work in parallel execution.
    Returns the result dict if successful or None if generation/execution failed.
    """
    # Get some examples from the history
    examples = history.sample(n_shot, prop=False)

    # Generate code and an image from the coding agent
    code, base64_str = coding_agent.generate(subject, [ex["code"] for ex in examples])
    if base64_str is None:
        return None

    # Evaluate novelty score of generated image
    score, embedding = novelty_scorer.evaluate(base64_str)

    # Returned generation is the max of the examples + 1
    return {
        "parents": [ex["entry_id"] for ex in examples],
        "code": code,
        "base64_img": base64_str,
        "novelty_score": score,
        "embedding": embedding,
        "rationale": "",
        "gen": max([ex["gen"] for ex in examples], default=-1) + 1
    }


def parallel_loop(subject: str, iters: int, n_shot: int, n_workers: int, history_path: str) -> History:
    """
    Runs the experiment in parallel.
    Run n_workers iters times prompted with n_shot examples and save to history_path.
    """
    coding_agent = CodingAgent(model="gpt-5-mini", temperature=1.0)
    novelty_scorer = EmbeddingNoveltyEvaluator(k=30, device="mps")
    history = History()
    threshold = 0.7

    for _ in tqdm(range(iters)):
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            # Set up jobs and hold results in to_add
            submit = [single_iteration, subject, history, coding_agent, novelty_scorer, n_shot]
            futures = [ex.submit(*submit) for _ in range(n_workers)]
            to_add = []
            for fut in as_completed(futures):
                result = fut.result()
                if result is not None and (result["novelty_score"] > threshold or len(history.entries) < n_shot):
                    to_add.append(result)

            # Tag entries with an id and add them to history
            embeddings = []
            for entry in to_add:
                embedding = entry.pop("embedding")
                history.add_entry(**entry)
                embeddings.append(embedding)
            novelty_scorer.add_embeddings(embeddings)

            # Flush history to disk
            history.save_history(history_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save_path", type=str, required=True, help="Path to save history to")
    parser.add_argument("--iters", type=int, default=10, help="Number of total iterations to run")
    parser.add_argument("--n_shot", type=int, default=5, help="Number of examples to prompt with")
    parser.add_argument("--n_workers", type=int, default=10, help="Number of parallel workers to use")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    parallel_loop(
        "Cat",
        iters=args.iters,
        n_shot=args.n_shot,
        n_workers=args.n_workers,
        history_path=args.save_path
    )
