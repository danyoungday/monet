"""
Main runner script that executes our experiment
"""
from argparse import ArgumentParser
import random

import numpy as np
import pandas as pd

from draw.code import Coder
from novelty.direct import ImageNoveltyAgent


def run_iteration(
        n_shot: int,
        n_children: int,
        threshold: float,
        history_df: pd.DataFrame,
        coder: Coder,
        evaluator: ImageNoveltyAgent) -> pd.DataFrame:
    """
    Run a single iteration of the novelty search loop.
    Samples n_shot parents from history_df n_children times to produce n_children new children.
    Processes children into images
    Evaluates novelty based on the parents.
    Returns a DataFrame of the entries that passed.
    """
    # Get some parent examples
    all_examples = [history_df.sample(min(n_shot, len(history_df)), random_state=42) for _ in range(n_children)]

    # Generate new code with coder
    print("Writing code...")
    new_codes = coder.reproduce_parallel([list(examples["genotype"]) for examples in all_examples])

    # Generate images from code
    print("Generating images...")
    base64_imgs = coder.express_parallel(new_codes)

    # Evaluate novelty of generated images
    print("Evaluating novelty...")
    novelty_examples = [examples.to_dict(orient="records") for examples in all_examples]
    novelties = evaluator.evaluate_parallel(base64_imgs, novelty_examples)

    print("Logging results...")
    # Add new entries to history
    to_add = []
    for i in range(n_children):
        if novelties[i] is not None and novelties[i] > threshold:
            examples = all_examples[i]
            new_entry = {
                "entry_id": len(history_df) + i,
                "gen": max(examples["gen"]) + 1 if len(examples) > 0 else 0,
                "parents": sorted(list(examples["entry_id"])),
                "genotype": new_codes[i],
                "phenotype": base64_imgs[i],
                "novelty_score": novelties[i]
            }
            to_add.append(new_entry)

    to_add = pd.DataFrame(to_add)
    return to_add


def main_loop(save_path: str, n_iters: int, n_shot: int, n_children: int):
    coder = Coder("cat", model="gpt-5", temperature=1.0, max_workers=n_children)
    evaluator = ImageNoveltyAgent("cat", model="gpt-5", temperature=1.0, max_workers=n_children)
    history_df = pd.DataFrame(columns=["entry_id", "gen", "parents", "genotype", "phenotype", "novelty_score"])

    for i in range(n_iters):
        to_add = run_iteration(n_shot, n_children, 0.0, history_df, coder, evaluator)
        history_df = pd.concat([history_df, to_add], ignore_index=True)
        history_df.to_csv(save_path, index=False)
        print(f"Completed iteration {i+1}/{n_iters}, total entries: {len(history_df)}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save_path", type=str, required=True, help="Path to save results to")
    parser.add_argument("--n_iters", type=int, default=10, help="Number of total iterations to run")
    parser.add_argument("--n_shot", type=int, default=5, help="Number of examples to prompt with")
    parser.add_argument("--n_workers", type=int, default=10, help="Number of parallel workers to use")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    main_loop(args.save_path, args.n_iters, args.n_shot, args.n_workers)
