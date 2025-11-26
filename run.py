"""
Main runner script that executes our experiment
"""
from argparse import ArgumentParser
import random

import numpy as np
import pandas as pd
import wandb

from draw.code import Coder
from novelty.direct import ImageNoveltyAgent


def sample_examples(history_df: pd.DataFrame, n_shot: int, n_children: int, threshold: float) -> list[pd.DataFrame]:
    """
    Sample n_children n_shot examples from history_df that are above the novelty threshold.
    Weigh by novelty score.
    Returns a list of DataFrames so the output is dimension: n_children x n_shot
    """
    all_examples = []
    for _ in range(n_children):
        if len(history_df) > 0:
            passed_examples = history_df[history_df["novelty_score"] > threshold]
            weights = passed_examples["novelty_score"] / passed_examples["novelty_score"].sum()
            examples = passed_examples.sample(min(n_shot, len(passed_examples)), weights=weights, replace=False)
            all_examples.append(examples)
        else:
            # If no history, return empty DataFrame as if we sampled 0 examples
            all_examples.append(pd.DataFrame(columns=history_df.columns))
    return all_examples


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
    # Get some parent examples. Only sample ones that are above the novelty threshold. Weigh by novelty score
    all_examples = sample_examples(history_df, n_shot, n_children, threshold)

    # Generate new code with coder
    print("Writing code...")
    new_codes = coder.reproduce_parallel([list(examples["genotype"]) for examples in all_examples])

    # Generate images from code
    print("Generating images...")
    base64_imgs = coder.express_parallel(new_codes)

    # Evaluate novelty of generated images
    print("Evaluating novelty...")
    example_imgs = [examples["phenotype"].tolist() for examples in all_examples]
    example_scores = [examples["novelty_score"].tolist() for examples in all_examples]
    eval_results = evaluator.evaluate_parallel(base64_imgs, example_imgs, example_scores)
    # Eval results come as a list of tuples (score, rationale) -> convert to 2 parallel lists
    novelties, rationales = zip(*eval_results)

    print("Logging results...")
    # Add all entries to history
    to_add = []
    for i in range(n_children):
        examples = all_examples[i]
        new_entry = {
            "entry_id": len(history_df) + i,
            "gen": max(examples["gen"]) + 1 if len(examples) > 0 else 0,
            "parents": sorted(list(examples["entry_id"])),
            "genotype": new_codes[i],
            "phenotype": base64_imgs[i],
            "novelty_score": novelties[i],
            "rationale": rationales[i]
        }
        to_add.append(new_entry)

    to_add = pd.DataFrame(to_add)
    return to_add


def main_loop(
        save_path: str,
        n_iters: int,
        n_shot: int,
        n_children: int,
        threshold: float):

    # wandb initialization
    run = wandb.init(
        entity="evolutionaryai",
        project="draw",
        name=save_path.split("/")[-1].replace(".csv", ""),
        config={
            "save_path": save_path,
            "n_iters": n_iters,
            "n_shot": n_shot,
            "n_children": n_children,
            "threshold": threshold
        }
    )

    artifact = wandb.Artifact(name="sysprompts", type="sysprompt")
    artifact.add_file(local_path="sysprompts/pillow.txt", name="code.txt")
    artifact.add_file(local_path="sysprompts/novelty_winrate.txt", name="novelty.txt")
    run.log_artifact(artifact)

    # Initialize our models and history
    coder = Coder("cat", model="gpt-5", temperature=1.0, max_workers=n_children)
    evaluator = ImageNoveltyAgent("cat", model="gpt-5", temperature=1.0, max_workers=n_children)
    history_df = pd.DataFrame(
        columns=["entry_id", "gen", "parents", "genotype", "phenotype", "novelty_score", "rationale"]
    )

    for i in range(n_iters):
        to_add = run_iteration(n_shot, n_children, threshold, history_df, coder, evaluator)

        history_df = pd.concat([history_df, to_add], ignore_index=True)
        history_df.to_csv(save_path, index=False)

        # Log history df to wandb
        wandb.log({"history_df": wandb.Table(dataframe=history_df, log_mode="MUTABLE")}, step=i)

        non_nan_novelties = history_df["novelty_score"].dropna()
        passed = (non_nan_novelties > threshold).sum()
        print(f"Completed iteration {i+1}/{n_iters}, total entries: {passed}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save_path", type=str, required=True, help="Path to save results to")
    parser.add_argument("--n_iters", type=int, default=10, help="Number of total iterations to run")
    parser.add_argument("--n_shot", type=int, default=5, help="Number of examples to prompt with")
    parser.add_argument("--n_workers", type=int, default=10, help="Number of parallel workers to use")
    parser.add_argument("--threshold", type=float, default=20.0, help="Novelty threshold for selection")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    main_loop(args.save_path, args.n_iters, args.n_shot, args.n_workers, threshold=args.threshold)

    with open("log.txt", "a", encoding="utf-8") as f:
        f.write(f"{args.save_path},{args.n_iters},{args.n_shot},{args.n_workers}\n")
