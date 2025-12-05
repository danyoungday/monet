"""
Run the baseline experiment. Take a baseline agent and run until it converges.
"""
from pathlib import Path
import random

import pandas as pd

from evaluator import Evaluator
from evolution import log_population
from reproducer import Reproducer

def run_baseline():
    seed_path = Path("results/seed.csv")
    seed_df = pd.read_csv(seed_path)
    seed_df = seed_df[seed_df["gen"] == 0].sort_values("cand_id")

    reproducer = Reproducer()

    population = reproducer.load_from_log(seed_df)
    print(f"Loaded {len(population)} individuals from seed.")

    # Prep evaluator
    evaluator = Evaluator()
    evaluator.prepare_candidates(population)
    for individual in population:
        if individual.embedding is not None:
            evaluator.index.add_embedding(individual)
    reproducer.current_id = max(ind.cand_id for ind in population) + 1
    print(f"Added {evaluator.index.index.ntotal} individuals to index.")

    save_path = Path("results/baseline")
    log_df = pd.DataFrame(columns=["gen", "cand_id", "parents", "genotype", "novelty_score", "is_subject"])
    log_df = log_population(log_df, population, reproducer, evaluator, save_path)

    n_parents = 2
    pop_size = 10
    gens = 10
    for gen in range(1, gens):
        print(f"Generation {gen}...")
        parents = [random.sample(population, k=n_parents) for _ in range(pop_size)]
        print("Reproducing...")
        children = reproducer.reproduce_parallel(parents)

        print("Evaluating...")
        evaluator.evaluate_parallel(children)

        print("Created children:", [(child.cand_id, child.novelty_score) for child in children])
        log_df = log_population(log_df, children, reproducer, evaluator, save_path)


if __name__ == "__main__":
    run_baseline()
