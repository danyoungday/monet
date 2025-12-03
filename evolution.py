import random

import numpy as np
import pandas as pd
from tqdm import tqdm

from embedder import Embedder
from evaluator import Evaluator
from expresser import Expresser
from population import Individual, Index
from reproducer import Reproducer

POP_SIZE = 10
N_PARENTS = 2


def loop(
        population: list[Individual],
        evaluator: Evaluator,
        reproducer: Reproducer,
        log_df: pd.DataFrame) -> list[Individual]:
    """
    Single evolutionary loop iteration.
    Takes in a population, randomly selects 5 examples to show the reproducer to create 10 new children.
    Evaluates the entire population since we updated the index since last time.
    Selects the top 10 most novel individuals from the combined population + children to be the next population.
    Then adds the new individuals to the index.
    Returns the new population.
    """
    # Select parents and reproduce
    print("Reproducing...")
    all_parents = [random.sample(population, k=N_PARENTS) for _ in range(10)]
    children = reproducer.reproduce_parallel(all_parents)

    # Evaluate novelty of population + children
    to_eval = population + children
    print("Evaluating novelty...")
    evaluator.evaluate_parallel(to_eval)

    # Sort by novelty and select top individuals
    evaluated_sorted = sorted(to_eval, key=lambda c: c.novelty_score, reverse=False)

    # Log the created population
    log_df = log_population(log_df, to_eval)

    for individual in evaluated_sorted:
        print(f"\t{individual.cand_id}: {individual.novelty_score}")

    new_population = evaluated_sorted[:len(population)]
    print("New population and scores:", [(ind.cand_id, ind.novelty_score) for ind in new_population])

    return new_population, log_df


def log_population(log_df: pd.DataFrame, population: list[Individual]) -> pd.DataFrame:
    """
    Appends the given population to the log dataframe with a new generation number.
    """
    gen = log_df["gen"].max() + 1 if not log_df.empty else 0
    rows = []
    for individual in population:
        rows.append({
            "gen": gen,
            "cand_id": individual.cand_id,
            "parents": individual.parents,
            "genotype": individual.genotype,
            "novelty_score": individual.novelty_score
        })
    return pd.concat([log_df, pd.DataFrame(rows)], ignore_index=True)


def evolution():

    log_df = pd.DataFrame(columns=["gen", "genotype", "novelty_score"])

    expresser = Expresser()
    embedder = Embedder(device="mps", batch_size=16)
    index = Index(k=4)
    evaluator = Evaluator(expresser, embedder, index)

    reproducer = Reproducer()

    # Initialize population
    print("Creating initial population...")
    population = reproducer.create_initial_population(n=POP_SIZE)
    log_df = log_population(log_df, population)
    log_df.to_csv("evolution_log.csv", index=False)

    # Add initial population to index
    evaluator.prepare_candidates(population)
    for individual in population:
        if individual.embedding is not None:
            index.add_embedding(individual)

    for gen in range(1, 10):
        print(f"Generation {gen}...")
        population, log_df = loop(population, evaluator, reproducer, log_df)
        log_df.to_csv("evolution_log.csv", index=False)


if __name__ == "__main__":
    evolution()
