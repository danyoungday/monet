import random

import numpy as np
import pandas as pd
from tqdm import tqdm

from new.embedder import Embedder
from new.evaluator import Evaluator
from new.expresser import Expresser
from new.population import Individual, Index
from new.reproducer import Reproducer


def loop(population: list[Individual], evaluator: Evaluator, reproducer: Reproducer):
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
    n_parents = 2
    all_parents = []
    for _ in range(10):
        parents = random.sample(population, n_parents)
        all_parents.append([parent.genotype for parent in parents])
    child_genotypes = reproducer.reproduce_parallel(all_parents)
    children = [Individual(i, genotype) for i, genotype in enumerate(child_genotypes)]

    # Evaluate novelty of population + children
    to_eval = population + children
    print("Evaluating novelty...")
    novelty_scores = evaluator.evaluate_parallel(to_eval)

    # Sort children by novelty score
    sorted_indices = np.argsort(np.array(novelty_scores))[:len(population)]

    # Add new embeddings to index
    for idx in sorted_indices:
        if idx > len(population) - 1:
            print("Adding child to index")
            candidate = to_eval[idx]
            evaluator.index.add_embedding(candidate)

    new_population = []
    for idx in sorted_indices:
        new_population.append(to_eval[idx])

    return new_population

def log_population(gen: int, population: list[Individual]):
    pop_data = []
    for individual in population:
        pop_data.append({
            "gen": gen,
            "cand_id": individual.cand_id,
            "genotype": individual.genotype,
            "novelty_score": individual.novelty_score
        })
    return pd.DataFrame(pop_data)

def evolution():

    log = pd.DataFrame(columns=["gen", "genotype", "novelty_score"])

    expresser = Expresser()
    embedder = Embedder(device="mps", batch_size=16)
    index = Index(k=3)
    evaluator = Evaluator(expresser, embedder, index)

    reproducer = Reproducer()

    # Initialize population
    pop_size = 5
    initial_genotypes = reproducer.create_initial_population(n=pop_size)
    population = [Individual(i, genotype) for i, genotype in enumerate(initial_genotypes)]
    log = pd.concat([log, log_population(0, population)], ignore_index=True)
    log.to_csv("evolution_log.csv", index=False)

    for gen in range(1, 10):
        print(f"Generation {gen}...")
        population = loop(population, evaluator, reproducer)
        log = pd.concat([log, log_population(gen, population)], ignore_index=True)
        print(f"Saved candidates: {log[log['gen'] == gen]['cand_id'].tolist()}")
        log.to_csv("evolution_log.csv", index=False)

if __name__ == "__main__":
    evolution()