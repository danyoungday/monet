"""
The main evolution functionality.
"""
import ast
import json
from pathlib import Path
import random

import pandas as pd

from evaluator import Evaluator
from population import Individual
from reproducer import Reproducer


class Evolution:
    """
    Class in charge of running evolution
    """
    def __init__(self, save_path: Path, pop_size: int, n_parents: int, seed_path: Path = None):
        self.save_path = save_path
        self.save_path.mkdir(parents=False, exist_ok=False)
        self.seed_path = seed_path
        self.log_df = pd.DataFrame(columns=["gen", "genotype", "novelty_score", "is_subject"])

        self.pop_size = pop_size
        self.n_parents = n_parents

        self.evaluator = Evaluator()

        self.reproducer = Reproducer()

    def loop(self, population: list[Individual]) -> list[Individual]:
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
        all_parents = [random.sample(population, k=self.n_parents) for _ in range(self.pop_size)]
        children = self.reproducer.reproduce_parallel(all_parents)

        # Evaluate novelty of population + children
        to_eval = population + children
        print("Evaluating novelty...")
        self.evaluator.evaluate_parallel(to_eval)

        # Sort by novelty and select top individuals
        evaluated_sorted = sorted(to_eval, key=lambda c: c.novelty_score, reverse=False)

        # Log the created population
        self.log_df = log_population(self.log_df, evaluated_sorted, self.reproducer, self.evaluator, self.save_path)

        for individual in evaluated_sorted:
            print(f"\t{individual.cand_id}: {individual.novelty_score}")

        new_population = evaluated_sorted[:len(population)]
        print("New population and scores:", [(ind.cand_id, ind.novelty_score) for ind in new_population])

        return new_population

    def evolution(self):
        """
        Overall evolution loop.
        """
        # Initialize population, load from seed if given
        population: list[Individual] = None
        if self.seed_path:
            print("Loading population from log...")
            seed_df = pd.read_csv(self.seed_path)
            seed_df = seed_df[seed_df["gen"] == 0].sort_values("cand_id")
            population = self.reproducer.load_from_log(seed_df)
            # Have to set current_id so we don't have duplicate ids
            self.reproducer.current_id = max(ind.cand_id for ind in population) + 1
        else:
            print("Creating initial population...")
            population = self.reproducer.create_initial_population(n=self.pop_size)
        self.log_df = log_population(self.log_df, population, self.reproducer, self.evaluator, self.save_path)

        # Add initial population to index
        self.evaluator.prepare_candidates(population)
        for individual in population:
            if individual.embedding is not None:
                self.evaluator.index.add_embedding(individual)
        print(f"Added {self.evaluator.index.index.ntotal} individuals to index.")

        # Run evolution
        for gen in range(1, 10):
            print(f"Generation {gen}...")
            population = self.loop(population)


def log_population(
        log_df: pd.DataFrame,
        population: list[Individual],
        reproducer: Reproducer,
        evaluator: Evaluator,
        save_path: Path = None) -> pd.DataFrame:
    """
    Appends the given population to the log dataframe with a new generation number.
    """
    # Update the log dataframe
    gen = log_df["gen"].max() + 1 if not log_df.empty else 0
    rows = []
    for individual in population:
        rows.append({
            "gen": gen,
            "cand_id": individual.cand_id,
            "parents": individual.parents,
            "genotype": individual.genotype,
            "novelty_score": individual.novelty_score,
            "is_subject": individual.is_subject
        })
    new_log_df = pd.concat([log_df, pd.DataFrame(rows)], ignore_index=True) if not log_df.empty else pd.DataFrame(rows)

    # Save the results to disk
    if save_path:
        # Save log_df to results.csv
        new_log_df.to_csv(save_path / "results.csv", index=False)

        # Save agent and discriminator token information to tokens.json
        with open(save_path / "tokens.json", "w", encoding="utf-8") as f:
            json.dump({
                "reproducer_tokens": reproducer.agent.token_usage,
                "discriminator_tokens": evaluator.discriminator.agent.token_usage
            }, f, indent=4)

    return new_log_df


if __name__ == "__main__":
    evolution = Evolution(
        save_path=Path("results/discriminator2"),
        seed_path=Path("results/seed.csv"),
        pop_size=10,
        n_parents=2
    )
    evolution.evolution()
