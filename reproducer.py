import ast
from concurrent.futures import ThreadPoolExecutor
import re

import pandas as pd

from agent import Agent
from population import Individual


class Reproducer:
    def __init__(self, max_workers: int = 10):

        self.max_workers = max_workers

        with open("sysprompts/pillow.txt", "r", encoding="utf-8") as f:
            system_prompt = f.read()

        self.agent = Agent(system_prompt, "gpt-5", 1.0)

        self.current_id = 0

    def reproduce(self, parents: list[Individual]) -> Individual:
        """
        Produces a child individual given parents.
        Returns the child genotype as a string or None if reproduction failed.
        """
        prompt = "Draw a cat. Here are some previous examples:\n\n"
        parent_genotypes = [parent.genotype for parent in parents]
        parent_ids = [parent.cand_id for parent in parents]
        prompt += "\n\n".join(parent_genotypes)
        response = self.agent.generate_response(prompt)
        # Find the last instance of a code block
        code_blocks = re.findall(r"```(?:python)?\s*(.*?)```", response, re.DOTALL)

        genotype = None
        if code_blocks:
            genotype = code_blocks[-1]

        individual = Individual(self.current_id, parent_ids, genotype)
        self.current_id += 1
        return individual

    def reproduce_parallel(self, parents_list: list[list[Individual]]) -> list[Individual]:
        """
        Produces multiple child genotypes in parallel given lists of parent genotypes.
        NOTE: I think there's technically a race condition on current_id but I hope it's not a big deal
        """
        outputs = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = [ex.submit(self.reproduce, parents) for parents in parents_list]
            for fut in futures:
                output = fut.result()
                outputs.append(output)
        return outputs

    def create_initial_population(self, n: int) -> list[Individual]:
        """
        Create an initial population of n genotypes
        """
        prompt = f"Write {n} unique code blocks that draw a cat."
        response = self.agent.generate_response(prompt)

        code_blocks = re.findall(r"```(?:python)?\s*(.*?)```", response, re.DOTALL)

        # Return last n code blocks
        individuals = []
        for i in range(len(code_blocks) - n, len(code_blocks)):
            genotype = code_blocks[i]
            individual = Individual(self.current_id, [-1], genotype)
            self.current_id += 1
            individuals.append(individual)

        return individuals

    def load_from_log(self, log_df: pd.DataFrame) -> list[Individual]:
        """
        Loads a population from a log dataframe.
        """
        individuals = []
        for _, row in log_df.iterrows():
            individual = Individual(
                cand_id=row["cand_id"],
                parents=ast.literal_eval(row["parents"]),
                genotype=row["genotype"]
            )
            individual.novelty_score = 1.0
            if "is_subject" in row:
                individual.is_subject = row["is_subject"]
            individuals.append(individual)

        return individuals


def test_reproducer():
    """
    Test function to make sure the reproducer works as expected.
    """
    reproducer = Reproducer()
    population = reproducer.create_initial_population(10)
    for i, cand in enumerate(population):
        print(f"Candidate {i}:\n{cand}\n\n")


if __name__ == "__main__":
    test_reproducer()
