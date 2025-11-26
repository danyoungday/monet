from concurrent.futures import ThreadPoolExecutor
import re

from agent import Agent


class Reproducer:
    def __init__(self, max_workers: int = 10):

        self.max_workers = max_workers

        with open("sysprompts/pillow.txt", "r", encoding="utf-8") as f:
            system_prompt = f.read()

        self.agent = Agent(system_prompt, "gpt-5", 1.0)

    def reproduce(self, parents: list[str]) -> str:
        """
        Produces a child genotype given parent genotypes.
        Returns the child genotype as a string or None if reproduction failed.
        """
        prompt = "Draw a cat. Here are some previous examples:\n\n"
        prompt += "\n\n".join(parents)
        response = self.agent.generate_response(prompt)
        # Find the last instance of a code block
        code_blocks = re.findall(r"```(?:python)?\s*(.*?)```", response, re.DOTALL)
        if code_blocks:
            return code_blocks[-1]

        return None

    def reproduce_parallel(self, parents_list: list[list[str]]) -> list[str]:
        """
        Produces multiple child genotypes in parallel given lists of parent genotypes.
        """
        outputs = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = [ex.submit(self.reproduce, parents) for parents in parents_list]
            for fut in futures:
                output = fut.result()
                outputs.append(output)
        return outputs


    def create_initial_population(self, n: int) -> list[str]:
        """
        Create an initial population of n genotypes
        """
        prompt = f"Write {n} unique code blocks to draw a cat."
        response = self.agent.generate_response(prompt)

        code_blocks = re.findall(r"```(?:python)?\s*(.*?)```", response, re.DOTALL)

        # Return last n code blocks
        return code_blocks[-10:]


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
