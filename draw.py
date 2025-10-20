import os
import random
import re

from tqdm import tqdm

from agent import Agent
from evaluation import ImageNoveltyAgent


class CodingAgent(Agent):
    def __init__(self, model: str, temperature: float, log: bool = False):
        with open("sysprompts/code.txt", "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()
        super().__init__(system_prompt, model, temperature, log_name="code" if log else None)

    def parse_code(self, response: str) -> str:
        """
        Extract code from the agent's response.
        """
        code_match = r"```python\s*(.*?)```"
        matches = re.findall(code_match, response, flags=re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[-1].strip()
        else:
            return None


def save_code(code: str):
    n_in_storage = len(os.listdir("storage"))
    with open(f"storage/draw_{n_in_storage}.py", "w", encoding="utf-8") as f:
        f.write(code)


def sample_storage(n: int) -> list[str]:
    files = os.listdir("storage")
    sampled_files = random.sample(files, min(n, len(files)))
    examples = []
    for filename in sampled_files:
        with open(os.path.join("storage", filename), "r", encoding="utf-8") as f:
            examples.append(f.read().strip())
    return examples


def basic_loop(n: int):
    coding_agent = CodingAgent(model="gpt-5-mini", temperature=1.0, log=True)
    novelty_agent = ImageNoveltyAgent(n_shot=3, model="gpt-5-mini", temperature=1.0, log_name="novelty")

    for _ in tqdm(range(n)):
        examples = sample_storage(3)
        response = coding_agent.generate_with_examples("Draw a cat.", examples)
        code = coding_agent.parse_code(response)

        novelty_output, score = novelty_agent.evaluate(code)
        if score > 5:
            save_code(code)


if __name__ == "__main__":
    basic_loop(10)
