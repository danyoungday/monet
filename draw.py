import os
import random
import re

import openai
from tqdm import tqdm


class Agent:
    """
    A simple agent with no history.
    """
    def __init__(self, system_prompt: str, model: str, temperature: float, log_name: str = None):
        self.system_prompt = system_prompt
        self.model = model
        self.temperature = temperature

        self.log_name = log_name
        if log_name:
            with open(f"log/{log_name}.txt", "a", encoding="utf-8") as f:
                f.write("--- New Session ---\n\n")
                f.write(f"System Prompt:\n{system_prompt}\n\n")

    def generate_response(self, prompt: str) -> str:
        """
        Call the OpenAI API to generate a response based on the prompt.
        """
        try:
            messages = [{"role": "system", "content": self.system_prompt}]
            messages.append({"role": "user", "content": prompt})

            response = openai.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=messages
            )
            reply = response.choices[0].message.content

            if self.log_name:
                with open(f"log/{self.log_name}.txt", "a", encoding="utf-8") as f:
                    f.write(f"User:\n{prompt}\n\nResponse:\n{reply}\n\n")

            return reply

        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return "Error generating response."

    def generate_with_examples(self, prompt: str, examples: list[str]) -> str:
        """
        Generate using some few-shot examples.
        """
        full_prompt = prompt
        if len(examples) > 0:
            example_text = "\n\n".join(examples)
            full_prompt += f"\n\nThe following are some previously generated examples:\n\n{example_text}"
        return self.generate_response(full_prompt)


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


class NoveltyAgent(Agent):
    def __init__(self, model: str, temperature: float, log: bool = False):
        with open("sysprompts/novelty_evaluator.txt", "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()
        super().__init__(system_prompt, model, temperature, log_name="novelty" if log else None)

    def evaluate(self, obj_type: str, code: str, examples: list[str]) -> int:
        prompt = f"Evaluate the novelty of the following code that generates a {obj_type}:\n\n{code}"
        output = self.generate_with_examples(prompt, examples)
        score = re.search(r"\{(\d+)\}", output)
        if score:
            return output, int(score.group(1))
        else:
            return output, -1  # Indicate that parsing failed


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

    for _ in tqdm(range(n)):
        examples = sample_storage(3)
        response = coding_agent.generate_with_examples("Draw a cat.", examples)
        code = coding_agent.parse_code(response)

        novelty_agent = NoveltyAgent(model="gpt-5-mini", temperature=1.0, log=True)
        novelty_output, score = novelty_agent.evaluate("cat", code, examples)
        if score > 5:
            save_code(code)


if __name__ == "__main__":
    basic_loop(10)
