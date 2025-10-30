"""
Agent responsible for scoring the novelty of generated images.
"""
import re

from agent import Agent


class ImageNoveltyAgent(Agent):
    """
    Agent that evaluates novelty of generated image against some examples.
    """
    def __init__(self, model: str, temperature: float):
        with open("sysprompts/novelty_winrate.txt", "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()
            super().__init__(system_prompt, model, temperature)

    def format_example(self, example: dict) -> list[dict[str, str]]:
        """
        Formats historical example for few-shot prompting.
        Prompts the agent with the previous novelty score and the image.
        """
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": "Here is a previous image example to compare against:"},
                {"type": "image_url", "image_url": f"data:image/png;base64,{example['base64_img']}"}
            ]
        }

    def evaluate(self, subject: str, base64_img: str, examples: list[dict]) -> tuple[str, int]:
        """
        Evaluates image against examples, returning (rationale, score).
        If score parsing fails return a -1 score.
        """
        fewshot_examples = [self.format_example(ex) for ex in examples]
        self.set_history(fewshot_examples)

        prompt = [
            {"type": "text", "text": f"Evaluate the novelty of the following generated image of a {subject}:"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
        ]
        output = self.generate_response(prompt)

        score = re.search(r"\{(\d+)\}", output)
        # If we can parse a score, add to examples and return the score
        if score:
            return output, int(score.group(1))
        else:
            return output, -1  # Indicate that parsing failed
