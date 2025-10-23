"""
Agent responsible for scoring the novelty of generated images.
"""
import re

from agent import Agent


class ImageNoveltyAgent(Agent):
    """
    Agent that evaluates novelty of generated image against some examples.
    """
    def __init__(self, model: str, temperature: float, log_name: str = None):
        with open("sysprompts/novelty_evaluator.txt", "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()
            super().__init__(system_prompt, model, temperature, log_name)

    def format_example(self, image_base64: str) -> list[dict[str, str]]:
        """
        Formats image example for inclusion in prompt.
        """
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": "Here is a previous example image to compare against"},
                {"type": "image_url", "image_url": f"data:image/png;base64,{image_base64}"}
            ]
        }

    def evaluate(self, base64_img: str, examples: list[str]) -> tuple[str, int]:
        """
        Evaluates image against examples, returning (rationale, score).
        If score parsing fails return a -1 score.
        """
        fewshot_examples = [self.format_example(ex) for ex in examples]
        self.set_history(fewshot_examples)

        prompt = [
            {"type": "text", "text": "Evaluate the novelty of the following generated image:"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
        ]
        output = self.generate_response(prompt)

        score = re.search(r"\{(\d+)\}", output)
        # If we can parse a score, add to examples and return the score
        if score:
            return output, int(score.group(1))
        else:
            return output, -1  # Indicate that parsing failed
