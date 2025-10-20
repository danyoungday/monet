"""
Agent responsible for scoring the novelty of generated images.
"""
import base64
import random
import re

from agent import Agent


class ImageNoveltyAgent(Agent):

    def __init__(self, n_shot: int, model: str, temperature: float, log_name: str = None):
        with open("sysprompts/novelty_evaluator.txt", "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()
            super().__init__(system_prompt, model, temperature, log_name)

        self.n_shot = n_shot
        self.examples = []

    def format_example(self, image_base64: str) -> list[dict[str, str]]:
        """
        Formats image example for inclusion in prompt.
        """
        return [
            {"type": "text", "text": "Here is a previous example image to compare against"},
            {"type": "input_image", "image_url": f"data:image/png;base64,{image_base64}"}
        ]

    def evaluate(self, code: str) -> tuple[str, int]:
        examples = random.sample(self.examples, min(self.n_shot, len(self.examples)))
        examples = [self.format_example(ex) for ex in examples]

        self.set_history(examples)

        # This is scary!
        print(code)
        exec(code)

        with open("img.png", "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

        prompt = [
            {"type": "input_text", "text": "Evaluate the novelty of the following generated image:"},
            {"type": "input_image", "image_url": f"data:image/png;base64,{img_base64}"}
        ]
        output = self.generate_response(prompt)

        score = re.search(r"\{(\d+)\}", output)
        if score:
            return output, int(score.group(1))
        else:
            return output, -1  # Indicate that parsing failed
