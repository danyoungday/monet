from concurrent.futures import ThreadPoolExecutor
import re

from PIL import Image

from agent import Agent
from utils import encode_image


class Discriminator(Agent):
    def __init__(self, model: str, temperature: float = 1.0, max_workers: int = 10):

        self.max_workers = max_workers

        system_prompt = ""
        with open("sysprompts/discriminator.txt", "r", encoding="utf-8") as f:
            system_prompt = f.read()
        super().__init__(system_prompt, model, temperature)

    def generate(self, base64_img: str) -> str:
        """
        Generates a discrimination response for the given base64 image string.
        """
        prompt = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
        ]
        output = self.generate_response(prompt)
        return output

    def find_answer(self, output: str) -> bool:
        """
        Parses the discriminator output to find the answer.
        Returns True if the answer is 'B', False otherwise.
        """
        # Search for the final instance of a letter within braces
        # B is the correct answer
        matches = re.findall(r"\{([A-E])\}", output)
        if matches:
            answer = matches[-1]
            return answer.lower() == "b"
        return False

    def discriminate(self, imgs: list[Image.Image]) -> bool:
        """
        Takes a list of images and returns a corresponding list of True or False if they are the given subject.
        """
        base64_imgs = [encode_image(img) for img in imgs]
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = [ex.submit(self.generate, base64_img) for base64_img in base64_imgs]
            results = []
            for fut in futures:
                output = fut.result()
                answer = self.find_answer(output)
                results.append(answer)

        return results
