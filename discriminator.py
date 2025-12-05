import base64
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from PIL import Image
import re

from agent import Agent


class Discriminator:
    """
    Make sure the input image matches the desired criteria using an LLM.
    """
    def __init__(self, max_workers: int = 10):
        with open("sysprompts/discriminator.txt", "r", encoding="utf-8") as f:
            system_prompt = f.read()
        system_prompt = system_prompt.replace("<SUBJECT>", "cat")

        self.agent = Agent(system_prompt, "gpt-5", 1.0)

        self.max_workers = max_workers

    def classify_image(self, img: Image.Image) -> tuple[bool, str]:
        """
        Classify the image using the discriminator LLM. Return True or False for its response and its full rationale.
        """
        if img is None:
            return False, "No image provided."

        # Encode image into base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        content = [{
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_str}"}
        }]
        response = self.agent.generate_response(content)

        # Find the final {Yes} or {No} in the response. Ignore case.
        match = re.findall(r"\{(Yes|No)\}", response, re.IGNORECASE)
        is_match = False
        if match:
            answer = match[-1].lower()
            is_match = answer == "yes"

        return is_match, response

    def classify_image_parallel(self, imgs: list[Image.Image]) -> list[tuple[bool, str]]:
        """
        Classify multiple images in parallel.
        """
        outputs = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = [ex.submit(self.classify_image, img) for img in imgs]
            for fut in futures:
                output = fut.result()
                outputs.append(output)
        return outputs


def test_discriminator():
    with open("tests/test_img.png", "rb") as f:
        image_data = f.read()
    img = Image.open(BytesIO(image_data))
    discriminator = Discriminator()
    is_cat, rationale = discriminator.classify_image(img)
    print(f"Is cat: {is_cat}")
    print(f"Rationale: {rationale}")


if __name__ == "__main__":
    test_discriminator()
