from concurrent.futures import ThreadPoolExecutor
import re

import diffusers
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from agent import Agent
from draw.artist import Artist


class LMXDiffuser(Artist):
    class DiffuserDataset(Dataset):
        def __init__(self, prompts: list[str]):
            self.prompts = prompts
        def __len__(self):
            return len(self.prompts)
        def __getitem__(self, idx):
            return self.prompts[idx]

    def __init__(self, device: str = "cpu", batch_size: int = 4, max_workers: int = 10):
        self.device = device
        self.batch_size = batch_size
        self.max_workers = max_workers

        # Set up the diffusion model
        self.pipe = diffusers.DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")
        self.pipe = self.pipe.to(self.device)
        if self.device == "mps":
            # Recommended if your computer has < 64 GB of RAM
            self.pipe.enable_attention_slicing()

        self.lmx = LMX()

    def reproduce(self, all_examples: list[list[str]]) -> list[str]:
        """
        Takes in a list of prompts and produces a new prompt.
        """
        prompts = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = [ex.submit(self.lmx.generate, examples) for examples in all_examples]
            for fut in futures:
                prompt = fut.result()
                prompts.append(prompt)
        return prompts

    def express(self, prompts: list[str]) -> list[Image.Image]:
        loader = DataLoader(self.DiffuserDataset(prompts), batch_size=self.batch_size)
        images = []
        for batch in loader:
            batch_images = self.pipe(batch).images
            images.extend(batch_images)
        return images


class LMX(Artist):
    def __init__(self):
        # Set up the LMX model
        system_prompt = ""
        with open("sysprompts/lmx.txt", "r", encoding="utf-8") as f:
            system_prompt = f.read()
            system_prompt = system_prompt.replace("<SUBJECT>", "cat")
        print(system_prompt)
        self.agent = Agent(system_prompt, model="gpt-5-mini", temperature=1.0)

    def generate(self, examples: list[str]) -> str:
        formatted_examples = [f"Example: <Prompt>{ex}</Prompt>" for ex in examples]
        response = self.agent.generate_response("\n".join(formatted_examples))
        prompt_match = r"<Prompt>\s*(.*?)\s*</Prompt>"
        prompts = re.findall(prompt_match, response, flags=re.DOTALL | re.IGNORECASE)
        if prompts:
            return prompts[-1].strip()
        else:
            return None
