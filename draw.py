"""
The drawing agent used to produce images from code.
"""
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from agent import Agent


class CodingAgent(Agent):
    """
    Agent responsible for generating code to produce images.
    In charge of running the arbitrary code generated and returning the resulting image as a base64 string.
    """
    def __init__(self, model: str, temperature: float, log: bool = False):
        with open("sysprompts/pillow.txt", "r", encoding="utf-8") as f:
            system_prompt = f.read()

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

    def run_code(self, code: str) -> str:
        """
        Runs code and returns the stdout.
        Write code to a tempfile and execute it with subprocess, returning the generated image as a base64 string.
        Cases we have to look out for:
            1. The code runs successfully and outputs a base64 string.
            2. The code raises an exception.
            3. TODO: The code runs but doesn't output a proper base64 string.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_code_path = Path(temp_dir) / "temp_code.py"
            temp_code_path.write_text(code, encoding="utf-8")
            try:
                # Add modules directory to PYTHONPATH so our executing code can access it.
                env = os.environ.copy()
                env["PYTHONPATH"] = os.pathsep.join([str(Path.cwd() / "modules"), env.get("PYTHONPATH", "")])

                # Run the code in a subprocess
                process = subprocess.run(
                    [sys.executable, str(temp_code_path)],
                    check=True,
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=10,
                )
                output_str = process.stdout
                output_str = output_str.strip()

                # TODO: Check if this is a valid base64 string

                return output_str

            except subprocess.CalledProcessError:
                return None

    def generate(self, subject: str, examples: list[str]) -> tuple[str, str]:
        """
        Few-shot prompts the agent to generate code based on the provided prompt and examples.
        """
        full_prompt = f"Draw a {subject}"
        if len(examples) > 0:
            example_text = "\n\n".join(examples)
            full_prompt += f"\n\nThe following are some previously generated examples:\n\n{example_text}"
        response = self.generate_response(full_prompt)
        code = self.parse_code(response)

        if not code:
            return response, None

        base64_str = self.run_code(code)
        return response, base64_str
