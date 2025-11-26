import base64
from io import BytesIO
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from PIL import Image


class Expresser:
    """
    Expresses a genotype (code str) into phenotype (PIL Image).
    """
    def decode_image(self, base64_str: str) -> Image:
        """
        Decodes a base64 string to a PIL Image.
        """
        image_data = base64.b64decode(base64_str)
        return Image.open(BytesIO(image_data))

    def express(self, genotype: str):
        """
        Runs code and returns the stdout.
        Write code to a tempfile and execute it with subprocess, returning the generated image as a base64 string.
        Cases we have to look out for:
            1. The code runs successfully and outputs a base64 string.
            2. The code raises an exception.
            3. TODO: The code runs but doesn't output a proper base64 string.
        """
        if genotype is None:
            return None

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_code_path = Path(temp_dir) / "temp_code.py"
            temp_code_path.write_text(genotype, encoding="utf-8")
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

                # TODO: Check if this is a valid image
                output_img = self.decode_image(output_str)
                return output_img

            except subprocess.CalledProcessError:
                return None


def test_expresser():
    """
    Test function to make sure the expresser works.
    """
    expresser = Expresser()

    with open("test_genotype.txt", "r", encoding="utf-8") as f:
        genotype = f.read()

    phenotype = expresser.express(genotype)
    if phenotype is not None:
        phenotype.show()
    else:
        print("Expression failed.")


if __name__ == "__main__":
    test_expresser()
