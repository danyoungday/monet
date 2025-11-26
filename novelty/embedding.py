"""
Does the novelty evaluation with CLIP embeddings and FAISS.
"""
import faiss
# pylint: disable=E1120
# faiss has messed up typing so we disable this error in pylint
import numpy as np
import open_clip
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

import tempfile
from pathlib import Path
import os
import subprocess
import sys
from utils import decode_image


class EmbeddingNoveltyEvaluator:
    """
    Evaluator that uses embeddings to assess novelty.
    k = number of samples to consider in knn
    """
    class ImageDS(Dataset):
        def __init__(self, imgs: list[Image.Image], preprocess):
            self.imgs = imgs
            self.preprocess = preprocess

        def __len__(self):
            return len(self.imgs)

        def __getitem__(self, idx):
            img = self.imgs[idx]
            return self.preprocess(img)

    def __init__(self, k: int, device: str = "cpu", batch_size: int = 4):
        self.k = k
        self.device = device
        self.batch_size = batch_size
        self.d = 512
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32',
            pretrained='laion2b_s34b_b79k',
            device=self.device
        )
        self.model.eval()

        # Some indices to consider:
        # IndexFlatL2 - brute force
        # PQd/2FS
        # RFlat
        self.index = faiss.IndexFlatL2(self.d)
        self.history_embeddings = np.empty((0, self.d), dtype="float32")
        self.history_genotypes = []

    def add_embeddings(self, embeddings: torch.Tensor):
        embeddings_np = embeddings.cpu().numpy()
        self.index.add(embeddings_np)

    def express_genotype(self, genotype: str) -> Image.Image:
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
                img = decode_image(output_str)
                return img

            except subprocess.CalledProcessError:
                return None

    def encode_images(self, imgs: list[Image.Image]) -> torch.Tensor:
        dataset = self.ImageDS(imgs, self.preprocess)
        loader = DataLoader(dataset, batch_size=self.batch_size)

        all_features = []
        for batch in loader:
            with torch.no_grad():
                image_features = self.model.encode_image(batch.to(self.device))
                image_features /= image_features.norm(dim=-1, keepdim=True)
                all_features.append(image_features)

        image_features = torch.cat(all_features, dim=0)
        return image_features

    def evaluate(self, genotypes: list[str]) -> tuple[np.ndarray, torch.Tensor]:
        imgs = [self.express_genotype(genotype) for genotype in genotypes]
        valid_imgs = []
        valid_genotypes = []
        for img, genotype in zip(imgs, genotypes):
            if img is not None:
                valid_imgs.append(img)
                valid_genotypes.append(genotype)

        embeddings = self.encode_images(valid_imgs)

        if self.index.ntotal == 0:
            return 0.0, embeddings

        D, _ = self.index.search(embeddings.cpu().numpy(), min(self.k, self.index.ntotal))
        return D.mean(axis=1), embeddings
