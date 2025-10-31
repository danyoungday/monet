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

from utils import decode_image

class ImageDS(torch.utils.data.Dataset):
    def __init__(self, imgs: list[Image.Image], preprocess):
        self.imgs = imgs
        self.preprocess = preprocess

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        return self.preprocess(img)


class EmbeddingNoveltyEvaluator:
    """
    Evaluator that uses embeddings to assess novelty.
    k = number of samples to consider in knn
    """
    def __init__(self, k: int, device: str = "cpu"):
        self.k = k
        self.device = device
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

    def add_embeddings(self, embeddings: torch.Tensor):
        embeddings_np = embeddings.cpu().numpy()
        self.index.add(embeddings_np)

    def encode_images(self, imgs: list[Image.Image]) -> torch.Tensor:
        processed = [self.preprocess(img).unsqueeze(0).to(self.device) for img in imgs]
        processed = torch.cat(processed, dim=0)
        print(processed.shape)
        with torch.no_grad():
            image_features = self.model.encode_image(processed)
            image_features /= image_features.norm(dim=-1, keepdim=True)

        return image_features

    def evaluate(self, base64_imgs: list[str]) -> tuple[np.ndarray, torch.Tensor]:

        imgs = [decode_image(b64) for b64 in base64_imgs]
        embeddings = self.encode_images(imgs)

        if self.index.ntotal == 0:
            return 0.0, embeddings

        D, _ = self.index.search(embeddings.cpu().numpy(), min(self.k, self.index.ntotal))
        return D.mean(axis=1), embeddings
