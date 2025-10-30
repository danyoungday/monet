"""
Does the novelty evaluation with CLIP embeddings and FAISS.
"""
import faiss
# pylint: disable=E1120
# faiss has messed up typing so we disable this error in pylint
import open_clip
from PIL import Image
import torch

from utils import decode_image


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

    def add_embeddings(self, embeddings_list: list[torch.Tensor]):
        if len(embeddings_list) == 0:
            return
        embeddings = torch.cat(embeddings_list, dim=0)
        embeddings_np = embeddings.cpu().numpy()
        self.index.add(embeddings_np)

    def encode_image(self, img: Image) -> torch.Tensor:
        preprocessed = self.preprocess(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(preprocessed)
            image_features /= image_features.norm(dim=-1, keepdim=True)

        return image_features

    def evaluate(self, base64_img: str) -> tuple[float, torch.Tensor]:

        img = decode_image(base64_img)
        embedding = self.encode_image(img)

        if self.index.ntotal == 0:
            return 0.0, embedding

        D, _ = self.index.search(embedding.cpu().numpy(), min(self.k, self.index.ntotal))
        return float(D.mean()), embedding
