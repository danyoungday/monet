import base64
from io import BytesIO

import faiss
# pylint: disable=E1120
# faiss has messed up typing so we disable this error in pylint
import numpy as np
from PIL import Image


class Individual:
    """
    Data class storing genotype, phenotype, and embedding.
    """
    def __init__(self, cand_id: int, parents: list[int], genotype: str):
        self.cand_id = cand_id
        self.parents = parents
        self.genotype = genotype
        self.phenotype: Image.Image = None
        self.embedding: np.ndarray = None
        self.novelty_score = 1.0
        self.is_subject = False

    def encode_phenotype(self):
        """
        Encodes the phenotype image into a base64 string for easy storage.
        """
        if self.phenotype is None:
            return None
        buffered = BytesIO()
        self.phenotype.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str


class Index:
    def __init__(self, k):
        self.k = k
        d = 512
        self.index = faiss.IndexFlatIP(d)
        self.individuals: list[Individual] = []

    def add_embedding(self, candidate: Individual):
        """
        Adds the embedding from an individual into the index.
        Normalize the embedding to unit length before adding so that inner product = cosine similarity.
        """
        embedding = candidate.embedding.reshape(1, -1).astype("float32")
        embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
        self.index.add(embedding)
        self.individuals.append(candidate)

    def measure_novelty(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Measure novelty of the given embeddings against the history. The candidate gets added to the index first so
        in essence k is actually k-1.
        Embeddings: (n, d) numpy array
        """
        if self.index.ntotal == 0:
            return np.array([1.0] * embeddings.shape[0])

        embeddings = embeddings.astype("float32")
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        if self.k == -1:
            k = self.index.ntotal
        else:
            k = min(self.k, self.index.ntotal)

        D, _ = self.index.search(embeddings, k=k)

        novelty_scores = D.mean(axis=1)
        return novelty_scores


def test_index():
    """
    Check if the index of 2 of the same image gives 0 distance.
    """
    index = Index(k=10)
    embedding = np.random.rand(1, 512).astype("float32")

    cand1 = Individual(0, [-1], "genotype1")
    cand1.embedding = embedding

    cand2 = Individual(1, [-1], "genotype2")
    cand2.embedding = embedding

    index.add_embedding(cand1)
    novelty_scores = index.measure_novelty(cand2.embedding)
    print(f"Novelty scores: {novelty_scores}")

    for i in range(100):
        emb = np.random.rand(1, 512).astype("float32")
        cand = Individual(i + 2, [-1], f"genotype{i + 2}")
        cand.embedding = emb
        index.add_embedding(cand)

    novelty_scores = index.measure_novelty(cand2.embedding)
    print(f"Novelty scores after adding more individuals: {novelty_scores}")

if __name__ == "__main__":
    test_index()
