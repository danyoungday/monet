import faiss
# pylint: disable=E1120
# faiss has messed up typing so we disable this error in pylint
import numpy as np
from PIL import Image


class Individual:
    """
    Data class storing genotype, phenotype, and embedding.
    """
    def __init__(self, cand_id: int, genotype: str):
        self.cand_id = cand_id
        self.genotype = genotype
        self.phenotype: Image.Image = None
        self.embedding: np.ndarray = None
        self.novelty_score = 0.0


class Index:
    def __init__(self, k):
        self.k = k
        d = 512
        self.index = faiss.IndexFlatL2(d)
        self.individuals: list[Individual] = []

    def add_embedding(self, candidate: Individual):
        """
        Adds the embedding from an individual into the index.
        """
        embedding = candidate.embedding.reshape(1, -1).astype("float32")
        self.index.add(embedding)
        self.individuals.append(candidate)

    def measure_novelty(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Measure novelty of the given embeddings against the history.
        This is a little more complicated than it seems because we don't want to measure novelty against ourselves.
        Do k+1 nearest neighbors and ignore the first one if the index contains the same embedding.
        Embeddings: (n, d) numpy array
        """
        if self.index.ntotal == 0:
            return np.array([0.0] * embeddings.shape[0])

        k = min(self.k+1, self.index.ntotal)
        D, I = self.index.search(embeddings, k=min(self.k + 1, self.index.ntotal))

        # Zero out the k+1th distance if the first neighbor is not identical
        identical = (D[:, 0] == 0.0)
        D[~identical, k-1] = 0.0

        novelty_scores = D.mean(axis=1)
        return novelty_scores


def test_index():
    """
    Check if the index of 2 of the same image gives 0 distance.
    """
    index = Index(k=3)
    embedding = np.random.rand(1, 512).astype("float32")

    cand1 = Individual(0, "genotype1")
    cand1.embedding = embedding

    cand2 = Individual(1, "genotype2")
    cand2.embedding = embedding

    index.add_embedding(cand1)
    novelty_scores = index.measure_novelty(cand2.embedding)
    print(f"Novelty scores: {novelty_scores}")


if __name__ == "__main__":
    test_index()
