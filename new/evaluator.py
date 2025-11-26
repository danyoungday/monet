from new.embedder import Embedder
from new.expresser import Expresser
from new.population import Individual, Index


class Evaluator:
    """
    Evaluates the novelty of candidates.
    """
    def __init__(self, expresser: Expresser, embedder: Embedder, index: Index):
        self.expresser = expresser
        self.embedder = embedder
        self.index = index

    def evaluate(self, candidate: Individual) -> float:
        """
        Evaluates the novelty of a candidate by finding its knn distance in the index.
        If there is no genotype, generating code failed and we return 0.0 novelty.
        If there is no phenotype, try to express the genotype. If it fails, return 0.0 novelty.
        If there is no embedding, encode the phenotype to get the embedding.
        Then evaluate the embedding against the index to get the novelty score.
        """
        if candidate.genotype is None:
            return 0.0

        if candidate.phenotype is None:
            candidate.phenotype = self.expresser.express(candidate.genotype)
            if candidate.phenotype is None:
                return 0.0

        if candidate.embedding is None:
            candidate.embedding = self.embedder.encode_images([candidate.phenotype])

        novelty_score = self.index.measure_novelty(candidate.embedding.reshape(1, -1))[0]

        return novelty_score
    
    def evaluate_parallel(self, candidates: list[Individual]) -> list[float]:
        """
        Evaluates the novelty of multiple candidates in parallel.
        Sets the candidate's phenotype and embedding as needed.
        """
        # Get phenotypes or express if needed. Add None to list if expression or genotype fails.
        phenotypes = []
        for candidate in candidates:
            if candidate.genotype is None:
                phenotypes.append(None)

            elif candidate.phenotype is None:
                phenotype = self.expresser.express(candidate.genotype)
                candidate.phenotype = phenotype

            phenotypes.append(candidate.phenotype)

        # Get embeddings for phenotypes that are not None
        to_embed_indices = [i for i, pheno in enumerate(phenotypes) if pheno is not None]
        to_embed_imgs = [phenotypes[i] for i in to_embed_indices]
        embeddings = self.embedder.encode_images(to_embed_imgs)

        # Get novelty scores
        novelty_score_arr = self.index.measure_novelty(embeddings)

        # Set novelty scores and add embedding to candidate
        novelty_scores = [0.0] * len(candidates)
        for idx, score, embedding in zip(to_embed_indices, novelty_score_arr, embeddings):
            novelty_scores[idx] = score
            candidates[idx].embedding = embedding

        return novelty_scores


def test_evaluator():
    """
    Test function to make sure the evaluator works.
    """
    import numpy as np

    expresser = Expresser()
    embedder = Embedder(device="mps", batch_size=16)
    index = Index(k=3)
    for i in range(5):
        cand = Individual(i, genotype=f"blah{i}")
        cand.embedding = np.random.rand(512).astype("float32")
        index.add_embedding(cand)

    evaluator = Evaluator(expresser, embedder, index)

    # Create a dummy individual
    with open("test_genotype.txt", "r", encoding="utf-8") as f:
        genotype = f.read()

    individual = Individual(0, genotype)
    novelty_score = evaluator.evaluate(individual)
    print(f"Novelty score: {novelty_score}")

    individual2 = Individual(1, genotype)

    novelty_scores = evaluator.evaluate_parallel([individual, individual2])
    print(f"Parallel novelty scores: {novelty_scores}")


if __name__ == "__main__":
    test_evaluator()
