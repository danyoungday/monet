import numpy as np

from embedder import Embedder
from expresser import Expresser
from population import Individual, Index


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
            return 1.0

        if candidate.phenotype is None:
            candidate.phenotype = self.expresser.express(candidate.genotype)
            if candidate.phenotype is None:
                return 1.0

        if candidate.embedding is None:
            candidate.embedding = self.embedder.encode_images([candidate.phenotype])

        novelty_score = self.index.measure_novelty(candidate.embedding.reshape(1, -1))[0]

        return novelty_score

    def prepare_candidates(self, candidates: list[Individual]) -> None:
        """
        Prepares candidates by expressing genotype and embedding phenotype if needed.
        Resets all scores to 1.0 initially.
        """
        phenotype_idxs = []
        phenotypes = []
        for i, candidate in enumerate(candidates):

            candidate.novelty_score = 1.0

            if candidate.genotype is None:
                continue

            if candidate.phenotype is None:
                phenotype = self.expresser.express(candidate.genotype)
                candidate.phenotype = phenotype

            if candidate.phenotype is not None:
                phenotype_idxs.append(i)
                phenotypes.append(candidate.phenotype)

        if len(phenotypes) > 0:
            embeddings = self.embedder.encode_images(phenotypes)
            for idx, embedding in zip(phenotype_idxs, embeddings):
                candidates[idx].embedding = embedding

    def evaluate_parallel(self, candidates: list[Individual]) -> None:
        """
        Evaluates the novelty of multiple candidates in parallel.
        Sets the candidate's phenotype and embedding as needed.
        Sets the candidate's novelty_score attribute.
        """
        self.prepare_candidates(candidates)

        embedding_idxs = []
        embeddings = []
        for i, candidate in enumerate(candidates):
            if candidate.embedding is not None:
                embedding_idxs.append(i)
                embeddings.append(candidate.embedding.reshape(-1))

        # Add to index if not present, measure novelty score, set on candidate
        for idx, embedding in zip(embedding_idxs, embeddings):
            candidate = candidates[idx]

            if candidate not in self.index.individuals:
                self.index.add_embedding(candidate)

            novelty_score = self.index.measure_novelty(embedding.reshape(1, -1))[0]

            candidate.novelty_score = novelty_score
