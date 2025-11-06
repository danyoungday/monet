from PIL import Image

from utils import run_fn_parallel


class Artist:
    """
    Generates a genotype from example genotypes.
    Produces a phenotype from genotype.
    """
    def __init__(self, max_workers: int):
        self.max_workers = max_workers

    def reproduce(self, examples: list[str]) -> str:
        """
        Takes a set of example genotypes and produces a new genotype.
        """
        raise NotImplementedError

    def reproduce_parallel(self, all_examples: list[list[str]]) -> list[str]:
        """
        Takes in a batch of a set of example genotypes and produces a new genotype for each set.
        """
        inputs = [{"examples": examples} for examples in all_examples]
        genotypes = run_fn_parallel(self.reproduce, inputs, self.max_workers)
        return genotypes

    def express(self, genotype: str) -> str:
        """
        Takes in a genotype and produces a corresponding phenotype.
        """
        raise NotImplementedError

    def express_parallel(self, genotypes: list[str]) -> list[str]:
        """
        Takes in a batch of genotypes and produces corresponding phenotypes.
        """
        inputs = [{"genotype": genotype} for genotype in genotypes]
        phenotypes = run_fn_parallel(self.express, inputs, self.max_workers)
        return phenotypes