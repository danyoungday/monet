from datasets import load_dataset

from draw.diffusion import LMXDiffuser
from novelty.discriminator import Discriminator
from novelty.embedding import EmbeddingNoveltyEvaluator
from utils import History, encode_image


def setup_initial_pop(n: int, history: History, diffuser: LMXDiffuser, evaluator: EmbeddingNoveltyEvaluator):
    """
    Sets up the initial population in history with n examples from the dataset.
    """

    # Load initial examples from dataset
    print("Loading initial examples from dataset...")
    dataset = load_dataset("Gustavosta/Stable-Diffusion-Prompts", split="train")
    filtered = dataset.filter(lambda x: " cat " in x["Prompt"].lower())
    initial_examples = filtered.to_pandas()
    initial_examples = initial_examples.sample(n=n, random_state=42)

    # Generate images for history
    print("Generating images...")
    initial_prompts = [prompt for prompt in initial_examples["Prompt"]]
    initial_images = diffuser.express(initial_prompts)

    # Add embeddings to evaluator
    print("Embedding images...")
    novelties, embeddings = evaluator.evaluate(initial_images)
    evaluator.add_embeddings(embeddings)

    # Add entries to history
    initial_examples["base64_img"] = [encode_image(img) for img in initial_images]
    initial_examples["novelty_score"] = novelties
    for i, row in initial_examples.iterrows():
        history.add_entry(
            entry_id=i,
            gen=0,
            parents=[],
            genotype=row["Prompt"],
            phenotype=row["base64_img"],
            novelty_score=row["novelty_score"]
        )

def diffusion_iteration(
    n_shot: int,
    n_children: int,
    threshold: float,
    history: History,
    diffuser: LMXDiffuser,
    discriminator: Discriminator,
    evaluator: EmbeddingNoveltyEvaluator):
    """
    A single iteration of the diffusion loop.
    """

    # Sample some previous examples
    all_examples = [history.sample(n_shot, prop=False) for _ in range(n_children)]

    # Generate new children with LMX
    all_example_prompts = [[ex["code"] for ex in examples] for examples in all_examples]
    prompts = diffuser.reproduce(all_example_prompts)

    # Filter out any failed generations
    prompts = [p for p in prompts if p is not None]

    # Generate images from prompts
    images = diffuser.express(prompts)

    # Discriminate images
    passed = discriminator.discriminate(images)
    
    # Evaluate novelty and get embeddings of passed images
    passed_imgs = [img for img, p in zip(images, passed) if p]
    novelty_scores, embeddings = evaluator.evaluate(passed_imgs)

    novel_imgs = [img for img, score in zip(passed_imgs, novelty_scores) if score > threshold]
    novel_embeddings = embeddings[novelty_scores > threshold]

    # Record novel images in history and add embeddings to evaluator
    evaluator.add_embeddings(novel_embeddings)
    gens = [max([ex["gen"] for ex in examples]) + 1 for examples in all_examples]
    return
