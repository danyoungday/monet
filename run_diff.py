from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_dataset

from draw.diffusion import Diffuser, LMX
from novelty.embedding import EmbeddingNoveltyEvaluator
from utils import History, encode_image

def setup_initial_pop(n: int, history: History, diffuser: Diffuser, evaluator: EmbeddingNoveltyEvaluator):
    history = History()

    # Load initial examples from dataset
    print("Loading initial examples from dataset...")
    dataset = load_dataset("Gustavosta/Stable-Diffusion-Prompts", split="train")
    filtered = dataset.filter(lambda x: (" cat." in x["Prompt"].lower()) or (" cat " in x["Prompt"].lower()))
    initial_examples = filtered.to_pandas()
    initial_examples = initial_examples.sample(n=n, random_state=42)

    # Generate images for history
    print("Generating images...")
    initial_prompts = [prompt for prompt in initial_examples["Prompt"]]
    initial_images = diffuser.generate_images(initial_prompts)

    # Add embeddings to evaluator
    print("Embedding images...")
    base64_imgs = [encode_image(img) for img in initial_images]
    novelties, embeddings = evaluator.evaluate(base64_imgs)
    evaluator.add_embeddings(embeddings)

    # Add entries to history
    initial_examples["base64_img"] = base64_imgs
    initial_examples["novelty_score"] = novelties
    for _, row in initial_examples.iterrows():
        history.add_entry(
            parents=[],
            code=row["Prompt"],
            base64_img=row["base64_img"],
            novelty_score=row["novelty_score"],
            rationale="",
            gen=0
        )


def lmx_job(lmx: LMX, examples: list[str]) -> str:
    return lmx.generate_prompt(examples)


def batch_lmx(lmx: LMX, all_examples: list[list[str]]) -> list[str]:
    prompts = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        # Set up jobs and hold results in to_add
        futures = [ex.submit(lmx_job, lmx, examples) for examples in all_examples]
        for fut in as_completed(futures):
            prompt = fut.result()
            prompts.append(prompt)
    return prompts


def run_diffusion_experiment():

    threshold = 0.0

    history = History()
    lmx = LMX()
    diffuser = Diffuser(device="mps")
    evaluator = EmbeddingNoveltyEvaluator(k=30, device="mps")

    print("Setting up initial population...")
    setup_initial_pop(10, history, diffuser, evaluator)

    all_examples = [history.sample(5, prop=False) for _ in range(10)]
    all_example_codes = [[ex["code"] for ex in examples] for examples in all_examples]

    print("Generating prompts with LMX...")
    prompts = batch_lmx(lmx, all_example_codes)
    print(f"LMX produced {len(prompts)} prompts.")

    print("Generating images with Diffuser...")
    images = diffuser.generate_images(prompts)

    print("Evaluating novelty with Novelty Evaluator...")
    base64_imgs = [encode_image(img) for img in images]
    novelties, embeddings = evaluator.evaluate(base64_imgs)

    to_add_idxs = [i for i, score in enumerate(novelties) if score > threshold]

    embeddings_to_add = embeddings[to_add_idxs]
    evaluator.add_embeddings(embeddings_to_add)

    for idx in to_add_idxs:
        examples = all_examples[idx]
        history.add_entry(
            parents=[ex["entry_id"] for ex in examples],
            code=prompts[idx],
            base64_img=base64_imgs[idx],
            novelty_score=novelties[idx],
            rationale="",
            gen=max([ex["gen"] for ex in examples], default=-1) + 1
        )

    history.save_history("results/diff-exp.jsonl")


if __name__ == "__main__":
    run_diffusion_experiment()