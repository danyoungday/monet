from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_dataset

from draw.diffusion import Diffuser, LMX
from novelty.embedding import EmbeddingNoveltyEvaluator
from utils import History, encode_image

def setup_initial_pop(history: History, diffuser: Diffuser, evaluator: EmbeddingNoveltyEvaluator):
    history = History()

    # Load initial examples from dataset
    print("Loading initial examples from dataset...")
    dataset = load_dataset("Gustavosta/Stable-Diffusion-Prompts", split="train")
    filtered = dataset.filter(lambda x: (" cat." in x["Prompt"].lower()) or (" cat " in x["Prompt"].lower()))
    initial_examples = filtered.to_pandas()
    initial_examples = initial_examples.sample(n=10, random_state=42)

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

    history.save_history("test.jsonl")


def lmx_job(lmx: LMX, examples: list[str]) -> str:
    return lmx.generate_prompt(examples)

def run_diffusion_experiment():

    history = History()
    lmx = LMX()
    diffuser = Diffuser(device="mps")
    evaluator = EmbeddingNoveltyEvaluator(k=30, device="mps")

    print("Setting up initial population...")
    setup_initial_pop(history, diffuser, evaluator)

    # prompts = []
    # with ThreadPoolExecutor(max_workers=10) as ex:
    #     # Set up jobs and hold results in to_add
    #     submit = [lmx_job, lmx, ]
    #     futures = [ex.submit(*submit) for _ in range(10)]
    #     to_add = []
    #     for fut in as_completed(futures):
    #         prompt = fut.result()
    #         prompts.append(prompt)


if __name__ == "__main__":
    run_diffusion_experiment()