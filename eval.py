"""
Pairwise evaluation of images.
"""
import numpy as np
from PIL import Image

from utils import History, decode_image


def compare_images(base64_img0: str, base64_img1: str) -> int:
    """
    Compares 2 base64 strings by novelty input by the user.
    Returns 0 or 1 depending on which the user thinks is more creative, or 2 if they are unsure.
    """
    img0 = decode_image(base64_img0)
    img1 = decode_image(base64_img1)

    Image.fromarray(np.hstack((np.array(img0),np.array(img1)))).show()
    user_input = input("Which image is more creative? (0 or 1; 2 if not sure):")
    while user_input not in ["0", "1", "2"]:
        user_input = input("Invalid input. Please enter 0, 1, or 2:")
    return int(user_input)


def sample_comparisons(history: History, n: int) -> list[tuple[int, int]]:
    """
    Sample n pairs of images from history without replacement.
    """
    indices = np.random.choice(len(history.entries), size=(n, 2), replace=False)
    return [(i0, i1) for i0, i1 in indices]


def eval_pairwise(history_path: str, n: int):
    """
    Perform the pairwise evaluation. Save the results as a 2D matrix np array.
    """
    history = History()
    history.load_history(history_path)

    comparison_idxs = sample_comparisons(history, n=n)

    pairwise_results = np.full((len(history.entries), len(history.entries)), -1)

    for idx0, idx1 in comparison_idxs:
        img0 = history.entries[idx0]["base64_img"]
        img1 = history.entries[idx1]["base64_img"]

        winner = compare_images(img0, img1)
        pairwise_results[idx0, idx1] = winner
        pairwise_results[idx1, idx0] = 1 - winner if winner in [0, 1] else 2

    print("Pairwise results:")
    print(pairwise_results)
    return pairwise_results


def main():
    history_path = "results/winrate.jsonl"
    pairwise_results = eval_pairwise(history_path, 30)
    np.save("results/winrate.npy", pairwise_results)


if __name__ == "__main__":
    main()