"""
Pairwise evaluation of images.
"""
import numpy as np
import pandas as pd
from PIL import Image

from utils import decode_image


def compare_images(base64_img0: str, base64_img1: str) -> int:
    """
    Compares 2 base64 strings by novelty input by the user.
    Returns 0 or 1 depending on which the user thinks is more creative, or 2 if they are unsure.
    """
    img0 = decode_image(base64_img0)
    img1 = decode_image(base64_img1)

    Image.fromarray(np.hstack((np.array(img0), np.array(img1)))).show()
    user_input = input("Which image is more creative? (a or b; x if not sure):")
    while user_input not in ["a", "b", "x"]:
        user_input = input("Invalid input. Please enter a, b, or x:")
    if user_input == "a":
        return 1
    elif user_input == "b":
        return -1
    else:
        return 0


def sample_comparisons(num_entries: int, n: int) -> list[tuple[int, int]]:
    """
    Sample n pairs of images from history without replacement.
    """
    all_indices = [(i, j) for i in range(num_entries) for j in range(i + 1, num_entries)]
    np.random.shuffle(all_indices)
    return all_indices[:n]


def eval_pairwise(history_path: str, n: int):
    """
    Perform the pairwise evaluation. Save the results as a 2D matrix np array.
    """
    history_df = pd.read_csv(history_path)

    if n == -1:
        n = len(history_df)
    comparison_idxs = sample_comparisons(len(history_df), n=n)

    pairwise_results = np.full((len(history_df), len(history_df)), 0)

    for idx0, idx1 in comparison_idxs:
        img0 = history_df.iloc[idx0]["phenotype"]
        img1 = history_df.iloc[idx1]["phenotype"]

        winner = compare_images(img0, img1)
        pairwise_results[idx0, idx1] = winner
        pairwise_results[idx1, idx0] = -1 * winner

    print("Pairwise results:")
    print(pairwise_results)
    return pairwise_results


def main():
    np.random.seed(42)
    history_path = "results/full-save.csv"
    pairwise_results = eval_pairwise(history_path, n=-1)
    np.save("results/full-save.npy", pairwise_results)


if __name__ == "__main__":
    main()
