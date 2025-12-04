import numpy as np
import open_clip
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


class ImageDS(Dataset):
    """
    Dataset for batching images.
    """
    def __init__(self, imgs: list[Image.Image], preprocess):
        self.imgs = imgs
        self.preprocess = preprocess

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        return self.preprocess(img)


class Embedder:
    """
    Class that encodes images into embeddings using OpenCLIP.
    """
    def __init__(self, device, batch_size):
        self.device = device
        self.batch_size = batch_size
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32',
            pretrained='laion2b_s34b_b79k',
            device=self.device
        )
        self.model.eval()

    def encode_images(self, imgs: list[Image.Image]) -> np.ndarray:
        """
        Encodes a list of PIL Images into embeddings, returns as a numpy array.
        """
        dataset = ImageDS(imgs, self.preprocess)
        loader = DataLoader(dataset, batch_size=self.batch_size)

        all_features = []
        for batch in loader:
            with torch.no_grad():
                image_features = self.model.encode_image(batch.to(self.device))
                image_features /= image_features.norm(dim=-1, keepdim=True)
                all_features.append(image_features)

        image_features = torch.cat(all_features, dim=0)
        image_features = image_features.cpu().numpy()
        return image_features


def test_embedder():
    """
    Tests that the embedder works as expected.
    """
    embedder = Embedder(device="mps", batch_size=8)

    img = Image.open("test_img.png")

    embeddings = embedder.encode_images([img])
    print(f"Embeddings shape: {embeddings.shape}")


if __name__ == "__main__":
    test_embedder()
