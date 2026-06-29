import os
import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from models.segmodels_multitasking import SEGFORMER


class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, mask_transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.images = sorted(os.listdir(img_dir))
        self.masks = sorted(os.listdir(mask_dir))
        assert len(self.images) == len(self.masks), "Mismatch between images and masks"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask.long()


def get_dataloader(img_folder, mask_folder, batch_size=4, num_workers=4):
    transform_img = transforms.Compose([
        transforms.Resize((224, 224)),   # adjust to model input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    transform_mask = transforms.Compose([
        transforms.Resize((112, 112), interpolation=Image.NEAREST),  # match logits
        transforms.PILToTensor()  # (1, H, W)
    ])

    dataset = SegmentationDataset(img_folder, mask_folder,
                                  transform=transform_img,
                                  mask_transform=transform_mask)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    return dataloader


def extract_pixel_features_and_labels(model: nn.Module,
                                      dataloader: DataLoader,
                                      device: torch.device,
                                      max_samples: int = 10000,
                                      include_background: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract pixel-level features (from last feature map) and corresponding class labels.
    Optionally exclude background (class 0).
    """
    model.eval()
    all_features, all_labels = [], []

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)  # y: (B,1,112,112)
            feat,_, logits = model(x)

            # last feature map: (B, 256, 14, 14)
            last_feat = feat[-1]
            B, C, Hf, Wf = last_feat.shape

            # downsample mask to match feature map size (14x14)
            y_down = torch.nn.functional.interpolate(
                y.float(), size=(Hf, Wf), mode="nearest"
            ).long().squeeze(1)  # (B, Hf, Wf)

            # reshape
            features = last_feat.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy()
            labels = y_down.reshape(-1).cpu().numpy()  # (B*Hf*Wf,)

            if not include_background:
                mask = labels != 0
                features = features[mask]
                labels = labels[mask]

            all_features.append(features)
            all_labels.append(labels)

            if (batch_idx + 1) % 5 == 0:
                print(f"Processed {batch_idx + 1} batches...")

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Optionally sample a subset for faster t-SNE
    if len(all_features) > max_samples:
        idx = np.random.choice(len(all_features), size=max_samples, replace=False)
        all_features = all_features[idx]
        all_labels = all_labels[idx]

    return all_features, all_labels
# -----------------------------


import seaborn as sns  # For better color palettes and styling

def plot_tsne(features, labels, title="Pixel-level t-SNE Visualization", save_path="tsne_pixels.png"):
    # Initialize t-SNE with consistent random state for reproducibility
    tsne = TSNE(n_components=2, random_state=42, verbose=1)
    tsne_out = tsne.fit_transform(features)

    # Set a modern style using seaborn with a universal sans-serif font
    sns.set_style("whitegrid", {"axes.grid": False, "font.family": "sans-serif"})
    plt.figure(figsize=(10, 8), dpi=150)  # Larger figure size for better resolution and clarity

    # Use a vibrant, continuous colormap (viridis) for better visual appeal
    scatter = plt.scatter(
        tsne_out[:, 0], 
        tsne_out[:, 1], 
        c=labels, 
        cmap="viridis",  # Smooth, accessible colormap
        s=50,  # Increased point size
        alpha=0.7,  # Slight transparency for overlaps
        edgecolors="white",  # White edges for contrast
        linewidth=0.5
    )

    # Customize title with larger, bold font and padding
    plt.title(
        title, 
        fontsize=16, 
        fontweight="bold", 
        pad=20, 
        family="sans-serif"
    )

 
    # Customize axes labels with larger, clean fonts
    plt.xlabel("t-SNE Dimension 1", fontsize=12, family="sans-serif")
    plt.ylabel("t-SNE Dimension 2", fontsize=12, family="sans-serif")

    # Add subtle gridlines for better readability
    plt.grid(True, linestyle="--", alpha=0.3)

    # Adjust layout to prevent clipping and ensure clean presentation
    plt.tight_layout()

    # Save the plot with high resolution
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"t-SNE plot saved to {save_path}")

    # Close the plot to free memory
    plt.close()

# -----------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_folder = "./MTD-SSL/test/test_images"
    mask_folder = "./MTD-SSL/test/test_annot"

    dataloader = get_dataloader(img_folder, mask_folder, batch_size=4)

    # load your model
    model = SEGFORMER('MiT-B0', 6).to(device)
    path = "./EXP/MTD/MiT-B0_seg_supervised_singlemod_100P_20250904_135839/Checkpoints/MiT-B0_MTD_supervised_100P.pth"
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    print('The state dicts are loaded')
    
    # Move the selected model to device
    model.to(device)
    
    features, labels = extract_pixel_features_and_labels(
        model, dataloader, device,
        include_background=False,   # <-- ignore background
        max_samples=50000
    )

    # plot
    plot_tsne(features, labels, title="Pixel-level t-SNE (No Background)",
            save_path="tsne_pixel_level_nobg_MTD_100p.png")
