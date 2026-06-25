import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from sklearn.decomposition import PCA

# Define paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_PATH = os.path.join(SCRIPT_DIR, "initial_data_visualization.png")

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

print("Preparing PCA-based data visualization for MNIST...")
transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

mnist_dir = os.path.join(SCRIPT_DIR, "mnist_data")
train_dataset = datasets.MNIST(
    mnist_dir, train=True, download=True, transform=transform
)

# Load a larger batch of samples to get a clear PCA distribution
num_samples = 1000
images = []
labels = []
for idx in range(num_samples):
    img, lbl = train_dataset[idx]
    images.append(img.view(-1).numpy())
    labels.append(lbl)

images = np.stack(images)  # Shape: (1000, 784)
labels = np.array(labels)

# 1. Compute 2D PCA on the raw 784D MNIST images
print("Computing PCA on raw images...")
pca_raw = PCA(n_components=2)
coords_raw = pca_raw.fit_transform(images)

# 2. Create 16D random projection matrix and project
print("Computing random projections...")
PROJ_DIM = 16
random_matrix = np.random.randn(784, PROJ_DIM) / np.sqrt(PROJ_DIM)
projected_images = np.dot(images, random_matrix)

# Compute 2D PCA on the 16D random projected features
print("Computing PCA on random projected features...")
pca_proj = PCA(n_components=2)
coords_proj = pca_proj.fit_transform(projected_images)

# Find first sample of each digit (0-9) to concatenate horizontally
digit_strip = []
for idx in range(10):
    sample_idx = np.where(labels == idx)[0][0]
    digit_strip.append(images[sample_idx].reshape(28, 28))

# Concatenate all 10 images side-by-side into a single wide image
horizontal_strip = np.hstack(digit_strip)

# 3. Setup plotting with a clean 1x3 layout
plt.figure(figsize=(18, 5.5))

# Subplot 1: Concatenated MNIST digits (0-9)
plt.subplot(1, 3, 1)
plt.imshow(horizontal_strip, cmap="gray")
plt.title("Sample MNIST Digits (Classes 0-9)")
plt.axis("off")

# Subplot 2: 2D PCA of raw 784D images
plt.subplot(1, 3, 2)
scatter_raw = plt.scatter(
    coords_raw[:, 0], coords_raw[:, 1], c=labels, cmap="tab10", alpha=0.6, s=15
)
plt.colorbar(scatter_raw, label="Digit Class")
plt.title("2D PCA of Raw Images (784D)")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.grid(True)

# Subplot 3: 2D PCA of 16D Random Projections
plt.subplot(1, 3, 3)
scatter_proj = plt.scatter(
    coords_proj[:, 0], coords_proj[:, 1], c=labels, cmap="tab10", alpha=0.6, s=15
)
plt.colorbar(scatter_proj, label="Digit Class")
plt.title("2D PCA of Random Projections (16D)")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.grid(True)

plt.tight_layout()
plt.savefig(PLOT_PATH)
print(f"Data visualization complete! Image saved to {PLOT_PATH}")
