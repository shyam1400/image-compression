# pca_mnist_compression.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# -------------------------
# 1) Load MNIST dataset
# -------------------------
print("Downloading MNIST dataset (if not cached)...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')

# Convert to NumPy arrays to avoid pandas issues
X = np.array(mnist.data, dtype=np.float32)   # shape: (70000, 784)
y = np.array(mnist.target, dtype=np.int64)   # labels 0-9
print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features per image.")

# -------------------------
# 2) PCA helper function
# -------------------------
def pca_fit_transform(X, k):
    """
    Perform PCA on data X (n_samples x n_features) keeping top k components.
    Returns: reconstructed X, mean vector, top-k components, scores
    """
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    V_k = Vt[:k, :]            # top-k principal components
    scores = Xc @ V_k.T        # projected coordinates
    X_recon = scores @ V_k + mean
    return X_recon, mean, V_k, scores

# -------------------------
# 3) Visualization helpers
# -------------------------
def plot_reconstructions(X_orig, X_recon, num_images=10):
    plt.figure(figsize=(20, 4))
    for i in range(num_images):
        # original
        ax = plt.subplot(2, num_images, i + 1)
        plt.imshow(X_orig[i].reshape(28,28), cmap='gray')
        plt.axis('off')
        # reconstructed
        ax = plt.subplot(2, num_images, i + 1 + num_images)
        plt.imshow(X_recon[i].reshape(28,28), cmap='gray')
        plt.axis('off')
    plt.show()

def plot_mse_vs_k(k_values, mse_values):
    plt.figure(figsize=(8,5))
    plt.plot(k_values, mse_values, marker='o')
    plt.xlabel("Number of PCA Components (k)")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title("Reconstruction Error vs PCA Components")
    plt.grid(True)
    plt.show()

# -------------------------
# 4) Run PCA for multiple k values
# -------------------------
k_values = [5, 10, 20, 50, 100, 200]
mse_values = []

# Use first 1000 samples for speed
X_subset = X[:1000]

for k in k_values:
    print(f"\nRunning PCA with k={k} components...")
    X_recon, mean, components, scores = pca_fit_transform(X_subset, k)
    mse_val = np.mean((X_subset - X_recon)**2)
    mse_values.append(mse_val)
    print(f"MSE: {mse_val:.2f}")
    plot_reconstructions(X_subset, X_recon, num_images=10)

# -------------------------
# 5) Plot MSE vs k
# -------------------------
plot_mse_vs_k(k_values, mse_values)

# -------------------------
# 6) Optional: 2D PCA visualization
# -------------------------
def plot_2d_pca(X, y, num_samples=2000):
    """
    Reduce MNIST to 2D using PCA and plot colored by digit label
    """
    X_subset = X[:num_samples]
    y_subset = y[:num_samples]
    _, _, V_k, scores = pca_fit_transform(X_subset, 2)
    
    plt.figure(figsize=(8,6))
    for digit in range(10):
        mask = y_subset == digit
        plt.scatter(scores[mask,0], scores[mask,1], label=str(digit), alpha=0.6)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("MNIST digits projected to 2D via PCA")
    plt.legend()
    plt.show()

# Uncomment to visualize 2D projection
# plot_2d_pca(X, y)
