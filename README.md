# Image Compression using PCA

## Overview
This project demonstrates the use of Principal Component Analysis (PCA) for image compression and dimensionality reduction using the MNIST handwritten digits dataset. PCA is applied to transform high-dimensional image data into a smaller set of principal components that capture the most significant variance in the dataset. The project visualizes reconstructed images for different numbers of components and evaluates reconstruction quality using Mean Squared Error (MSE).

## Objectives
- Implement PCA from scratch using NumPy and Singular Value Decomposition (SVD)
- Apply PCA on the MNIST dataset to achieve image compression
- Compare reconstruction quality for different numbers of principal components
- Visualize the trade-off between compression and accuracy

## Dataset
The project uses the MNIST dataset containing 70,000 grayscale images of handwritten digits (0–9), each of size 28×28 pixels. The dataset is automatically downloaded using the fetch_openml method from scikit-learn.

## Features
- Reconstruction of images using different numbers of principal components (k = 5, 10, 20, 50, 100, 200)
- Visualization of original vs reconstructed images
- Plot of reconstruction error (MSE) versus number of components
- Optional 2D PCA projection of digits for cluster visualization

## Technologies Used
Python  
NumPy  
Matplotlib  
scikit-learn  
Pillow  

## Installation and Setup
Follow these steps to run the project from scratch on any system:

1. Open Terminal and navigate to your desired folder: cd ~/Downloads
2. Clone this repository: git clone https://github.com/shyam1400/image-compression.git && cd image-compression
3. Create a virtual environment: python3 -m venv venv
4. Activate the virtual environment: source venv/bin/activate
5. Install all required dependencies: pip install numpy matplotlib scikit-learn pillow
6. Run the PCA MNIST script: python pca_mnist_compression.py
7. When finished, deactivate the environment: deactivate

## How It Works
1. The MNIST dataset is loaded and converted to NumPy arrays.  
2. The mean-centered data is decomposed using Singular Value Decomposition (SVD).  
3. The top k eigenvectors (principal components) are used to reconstruct images.  
4. MSE is computed to evaluate reconstruction accuracy.  
5. Reconstructed images and performance plots are displayed.

## Results
- Lower values of k result in higher compression but lower image quality.
- Higher values of k improve reconstruction at the cost of storage.
- The MSE decreases consistently as the number of components increases.

## Example Outputs
- Original vs reconstructed images for varying k values  
- MSE vs k line plot  
- Optional 2D PCA scatter plot showing clustering of digit classes  

## Conclusion
PCA effectively reduces image dimensionality while preserving essential visual details. The project highlights the balance between compression and accuracy and demonstrates PCA’s potential for image processing, feature extraction, and data visualization tasks.

## License
This project is licensed under the MIT License.
