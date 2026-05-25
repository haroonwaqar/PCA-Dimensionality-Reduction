# Principal Component Analysis on MNIST Dataset

Implementation of Principal Component Analysis (PCA) via Singular Value Decomposition (SVD) applied to the MNIST handwritten digits dataset. The project investigates how Gaussian noise at a specific Signal-to-Noise Ratio (SNR) affects PCA performance, dimensionality reduction efficiency, and reconstruction quality.

The core question: *can we represent each 784-pixel image using far fewer numbers while retaining most of the useful information and how does noise change that answer?*

## Dataset

**MNIST Handwritten Digits**
- 60,000 training images (6,000 used for computational efficiency)
- Each image: 28×28 pixels = **784 features**
- 10 classes (digits 0–9)
- Source: [Kaggle Datasets MNIST page](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)

Download the following files and place them in a `dataset/` folder:
```
dataset/
├── train-images.idx3-ubyte
└── train-labels.idx1-ubyte
```

## Project Structure

```
├── main.m              # Main script — runs full pipeline
├── PCA_SVD.m           # PCA implementation via SVD
├── load_mnist.m        # MNIST binary file loader
├── dataset/            # MNIST data files (not included)
└── README.md
```

## Methodology

### 1. Data Preparation
- MNIST images loaded from IDX binary format
- Reshaped into matrix form: **N × 784** (samples × features)
- Normalized to [0, 1] by dividing pixel values by 255

### 2. Noise Addition (SNR = 13 dB)
Gaussian noise added using the SNR formula:

```
noise_power = signal_power / 10^(SNR_dB / 10)
noise = sqrt(noise_power) × randn(size(X))
```

At SNR = 13 dB, noise power ≈ **5% of signal power**.

### 3. PCA via SVD
```
X_centered = X - mean(X)
[~, S, V]  = svd(X_centered, 'econ')
Z          = X_centered × V_k          % project to k dimensions
```

Eigenvalues derived as: `λ = diag(S)² / (N-1)`

### 4. Dynamic k Selection
Instead of hardcoding k, the optimal number of components is found automatically using a **95% variance threshold**:

```matlab
k = find(cumsum(explained_variance) >= 0.95, 1, 'first');
```

### 5. Reconstruction & Error Analysis
```
X_reconstructed = Z × V_k' + mean(X)
MSE = mean((X_original - X_reconstructed)²)
```

## Results

### Dynamic Threshold Analysis

| Threshold | Original k | Noisy k | Difference |
|-----------|-----------|---------|------------|
| 90%       |  84       |  162    |  78        |
| **95%**   |  **149**  |  **326**|  **177**   |
| 99%       |  323      |  640    |  317       |

> Noisy data requires **more than 2× the components** to reach the same variance threshold.

### Reconstruction MSE at 95% k

| Dataset  | k used | MSE     |
|----------|--------|---------|
| Original |  149   | 0.00334 |
| Noisy    |  326   | 0.00365 |

## Visualizations

The code produces four plots:

**Plot 1: Cumulative Explained Variance**

Compares how quickly original vs noisy data accumulates variance. Threshold lines at 90%, 95%, 99% with vertical markers showing where each dataset crosses 95%.

![Plot 1](<plots/plot1.png>)

**Plot 2: Reconstruction Error vs k**

Shows how MSE changes as k varies from 1 to 500. Demonstrates the cost of removing features and the noise floor effect.

![Plot 2](<plots/plot2.png>)

**Plot 3: 2D PCA Scatter**

Each image projected onto top 2 PCs, colored by digit class. Shows partial clustering in original, degraded separation in noisy.

![Plot 3](<plots/plot3.png>)

**Plot 4: 3D PCA Scatter**

Same as 2D but with top 3 components. Original shows elongated structure; noisy appears more spherical due to isotropic noise.

![Plot 4](<plots/plot4.png>)

## Key Findings

**1. Noise flattens the eigenvalue spectrum**

Gaussian noise is isotropic. It adds equal variance in every direction. This inflates the total variance and boosts lower eigenvalues, making each PC explain a smaller fraction of the total. More components are needed to accumulate to any fixed threshold.

**2. PCA compresses, it does not denoise**

PCA cannot distinguish signal variance from noise variance. Noise is baked into the data before PCA runs. The MSE gap between clean and noisy reconstruction is irreducible at any fixed k.

**3. Adaptive k mitigates but does not eliminate the noise effect**

At each dataset's own 95% k, MSE values are close (0.00334 vs 0.00365). But achieving that required 326 components for noisy vs 149 for clean, the same quality at more than double the cost.

<!-- ## How to Run

1. Download MNIST dataset files into `dataset/`
2. Open MATLAB
3. Run:
```matlab
main
```

The script will print threshold analysis and reconstruction errors to the console, and generate all four figures.

**Expected runtime:** ~3–5 minutes for 6000 samples. Reduce `num_samples` to 2000 for faster testing.

## Requirements

- MATLAB R2019b or later
- MNIST dataset (free download) -->

<!-- ## Theory

PCA finds a new coordinate system aligned with the directions of maximum variance in the data. Given a centered data matrix X:

```
X = U × S × V'     (SVD decomposition)

Principal directions : columns of V
Singular values      : diagonal of S
Eigenvalues          : diag(S)² / (N-1)
Projection           : Z = X_centered × V_k
```

The relationship between SVD and the covariance matrix:
- V columns = eigenvectors of X'X/(N-1)
- diag(S)²/(N-1) = eigenvalues of X'X/(N-1)

SVD is used directly rather than computing the covariance matrix explicitly for numerical stability. -->

## Course Information
CE 342 — Computational Methods & Techniques

Ghulam Ishaq Khan Institute of Engineering Sciences and Technology (GIKI)
