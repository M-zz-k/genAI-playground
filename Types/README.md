# AR Model (AutoRegressive Model)

This folder contains a Jupyter Notebook (`AR_Model.ipynb`) that demonstrates how to build and evaluate an AutoRegressive (AR) model for time series forecasting using the `statsmodels` library in Python.

## Overview
The notebook performs the following steps:
1. **Data Loading**: Loads weather data (`Weather_data.xlsx`) and parses the dates.
2. **Stationarity Check**: Uses the Augmented Dickey-Fuller (ADF) test to check if the temperature time series is stationary.
3. **Differencing**: Applies first-order differencing to make the data stationary if required.
4. **ACF & PACF Plots**: Plots the AutoCorrelation Function (ACF) and Partial AutoCorrelation Function (PACF) to determine the appropriate lag (`p`) for the AR model.
5. **Model Training**: Splits the data into training (80%) and testing (20%) sets, and fits a `statsmodels.tsa.ar_model.AutoReg` model using the determined lag.
6. **Prediction**: Generates predictions for both the training and testing sets.

## Dependencies
- `pandas`
- `matplotlib`
- `numpy`
- `scikit-learn`
- `statsmodels`

## Troubleshooting
If you encounter a `ValueError` regarding broadcasting shapes when calling `model.fit().summary()`, this is a known `statsmodels` index alignment issue with integer-based Pandas Series. To fix it, either:
- Set your datetime column as the DataFrame's index (`df.set_index("Date", inplace=True)`) before differencing.
- Pass the NumPy array directly to the model (`AutoReg(train.values, lags=p)`).

---

# Diffusion Models

This section explores **Diffusion Models** (`Diffusion_Model.ipynb`), a class of generative models that create high-quality data (like images) by learning to reverse a gradual noising process. 

### What is a Diffusion Model?
Unlike GANs (which pit a generator against a discriminator), Diffusion Models work in two main phases:
1. **Forward Process (Adding Noise):** We take a clean image and gradually add Gaussian noise to it over many time steps ($T$) until it becomes pure, unrecognizable noise.
2. **Reverse Process (Denoising):** A neural network (often a U-Net or a Convolutional network) is trained to predict the noise added at each step, learning to gradually clean the noisy image back into a coherent, high-quality image.

Mathematically, the forward process is defined by a variance schedule $\beta_t$. We precompute terms like $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$ to sample noisy images at any timestep $t$ in a single shot:
$$ q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t)\mathbf{I}) $$

---

# 📌 Checkpoint 1: Hyperparameters & Noise Schedule

### 🔹 Goal
Set up the variance schedule ($\beta_t$) that controls how much noise is added at each step.

### 🔹 Code Explanation
- **$T = 1000$**: The total number of diffusion steps.
- **$\beta$**: A linear schedule from `1e-4` to `0.02`, defining the variance of the noise at each step.
- **$\bar{\alpha}$ (`alpha_hat`)**: The cumulative product of $1 - \beta$. This is the magic tensor that allows us to skip steps and jump directly from a clean image ($x_0$) to a noisy image at any arbitrary step $t$.

---

# 📌 Checkpoint 2: Positional Embeddings

### 🔹 Goal
Because the neural network needs to know *which* time step $t$ it is currently denoising, we encode the time step into a vector.

### 🔹 Code Explanation
- **`SinusoidalPositionEmbeddings`**: Inspired by Transformers, this module uses sine and cosine functions of different frequencies to turn an integer time step $t$ into a dense vector (embedding). This embedding is later injected into the denoising network so it can condition its predictions on $t$.

---

# 📌 Checkpoint 3: The Denoising Network

### 🔹 Goal
Define the neural network that will look at a noisy image $x_t$ and the time embedding, and predict the noise that was added.

### 🔹 Code Explanation
- **`Denoiser` (Model)**: A simple Convolutional Neural Network (CNN). 
- **Time Injection**: The time embedding is passed through a small Multi-Layer Perceptron (`self.time_mlp`) and then reshaped so it can be broadcasted and added directly to the image feature maps (`x = x + t`).
- **Output**: The network outputs a tensor of the same shape as the input image, representing the predicted noise.

---

# 📌 Checkpoint 4: The Forward Diffusion Process

### 🔹 Goal
Create a function that simulates the forward process by adding a specific amount of noise to a clean image based on a random time step $t$.

### 🔹 Code Explanation
- **`forward_diffusion(x0, t)`**: Uses the precomputed `alpha_hat` to mathematically blend a clean image (`x0`) with pure Gaussian noise (`torch.randn_like`).
- It returns the corrupted image $x_t$ and the exact noise that was added (which serves as the ground-truth label for the neural network).

---

# 📌 Checkpoint 5: Training Loop

### 🔹 Goal
Train the `Denoiser` network to predict the noise added to the images.

### 🔹 Code Explanation
- We load the **CIFAR-10** dataset and move the images to the GPU.
- For each batch, we sample a random time step $t$ for every image.
- We generate noisy images $x_t$ using the `forward_diffusion` function.
- We pass $x_t$ and $t$ into the `Denoiser` model, which predicts the noise.
- **Loss**: We use Mean Squared Error (`F.mse_loss`) between the *predicted noise* and the *actual noise*.
- The optimizer (`Adam`) updates the network's weights via backpropagation (`loss.backward()`).

✅ **Takeaway:** Once trained, you can start with pure random noise, iteratively pass it through this network 1000 times (subtracting a fraction of the predicted noise each time), and a brand new, generated image will emerge!
