# Stable Diffusion Implementation on MNIST Dataset

## 📖 Project Overview
In this project, we implement a lightweight **Stable Diffusion framework**.  
The main components include:
- **VAE (Variational Autoencoder)**
- **DDPM (Denoising Diffusion Probabilistic Model)**

We use **CLIP’s text encoder** to extract text features and apply **cross-attention** to generate images conditioned on text.  
The model is trained on the **MNIST dataset** and can generate handwritten digit images based on the input digit text.

## 🚀 Usage
Run the following command to test:
```bash
python ./generating_test.py
```
We also implement DDIM accelerated sampling, which can be invoked as:
```bash
diffusion.ddim_fashion_sample(unet_model, embeddings, 8, sampling_steps=50, channels=channels, batch_size=batch_size)
```

## 📊 Results
On the MNIST dataset, after training for 50 epochs with 1000 timesteps, we obtain the following results using 50-step DDIM sampling:

