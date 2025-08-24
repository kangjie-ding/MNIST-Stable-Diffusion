import torch
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, CLIPModel

root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)

from models.VAE.vae import VAE
from models.Diffusion.diffusion import GaussianDiffusion
from models.Diffusion.Unet import UNetModel
from utils.tools import load_config

if __name__=="__main__":
    # configuring
    diffusion_config_path = root_dir+"/configs/diffusion_MNIST.json"
    vae_config_path = root_dir+"/configs/vae_MNIST.json"
    diffusion_config = load_config(diffusion_config_path)
    vae_config = load_config(vae_config_path)
    ## diffusion settings
    channel_mul = diffusion_config["model_settings"]["channel_mul_layer"]
    num_head = diffusion_config["model_settings"]["num_head"]
    attention_mul = diffusion_config["model_settings"]["attention_mul"]
    time_steps = diffusion_config["model_settings"]["time_steps"]
    add_2d_rope = diffusion_config["model_settings"]["add_2d_rope"]
    channels = diffusion_config["data_settings"]["channels"]
    dataset_name = diffusion_config["data_settings"]["dataset_name"]
    ## vae settings
    conv_dims = vae_config["model_settings"]["conv_dims"]
    fc_dim = vae_config["model_settings"]["fc_dim"]
    latent_dim = vae_config["model_settings"]["latent_dim"]
    image_size = vae_config["data_settings"]["image_size"]
    print(f"Configuring Done!")
    # model loading
    ## weights path
    vae_weights_path = root_dir+f"/outputs/vae_weights_{dataset_name}.pth"
    diffusion_weights_path = root_dir+f"/outputs/diffusion_weights_{dataset_name}_1000.pth"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    vae_model = VAE(image_size, 1, conv_dims, fc_dim, latent_dim).to(device=device)
    diffusion = GaussianDiffusion(device=device, time_steps=time_steps)
    unet_model = UNetModel(input_channels=channels, output_channels=channels, channel_mul_layer=channel_mul,
                      num_head=num_head, attention_mul=attention_mul, add_2d_rope=add_2d_rope).to(device=device)
    vae_model.load_state_dict(torch.load(vae_weights_path))
    unet_model.load_state_dict(torch.load(diffusion_weights_path))
    vae_model = vae_model.to(device=device)
    unet_model = unet_model.to(device=device)
    vae_model.eval()
    unet_model.eval()
    print(f"Successfully loading models' weights!")
    # CLIP encoder loading
    clip_dir = root_dir+"/models/CLIP/openai/clip-vit-base-patch32"
    tokenizer = AutoTokenizer.from_pretrained(clip_dir)
    clip_model = CLIPModel.from_pretrained(clip_dir)
    clip_model.eval()
    print(f"Successfully initializing CLIP model!")
    # generating
    text_number = ['1', '8', '1', '1',
                   '2', '3', '4', '5',
                   '6', '7', '0', '2',
                   '9', '2', '0', '2']
    batch_size = len(text_number)
    with torch.no_grad():
        inputs = tokenizer(text_number, padding=True, return_tensors="pt")
        embeddings = clip_model.get_text_features(**inputs)
        embeddings = embeddings.unsqueeze(1).to(device=device)
        # generated_latents = diffusion.ddpm_fashion_sample(unet_model, embeddings, 8, channels, batch_size=batch_size)
        generated_latents = diffusion.ddim_fashion_sample(unet_model, embeddings, 8, sampling_steps=50, channels=channels, batch_size=batch_size)
        generated_images = vae_model.decoder(generated_latents.flatten(1))
        generated_images = generated_images.cpu().numpy()
    generated_imgs = np.array(generated_images * 255, dtype=np.uint8).reshape(4, 4, image_size, image_size)
    fig = plt.figure(figsize=(8, 8), constrained_layout=True)
    gs = fig.add_gridspec(4, 4)
    for n_col in range(4):
        for n_row in range(4):
            f_ax = fig.add_subplot(gs[n_row, n_col])
            f_ax.imshow(generated_imgs[n_row, n_col], cmap="gray")
            f_ax.axis("off")
    plt.show()