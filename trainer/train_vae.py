import torch
import os
import sys
from torchinfo import summary
from contextlib import nullcontext
from tqdm import tqdm

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir, "../")
sys.path.insert(0, root_dir)

from datasets.dataloader import getFlickr30kDataloader, get_dataloader
from utils.tools import load_config, cyclical_kl_weight
from models.VAE.vae import VAE, VGGPerceptualLoss



if __name__=="__main__":
    # vae settings
    vae_config_path = root_dir+"/configs/vae_CIFAR10.json"
    vae_config = load_config(vae_config_path)

    conv_dims = vae_config["model_settings"]["conv_dims"]
    fc_dim = vae_config["model_settings"]["fc_dim"]
    latent_dim = vae_config["model_settings"]["latent_dim"]
    dataset_name = vae_config["data_settings"]["dataset_name"]
    image_size = vae_config["data_settings"]["image_size"]
    input_channels = vae_config["data_settings"]["channels"]
    batch_size = vae_config["data_settings"]["batch_size"]

    epochs = vae_config["training_settings"]["epochs"]
    amp_dtype = vae_config["training_settings"]["amp_dtype"]
    accumulation_steps = vae_config["training_settings"]["accumulation_steps"]
    grad_clip_norm = vae_config["training_settings"]["grad_clip_norm"]
    lr = vae_config["training_settings"]["lr"]

    weights_save_dir = vae_config["path_settings"]["weight_save_dir"]
    weights_save_path = os.path.join(root_dir, weights_save_dir, f"vae_weights_{dataset_name}.pth")
    print(f"Configuration Done!")

    # dataloader
    cifar10_dir = root_dir+"../../../DATASETS"
    cifar10_dataloader = get_dataloader(dataset_name, cifar10_dir, batch_size=batch_size)
    # flickr30k_dir = root_dir+"../../../DATASETS/flickr30k/flickr30k-images"
    # flickr30k_dataloader = getFlickr30kDataloader(flickr30k_dir, batch_size=batch_size)
    print(f"Dataloading Done!")

    # model loading
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_dtype in ["float16", "bfloat16"])
    ctx = nullcontext() if device == "cpu" else torch.cuda.amp.autocast()
    vae_model = VAE(image_size, input_channels, conv_dims, fc_dim, latent_dim).to(device=device)
    summary(vae_model, input_size=(1, input_channels, image_size, image_size))
    if os.path.exists(weights_save_path):
        vae_model.load_state_dict(torch.load(weights_save_path))
        print("Initializing weights of model from existing pth!")
    optimizer = torch.optim.AdamW(vae_model.parameters(), lr=lr)
    # training
    vae_model.train()
    print(f"---starting training---")
    cur_best_training_loss = 22.0
    for epoch in range(epochs):
        total_loss = 0
        kl_weight = 1e-3
        # kl_weight = cyclical_kl_weight(epoch, 1000, 1000, 0.01)
        for step, (images, labels) in tqdm(enumerate(cifar10_dataloader), desc=f"training preocess {epoch+1}/{epochs}", total=len(cifar10_dataloader), dynamic_ncols=True):
            images = images.to(device)
            with ctx:
                recon, mu, log_var = vae_model(images)
                loss = vae_model.compute_loss(images, recon, mu, log_var, beta=kl_weight)
                total_loss += loss.item()
                loss /= accumulation_steps

            scaler.scale(loss).backward()

            if (step+1)%accumulation_steps==0 or step+1==len(cifar10_dataloader):
                scaler.unscale_(optimizer) # unscaling gradient
                # torch.nn.utils.clip_grad_norm_(vae_model.parameters(), grad_clip_norm) # gradient clipping
                scaler.step(optimizer) # optimizer.step()
                scaler.update()  # adjust scaler dynamically
                optimizer.zero_grad(set_to_none=True)

        print(f"{epoch+1}/{epochs}, KL_weight:{kl_weight}, train loss: {total_loss/len(cifar10_dataloader)}")
        if total_loss/len(cifar10_dataloader)<cur_best_training_loss:
            cur_best_training_loss = total_loss/len(cifar10_dataloader)
            torch.save(vae_model.state_dict(), weights_save_path)
        