import os
import torch
import torch.amp
import torch.nn as nn
import torch.utils
import torch.utils.data
import json
import sys
from contextlib import nullcontext
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchinfo import summary
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPModel

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir, "../")
sys.path.insert(0, root_dir)

from datasets.dataloader import getMnistLatentDataloader
from models.Diffusion.diffusion import GaussianDiffusion
from models.Diffusion.Unet import UNetModel
from utils.tools import load_config




if __name__=="__main__":
    # configuring
    config_file_path = root_dir+"/configs/diffusion_mnist.json"
    model_config = load_config(config_file_path)

    # model settings
    time_steps = model_config["model_settings"]["time_steps"]
    channel_mul = model_config["model_settings"]["channel_mul_layer"]
    attention_mul = model_config["model_settings"]["attention_mul"]
    num_head = model_config["model_settings"]["num_head"]
    add_2d_rope = model_config["model_settings"]["add_2d_rope"]
    text_embedding_dim = model_config["model_settings"]["text_embedding_dim"]

    # training settings
    lr = model_config["training_settings"]["lr"]
    epochs = model_config["training_settings"]["epochs"]
    amp_dtype = model_config["training_settings"]["amp_dtype"]
    accumulation_steps = model_config["training_settings"]["accumulation_steps"]
    grad_clip_norm = model_config["training_settings"]["grad_clip_norm"]
    
    # data settings
    batch_size = model_config["data_settings"]["batch_size"]
    channels = model_config["data_settings"]["channels"]
    latent_dim = model_config["data_settings"]["latent_dim"]
    dataset_name = model_config["data_settings"]["dataset_name"]

    # paths settings
    log_root = model_config["path_settings"]["log_dir"]
    log_dir = os.path.join(log_root, "train_diffusion")
    model_save_dir = model_config["path_settings"]["weight_save_dir"]
    model_output_path = os.path.join(root_dir, model_save_dir, f"diffusion_weights_{dataset_name}_{time_steps}.pth")

    print(f"Configuration Done!")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ctx = nullcontext() if device == "cpu" else torch.cuda.amp.autocast()
    # fix random seed
    base_seed = 1337
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)

    # dataloader
    clip_model = CLIPModel.from_pretrained(root_dir+"/models/CLIP/openai/clip-vit-base-patch32")
    tokenizer = AutoTokenizer.from_pretrained(root_dir+"/models/CLIP/openai/clip-vit-base-patch32")
    dataset_path = root_dir+f"/outputs/vae_latent_labels_{dataset_name}.pt"
    train_dataloader = getMnistLatentDataloader(dataset_path, clip_model, tokenizer, batch_size)
    print(f"Dataloading Done!")
    # loading models
    unet = UNetModel(input_channels=channels, output_channels=channels, attention_mul=attention_mul, channel_mul_layer=channel_mul,
                      text_embedding_dim=text_embedding_dim, num_head=num_head, add_2d_rope=add_2d_rope).to(device=device)
    diffusion = GaussianDiffusion(device=device, time_steps=time_steps)

    # initialize weights if possible
    if os.path.exists(model_output_path):
        print("Initializing weights of model from existing pth!")
        unet.load_state_dict(torch.load(model_output_path))

    # define optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_dtype in ["float16", "bfloat16"])

    # define summary writer
    writer = SummaryWriter(log_dir=log_dir)
    summary(unet, input_size=[(1, channels, 8, 8), (1, 1, 512), (1, 1)])
    # training
    cur_min_loss = float('inf')
    print(f"---starting training---")
    unet.train()
    for epoch in range(epochs):
        total_loss = 0
        for step, (images, text_embeddings) in tqdm(enumerate(train_dataloader), desc="training preocess", total=len(train_dataloader), dynamic_ncols=True):
            images = images.to(device).reshape(-1, 1, 8, 8)
            t = torch.randint(1, time_steps, (images.shape[0], 1), device=device)
            text_embeddings = text_embeddings.to(device=device).unsqueeze(1)
            with ctx:
                loss = diffusion.get_train_loss(unet, images, text_embeddings, t)
                total_loss += loss.item()
                loss /= accumulation_steps

            scaler.scale(loss).backward()

            if (step+1)%accumulation_steps==0 or step+1==len(train_dataloader):
                scaler.unscale_(optimizer) # unscaling gradient
                # torch.nn.utils.clip_grad_norm_(unet.parameters(), grad_clip_norm) # gradient clipping
                scaler.step(optimizer) # optimizer.step()
                scaler.update()  # adjust scaler dynamically
                optimizer.zero_grad(set_to_none=True)

        print(f"{epoch+1}/{epochs}, train loss: {total_loss/len(train_dataloader)}")
        writer.add_scalar("training loss", total_loss/len(train_dataloader), epoch)
        if total_loss<cur_min_loss:
            cur_min_loss = total_loss
            torch.save(unet.state_dict(), model_output_path)
    writer.close()
