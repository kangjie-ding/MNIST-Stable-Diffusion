import torch
import os
import sys
from tqdm import tqdm
from torchvision import transforms

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir, "../")
sys.path.insert(0, root_dir)

from models.VAE.vae import VAE
from utils.tools import load_config
from datasets.dataloader import get_dataloader



if __name__=="__main__":
    vae_config_path = root_dir+"/configs/vae_MNIST.json"
    vae_config = load_config(vae_config_path)

    conv_dims = vae_config["model_settings"]["conv_dims"]
    fc_dim = vae_config["model_settings"]["fc_dim"]
    latent_dim = vae_config["model_settings"]["latent_dim"]
    dataset_name = vae_config["data_settings"]["dataset_name"]
    image_size = vae_config["data_settings"]["image_size"]
    input_channels = vae_config["data_settings"]["channels"]
    batch_size = vae_config["data_settings"]["batch_size"]
    data_root = vae_config["path_settings"]["data_root"]


    weights_save_dir = vae_config["path_settings"]["weight_save_dir"]
    weights_save_path = os.path.join(root_dir, weights_save_dir, f"vae_weights_{dataset_name}.pth")
    print(f"Configuration Done!")

    transforms = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
    train_loader = get_dataloader(dataset_name, data_root)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    vae_model = VAE(image_size, input_channels, conv_dims, fc_dim, latent_dim).to(device=device)

    latent_variables = []
    labels = []
    with torch.no_grad():
        for index, (images, targets) in tqdm(enumerate(train_loader), dynamic_ncols=True, total=len(train_loader), desc=f"generating process"):
            images = images.cuda()
            latent = vae_model.get_training_diffusion(images)

            latent_variables.append(latent.cpu())
            labels.append(targets.cpu())
    
    latent_variables = torch.cat(latent_variables, dim=0)
    labels = torch.cat(labels, dim=0)
    
    save_path = root_dir+f"/output/vae_latent_labels_{dataset_name}.pt"
    torch.save({'latent_variables': latent_variables,
                'labels': labels}, save_path)