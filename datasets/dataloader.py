import torch
import os
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from tqdm import tqdm
from PIL import Image



class CustomDataset(Dataset):
    def __init__(self, data, labels) -> None:
        super().__init__()
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]
        return sample, label
    
def getMnistLatentDataloader(dataset_path, clip_model, tokenizer, batch_size=128):
    assert os.path.exists(dataset_path), f"dataset path {dataset_path} does not exist!"
    dataset_dict = torch.load(dataset_path)
    assert 'latent_variables' in dataset_dict.keys()
    assert 'labels' in dataset_dict.keys()
    clip_model.eval()
    data = dataset_dict['latent_variables']
    labels1 = dataset_dict['labels'][:20000]
    labels2 = dataset_dict['labels'][20000:40000]
    labels3 = dataset_dict['labels'][40000:]
    labels_list = [labels1, labels2, labels3]
    # get text embeddings and load them
    chunk_size = 250
    text_label_list = [[str(i.item()) for i in labels] for labels in labels_list]
    chunks = [[text_label_list[index][i:i+chunk_size] for i in range(0, len(text_label_list[index]), chunk_size)] for index in range(3)]
    text_embeddings = []
    for split_index in range(3):
        processed_chunks  =[]
        for index, chunk_list in tqdm(enumerate(chunks[split_index]),desc=f"text encoding {split_index+1}/{len(labels_list)}", dynamic_ncols=True, total=len(chunks[split_index])):
            inputs = tokenizer(chunk_list, padding=True, return_tensors="pt")
            embeddings = clip_model.get_text_features(**inputs)
            processed_chunks.append(embeddings)
        text_embeddings.append(torch.cat(processed_chunks, dim=0).detach())

    dataset = CustomDataset(data, torch.cat(text_embeddings, dim=0))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader

def get_dataloader(name, root, transform=None, batch_size=64, split="train", download=False):
    assert name in ["CelebA", "MNIST", "CIFAR10"], f"{name}:Unknown dataset name!"
    if transform is None:
        transform=transforms.Compose([transforms.ToTensor()])
    datasets = None
    if name=="CelebA":
        datasets = torchvision.datasets.CelebA(root, transform=transform, split=split, download=download)
    elif name=="MNIST":
        datasets = torchvision.datasets.MNIST(root, train=(split=="train"), transform=transform, download=download)
    elif name=="CIFAR10":
        datasets = torchvision.datasets.CIFAR10(root, train=(split=="train"), transform=transform, download=download)
    else:
        pass
    assert datasets is not None, "Empty Datasets!"
    dataloader = torch.utils.data.DataLoader(datasets, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader

class Flickr30kDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_files = [
            os.path.join(folder_path, f) 
            for f in os.listdir(folder_path) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        self.image_files = self.image_files[:10]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

def getFlickr30kDataloader(folder_path, transform=None, batch_size=32):
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    dataset = Flickr30kDataset(folder_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataloader