import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

from torchvision.models import resnet152, ResNet152_Weights
import torchvision.transforms as transforms
from torch import nn
import torch


def load_cnn():
    cnn = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
    cnn.fc = nn.Identity()
    return cnn


def preprocess_image():
    if not os.path.exists("data/"):
        print("Run 'data_download.sh' first")
        exit(0)
    
    colnames = ["image ID", "caption"]
    captions = pd.read_csv("data/flickr8k/Flickr8k_text/Flickr8k.token.txt", sep="\t", names=colnames, header=None)
    images = captions["image ID"].unique()
    
    cnn = load_cnn()
    cnn.eval()
    
    extracted_dir = "data/extracted/"
    os.mkdir(extracted_dir)
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    for image in tqdm(images):
        img_location = os.path.join("data/flickr8k/Flicker8k_Dataset", image)
        img = Image.open(img_location).convert("RGB")
        img = preprocess(img)
        
        img_feature = cnn(img)
        
        torch.save(img_feature, os.path.join(extracted_dir, image.split('.')[0] + ".pt"))
        