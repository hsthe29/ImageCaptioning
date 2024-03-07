import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import pandas as pd

import os
from PIL import Image


def __change_ext(filename, new_ext):
    return filename.split(".")[0] + "." + new_ext


def load_data():
    if not os.path.exists("data/"):
        print("Run 'data_download.sh' first")
        exit(0)
        
    colnames = ["image ID", "caption"]
    captions = pd.read_csv("data/flickr8k/Flickr8k_text/Flickr8k.token.txt", sep="\t", names=colnames, header=None)
    train = pd.read_csv("/content/data/flickr8k/Flickr8k_text/Flickr_8k.trainImages.txt", names=["image ID"])
    dev = pd.read_csv("/content/data/flickr8k/Flickr8k_text/Flickr_8k.devImages.txt", names=["image ID"])
    test = pd.read_csv("/content/data/flickr8k/Flickr8k_text/Flickr_8k.testImages.txt", names=["image ID"])
    
    captions["image ID"] = captions["image ID"].map(lambda x: x[:-2])
    
    train_df = captions[captions["image ID"].isin(train["image ID"])]
    dev_df = captions[captions["image ID"].isin(dev["image ID"])]
    test_df = captions[captions["image ID"].isin(test["image ID"])]
    train_df["image ID"] = train_df["image ID"].map(lambda x: __change_ext(x, "pt"))
    dev_df["image ID"] = dev_df["image ID"].map(lambda x: __change_ext(x, "pt"))
    test_df["image ID"] = test_df["image ID"].map(lambda x: __change_ext(x, "pt"))
    
    return train_df, dev_df, test_df


class FlickrDataset(Dataset):
    """
    FlickrDataset
    """
    def __init__(self, dataframe, image_dir, tokenizer):
        self.image_dir = image_dir

        self.len = len(dataframe)

        self.images = dataframe["image ID"]
        self.captions = dataframe["caption"]
        
        self.tokenizer = tokenizer

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        caption = self.captions[idx]
        img = self.images[idx]
        
        img = torch.load(os.path.join("data/extracted", img))

        caption_ids = self.tokenizer.Encode(caption, add_bos=True, add_eos=True)

        return img, torch.tensor(caption_ids)
    