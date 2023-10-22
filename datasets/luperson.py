# ----------------------------------------------------------------------
# HAP: Structure-Aware Masked Image Modeling for Human-Centric Perception
# Written by Junkun Yuan (yuanjk0921@outlook.com)
# ----------------------------------------------------------------------
# Load LUPerson dataset (without keypoints)
# ----------------------------------------------------------------------
# References:
# MALE: https://github.com/YanzuoLu/MALE
# ----------------------------------------------------------------------

import os
import pickle
import random
from PIL import Image

from torch.utils.data import Dataset


class LUPerson(Dataset):
    """Build LUPerson dataset."""
    def __init__(self, cfg, transform, is_train=True):
        self.transform = transform

        self.ALL_LUPerson_NUM = 4180243
        
        # Ratio < 1: use a certain ratio of data 
        if cfg.DATA.SAMPLE_RATIO < 1:
            num_sample = int(self.ALL_LUPerson_NUM * cfg.DATA.SAMPLE_RATIO)    
        # Ratio > 1: use a certain number of samples
        elif cfg.DATA.SAMPLE_RATIO > 1:
            num_sample = cfg.DATA.SAMPLE_RATIO
        # Ratio = 1: use all samples
        else:
            num_sample = self.ALL_LUPerson_NUM

        self.img_items = []
        # If sample data randomly
        if cfg.DATA.SAMPLE_SOURCE == 'random':
            for root, _, files in os.walk(cfg.DATA.ROOT_DIR):
                for file in files:
                    img_path = os.path.join(root, file)
                    self.img_items.append((img_path,))

            assert len(self.img_items) == self.ALL_LUPerson_NUM

            self.img_items = random.sample(self.img_items, num_sample)
            print(f'=> Use {len(self.img_items)} random LUPerson samples without keypoints')
        
        # If sample data using a list
        elif cfg.DATA.SAMPLE_SOURCE.endswith('.pkl'):
            f = open(cfg.DATA.SAMPLE_SOURCE, 'rb')
            source_list = pickle.load(f)

            assert len(source_list) == self.ALL_LUPerson_NUM

            for file in source_list[:num_sample]:
                img_path = os.path.join(cfg.DATA.ROOT_DIR, file)
                self.img_items.append((img_path,))
            print(f'=> Use {len(self.img_items)} LUPerson samples without keypoints from source {cfg.DATA.SAMPLE_SOURCE}')
        else:
            raise NotImplementedError
    
    def __len__(self):
        return len(self.img_items)
    
    def __getitem__(self, index):
        img_item = self.img_items[index]
        img_path = img_item[0]

        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        
        img = self.transform(img)
        
        return (img, 0, 0)
