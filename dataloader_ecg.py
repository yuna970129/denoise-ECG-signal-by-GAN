import torch
# import torch.utils.data as data
import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class BasicDataset(object):
    def __init__(self, root_dir):
        super(BasicDataset, self).__init__()

        self.root_dir = root_dir
        
        self.x_dir = os.path.join(root_dir, 'clean_half.csv')
        self.y_dir = os.path.join(root_dir, 'noisy_half.csv')

        self.x_file = pd.read_csv(self.x_dir)
        self.y_file = pd.read_csv(self.y_dir)
        
    def __getitem__(self, index):
        
        x = np.array(self.x_file.iloc[index], dtype=np.double)
        y = np.array(self.y_file.iloc[index], dtype=np.double)
        return x, y
        
    def __len__(self):
        return len(self.x_file)
