import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
from torchvision import transforms

class ODIRDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.classes = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row['filename']
        if not filename.endswith('.jpg'):
            filename += '.jpg'
        img_path = os.path.join(self.img_dir, filename)
        
        image = Image.open(img_path).convert('RGB')
        labels = torch.tensor(row[self.classes].values.astype('float32'))

        if self.transform:
            image = self.transform(image)

        return image, labels

def prepare_odir_df(csv_path):
    full_df = pd.read_csv(csv_path)
    
    # Separate left and right eye columns into a unified 'filename' column
    left = full_df[['Left-Fundus', 'N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']].copy()
    left.columns = ['filename'] + ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
    
    right = full_df[['Right-Fundus', 'N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']].copy()
    right.columns = ['filename'] + ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
    
    return pd.concat([left, right], ignore_index=True)