from torch.utils.data import Dataset, DataLoader, random_split
from torch import tensor, float32, long
from torch.nn.utils.rnn import pad_sequence
import torch
import pandas as pd
import numpy as np

from os import PathLike
from typing import Tuple

class LandmarkDataset(Dataset):
    def __init__(self, dataset_path: str, max_seq_len: int = None):
        self.df = pd.read_parquet(dataset_path)
        if max_seq_len is None:
            max_seq_len = 0
            for row in self.df['features']:
                if len(row) > max_seq_len:
                    max_seq_len = len(row)
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        features = self.df.iloc[idx]['features']
        label = self.df.iloc[idx]['label']
        
        tensors = [torch.from_numpy(a.copy()).float() for a in features]
        row_tensor = torch.stack(tensors)
        
        if len(row_tensor) < self.max_seq_len:
            pad_size = self.max_seq_len - len(row_tensor)
            padded = torch.cat([
                row_tensor, 
                torch.zeros(pad_size, 84)
            ], dim=0)
        elif len(row_tensor) > self.max_seq_len:
            padded = row_tensor[:self.max_seq_len]
        else:
            padded = row_tensor
        
        return padded, label        


def get_dataloaders(
    path_to_dataset: str | PathLike,
    batch_size: int
) -> Tuple[DataLoader, DataLoader]:
    """Generate DataLoaders from preprocessed data

    Args:
        path_to_dataset (str, PathLike): Path to preprocessed dataset in parquet format
        max_seq_len (int): Maximum sequence length (longer sequences will be truncated)
    
    Returns:
        Tuple: Training, Validation and Test DataLoaders
    """
    train_dataset = LandmarkDataset(path_to_dataset + "train.parquet")
    val_dataset = LandmarkDataset(path_to_dataset + "val.parquet")
    test_dataset = LandmarkDataset(path_to_dataset + "test.parquet")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
  
    val_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
    )

    return train_loader, val_loader, test_loader
