from torch.utils.data import Dataset, DataLoader, random_split
from torch import tensor, float32, long
from torch.nn.utils.rnn import pad_sequence
import torch
import pandas as pd
import numpy as np

from os import PathLike
from typing import Tuple

class GestureDataset(Dataset):
    def __init__(self, parquet_path: str, max_seq_len=256):
        self.df = pd.read_parquet(parquet_path)
        self.df["features"] = self.df["features"].apply(lambda x: [list(f) for f in x])
        self.df = self.df[self.df["features"].apply(lambda x: len(x) > 0)].reset_index(drop=True)
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        seq = self.df.iloc[idx]["features"]
        
        # Convert to numpy and ensure it's 2D
        seq_array = np.array(seq, dtype=np.float32)
        
        # Truncate if sequence is too long
        if len(seq_array) > self.max_seq_len:
            seq_array = seq_array[:self.max_seq_len]
        
        tensor_seq = torch.tensor(seq_array, dtype=torch.float32)  # (seq_len, 84)
        label = int(self.df.iloc[idx]["label"])
        return tensor_seq, label

def collate_fn(batch):
    sequences, labels = zip(*batch)
    
    # Get original lengths before padding
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    
    # Pad sequences - use 0.0 as it's a safe padding value for normalized coordinates
    padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)  # (batch, max_seq_len, 84)
    
    # Create attention mask: 1 for real data, 0 for padding
    # Shape: (batch, seq_len)
    mask = torch.zeros(len(sequences), padded.shape[1], dtype=torch.bool)
    for i, length in enumerate(lengths):
        mask[i, :length] = 1
    
    labels = torch.tensor(labels, dtype=torch.long)
    return padded, labels, mask


def get_dataloaders(
    path_to_dataset: str | PathLike,
    split: float,
    batch_size: int,
    max_seq_len: int = 256
) -> Tuple[DataLoader, DataLoader]:
    """Generate DataLoaders from preprocessed data

    Args:
        path_to_dataset (str, PathLike): Path to preprocessed dataset in parquet format
        split (float): Split ratio of training and validating datasets
        batch_size (int): Size of batch for DataLoader
        max_seq_len (int): Maximum sequence length (longer sequences will be truncated)
    
    Returns:
        Tuple: Training and Validation DataLoaders
    """
    dataset = GestureDataset(path_to_dataset, max_seq_len=max_seq_len)

    # Calculate split sizes
    train_size = int(len(dataset) * split)
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        collate_fn=collate_fn, 
        shuffle=True,
        #num_workers=2,  # Add for better performance
        #pin_memory=True  # Add if using GPU
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        collate_fn=collate_fn,
        #num_workers=2,
        #pin_memory=True
    )

    return train_loader, val_loader