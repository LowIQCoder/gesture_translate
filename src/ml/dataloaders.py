from torch.utils.data import Dataset, DataLoader, random_split
from torch import tensor, float32, long
from torch.nn.utils.rnn import pad_sequence
import torch
import pandas as pd
import numpy as np

from os import PathLike
from typing import Tuple

class GestureDataset(Dataset):
    def __init__(self, parquet_path: str):
        self.df = pd.read_parquet(parquet_path)
        self.df["features"] = self.df["features"].apply(lambda x: [list(f) for f in x])
        self.df = self.df[self.df["features"].apply(lambda x: len(x) > 0)].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        seq = self.df.iloc[idx]["features"]

        seq_array = np.array(seq, dtype=np.float32)

        tensor_seq = torch.tensor(seq_array, dtype=torch.float32)  # (seq_len, 84)
        label = int(self.df.iloc[idx]["label"])
        return tensor_seq, label

def collate_fn(batch):
    # (tensor, label)
    sequences, labels = zip(*batch)
    padded = pad_sequence(sequences, batch_first=True, padding_value=1002)  # (batch, max_seq_len, 84)
    labels = torch.tensor(labels, dtype=torch.long)
    return padded, labels


def get_dataloaders(
    path_to_dataset: str | PathLike,
    split: float,
    batch_size: int
) -> Tuple[DataLoader, DataLoader]:
    """Generate DataLoaders from preprocessed data

    Args:
        path_to_dataset (str, PathLike): Path to preprocessed dataset in csv format
        split (float): Split ratio of training and validating datasets
        batch_size (int): Size of batch for DataLoader
    
    Returns:
        Tuple: Training and Validation DataLoaders
    """
    dataset = GestureDataset(path_to_dataset)

    train_dataset, val_dataset = random_split(
        dataset,
    [(int(len(dataset) * split)), len(dataset) - int(len(dataset) * split)],
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        collate_fn=collate_fn, 
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        collate_fn=collate_fn
    )

    return train_loader, val_loader
