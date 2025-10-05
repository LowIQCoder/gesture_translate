from torch.utils.data import TensorDataset, DataLoader, random_split
from torch import tensor, float32, long
import pandas as pd

from os import PathLike
from typing import Tuple

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
    df = pd.read_csv(path_to_dataset)
    features = df.iloc[:, :-1].values.tolist()
    labels = df.iloc[:, -1].values.tolist()

    dataset = TensorDataset(tensor(features, dtype=float32), tensor(labels, dtype=long))

    train_dataset, val_dataset = random_split(
        dataset,
    [(int(len(dataset) * split)), len(dataset) - int(len(dataset) * split)],
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader
