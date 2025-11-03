import torch
from torch import nn
from torch import optim
from sklearn.metrics import f1_score
import os

from tqdm import tqdm
from torchinfo import summary

import mlflow

from src.ml.dataloaders import get_dataloaders
from src.ml.model import LandmarkTransformerClassifier

def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: int,
    device: torch.device = 'cpu',
    checkpoint_path: os.PathLike = None
) -> None:    
    """Training loop with logging in to MlFlow

    Args:
        model (nn.Module): Model to train
        optimizer (optim.Optimizer): Used optimizer
        loss_fn (nn.Module): Losss function
        train_loader (torch.utils.data.DataLoader): Train loader
        val_loader (torch.utils.data.DataLoader): Test loader
        epochs (int): Number of epochs
        device (torch.device, optional): Device to train on. Defaults to 'cpu'.
        checkpoint_path (os.PathLike, optional): Path to last model checkpoint. Defaults to None.
    """
    # For storing model checkpoints
    if not os.path.exists("./data/models/"):
        os.makedirs("./data/models/")

    # Loading checkpoint
    best_acc = 0
    if checkpoint_path:
        try:
            checkpoint = torch.load(checkpoint_path)
            
            best_acc = checkpoint['accuracy']
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optim_state_dict'])
            
            with open("./data/model_summary.txt", "w") as f:
                f.write(str(summary(model, input_size=(1, 1, 28, 28))))
            
            print(f"Loaded {checkpoint_path} with accuracy {best_acc}")
        except: 
            print(f"No checkpoint found at {checkpoint_path}. Ignoring")
    model.to(device)

    # Setting mlflow up
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_system_metrics_sampling_interval(1)
    
    if mlflow.active_run():
        mlflow.end_run()

    mlflow.set_experiment("Model Training")

    with mlflow.start_run(log_system_metrics=True):
        # Logging params to MLFlow
        mlflow.log_params({
            "optimizer": optimizer.__class__.__name__,
            "loss_fn": loss_fn.__class__.__name__,
            "epochs": epochs,
            "batch_size": train_loader.batch_size,
            "device": device
        })

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            correct_train = 0
            total_train = 0
            all_preds_train = []
            all_labels_train = []

            # Default training loop
            for features, labels in (pbar := tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}] Training", leave=False)):
                labels, features = labels.to(device), features.to(device)

                optimizer.zero_grad()
                outputs = model(features)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

                batch_size = labels.size(0)
                train_loss += loss.item() * batch_size
                total_train += batch_size

                preds = torch.argmax(outputs, dim=1)
                correct_train += (preds == labels).sum().item()

                all_preds_train.extend(preds.cpu().numpy())
                all_labels_train.extend(labels.cpu().numpy())

                pbar.set_postfix({"loss": loss.item()})

            avg_train_loss = train_loss / total_train
            train_acc = correct_train / total_train
            train_f1 = f1_score(all_labels_train, all_preds_train, average="macro")

            # Evaluating model
            model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            all_preds_val = []
            all_labels_val = []

            # Default validation loop
            with torch.inference_mode():
                for features, labels in (pbar_val := tqdm(val_loader, desc=f"Epoch [{epoch+1}/{epochs}] Validation", leave=False)):
                    labels, features = labels.to(device), features.to(device)
                    outputs = model(features)
                    loss = loss_fn(outputs, labels)

                    batch_size = labels.size(0)
                    val_loss += loss.item() * batch_size
                    total_val += batch_size

                    preds = torch.argmax(outputs, dim=1)
                    correct_val += (preds == labels).sum().item()

                    all_preds_val.extend(preds.cpu().numpy())
                    all_labels_val.extend(labels.cpu().numpy())

                    pbar_val.set_postfix({"val_loss": loss.item()})

            avg_val_loss = val_loss / total_val
            val_acc = correct_val / total_val
            val_f1 = f1_score(all_labels_val, all_preds_val, average="macro")

            # Updating model checkpoint
            if val_acc > best_acc:
                best_acc = val_acc
                
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optim_state_dict": optimizer.state_dict(),
                    "accuracy": val_acc
                }, checkpoint_path)
                    
                print(f"Saved new best model with accuracy: {val_acc}")

            # Logging metrics
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
                f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}"
            )

            # Logging metrics to mlflow
            mlflow.log_metrics({
                "train_loss": avg_train_loss,
                "train_acc": train_acc,
                "train_f1": train_f1,
                "val_loss": avg_val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
            }, step=epoch)

            model_info = mlflow.pytorch.log_model(model, name="model")

        # Saving best model
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        example_inputs = torch.tensor(torch.randn(1, 84),).to(device)
        onnx_program = torch.onnx.export(model, example_inputs, dynamo=True)
        onnx_program.save("./data/models/best_model.onnx")

if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders("./data/processed/features.parquet", 0.8, 64)

    model = LandmarkTransformerClassifier()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss = nn.CrossEntropyLoss()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train(
        model=model,
        optimizer=optimizer,
        loss_fn=loss,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=10,
        device=device,
        checkpoint_path="./data/models/best_model.pth"
    )
