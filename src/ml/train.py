import torch
from torch import nn
from torch import optim
from sklearn.metrics import f1_score

from tqdm import tqdm

import mlflow

from src.ml.dataloaders import get_dataloaders
from src.ml.model import Net

def train(
    model,
    optimizer,
    loss_fn,
    train_loader,
    val_loader,
    epochs,
    device,
    checkpoint
) -> None:
    best_acc = 0

    if checkpoint:
        try:
            model.load_state_dict(torch.load(checkpoint))
            with open("./data/models/metrics.txt", "r") as f:
                best_acc = float(f.read().split()[-1])
            print(f"Loaded {checkpoint} with accuracy {best_acc}")
        except:
            print(f"No checkpoint found at {checkpoint}. Ignoring")

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.enable_system_metrics_logging()
    mlflow.set_system_metrics_sampling_interval(1)
    
    if mlflow.active_run():
        mlflow.end_run()

    mlflow.set_experiment("Model Training")

    model.to(device)
    with mlflow.start_run(log_system_metrics=True):
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

            pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}] Training", leave=False)
            for features, labels in pbar:
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

            model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            all_preds_val = []
            all_labels_val = []

            with torch.inference_mode():
                pbar_val = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{epochs}] Validation", leave=False)
                for features, labels in pbar_val:
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

            if val_acc > best_acc:
                print(f"Saving best model Accuracy: {val_acc}")
                torch.save(model.state_dict(), checkpoint)
                with open("./data/models/metrics.txt", "w") as f:
                    f.write(f"Accuracy: {val_acc}")

            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
                f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}"
            )

            mlflow.log_metrics({
                "train_loss": avg_train_loss,
                "train_acc": train_acc,
                "train_f1": train_f1,
                "val_loss": avg_val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
            }, step=epoch)

        example_inputs = torch.tensor(torch.randn(1, 84),).to(device)
        onnx_program = torch.onnx.export(model, example_inputs, dynamo=True)
        onnx_program.save("./data/models/best_model.onnx")

if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders("./data/processed/hand_landmarks_features.csv", 0.8, 64)

    model = Net()
    optimizer = optim.Adam(model.parameters())
    loss = nn.CrossEntropyLoss()

    train(
        model=model,
        optimizer=optimizer,
        loss_fn=loss,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=5,
        device='cuda',
        checkpoint="./data/models/best_model.pth"
    )
