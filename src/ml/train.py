import torch
import numpy as np
from torch import nn
from torch import optim
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, roc_auc_score, confusion_matrix
import time
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from src.ml.dataloaders import get_dataloaders
from src.ml.model import GestureTransformer
from torchinfo import summary
import json


def log_metrics(
    metrics_list: list[tuple[str, dict]],
    epoch_info: tuple = None,
    logger: logging.Logger = None
) -> None:
    """Log training metrics in a consistent format.
    
    Args:
        epoch_info: Tuple of (current_epoch, total_epochs)
        metrics_list: List of tuples (metric_type, metrics_dict)
        logger: Optional logger instance
        exclude_metrics: List of metric names to exclude from logging
        float_precision: Number of decimal places to display
    """
    # Combining all logs together
    logs = ""
    if epoch_info:
        logs += f"Epoch {epoch_info[0]+1:3d}/{epoch_info[1]} │ "
    for metrics in metrics_list:
        for metric, value in metrics.items():
            if metric in ["conf_mat", "roc_curve"]:
                continue
            logs += f"{metric}: {value:.4f} │ "

    # Logging to logger
    if logger:
        logger.info(logs)
    else:
        print(logs)

def evaluate(
    loss: float,
    total: int,
    correct: int,
    all_labels: list,
    all_predicted: list,
    metrics: list,
    prefix: str,
    batch_times: list = None,
    all_probs: list = None,
) -> dict:
    """Evaluation function

    Args:
        loss (float): All measurements losses
        total (int): Total measurements
        correct (int): All total classifications
        all_labels (list): All true lables
        all_predicted (list): All predicted lables
        all_probs (list): All predicted probabilities
        batches_times (list): All batches times
        metrics (list): List of needed metrics

    Returns:
        dict: Dictionary with required metrics
    """
    result_metrics = dict()
    if "avg_loss" in metrics:
        result_metrics[prefix + "avg_loss"] = loss / total
    if "batch_time" in metrics:
        result_metrics[prefix + "batch_time"] = np.mean(batch_times)
    if "accuracy" in metrics:
        result_metrics[prefix + "accuracy"] = correct / total
    if "precision" in metrics:
        result_metrics[prefix + "precision"] = precision_score(correct, total)
    if "recall" in metrics:
        result_metrics[prefix + "recall"] = recall_score(correct, total)
    if "f1" in metrics:
        result_metrics[prefix + "f1"] = f1_score(all_labels, all_predicted, average="macro")
    if "roc-auc" in metrics:
        result_metrics[prefix + "roc-auc"] = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    if "conf_mat" in metrics:
        result_metrics[prefix + "conf_mat"] = confusion_matrix(all_labels, all_predicted)

    return result_metrics

def train_epoch(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch_info: tuple[int, int],
    scaler: torch.amp.GradScaler = None,
    scheduler: torch.optim.lr_scheduler = None,
    effective_bathes: int = -1,
    grad_clip: float = None,
    metrics: list[str] = ["accuracy"]
) -> None:
    """One epoch of training

    Args:
        model (torch.nn.Module): Model to train
        loss_fn (torch.nn.Module): Used loss function
        optimizer (torch.optim): Used optimizer
        train_loader (torch.utils.data.DataLoader): Train data loader
        device (torch.device): Device to train on
        epoch_info (tuple): Tuple containing (current epoch, total epochs),
        scaler (torch.cuda.amp.GradScaler): Gradient scaler. Optional
        scheduler (torch.optim.lr_scheduler): Used scheduler. Optional
        effective_bathes (int, optional): If gradient accumulation enabled set to effective batch size. Defaults to -1.
        metrics (list): List of requirem metrics. Might be ["accuracy", "precission", "recall", "f1", "batch_time"]
    """
    # Gradient accumulation accumulation step
    accumulation_steps = max(train_loader.batch_size // effective_bathes, 1)

    # Switching to training mode
    model.train()
    train_loss = 0.0
    correct_train = 0
    total_train = 0
    all_logits = []
    all_labels = []
    batch_times = []

    # Training loop
    pbar = tqdm(
        train_loader, desc=f"Epoch {epoch_info[0] + 1:3d}/{epoch_info[1]} │ Training", leave=False)
    optimizer.zero_grad()
    for batch_idx, (features, labels) in enumerate(pbar):
        batch_start = time.time()
        features, labels = features.to(device), labels.to(device)

        # Applying gradient scaling
        if scaler:
            with torch.amp.autocast(device_type=device):
                # Forward pass
                logits = model(features)
                loss = loss_fn(logits, labels)

            # Backward pass
            scaler.scale(loss).backward()

            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            # Forward pass
            logits = model(features)
            loss = loss_fn(logits, labels)

            # Backward pass
            loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        # Gradient clipping
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # Checking loss
        if torch.isnan(loss):
            raise Exception(f"Warning: NaN loss encountered at batch {batch_idx}")

        # LR scheduling
        if scheduler:
            scheduler.step()

        # Collecting metrics
        batch_size = labels.size(0)
        train_loss += loss.item() * batch_size
        total_train += batch_size
        predicted = torch.argmax(logits, dim=1)
        correct_train += (predicted == labels).sum().item()
        all_labels.extend(labels.cpu().numpy())
        all_logits.extend(predicted.cpu().numpy())

        # Measuring batch time
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)

        # Updating progress bar
        pbar.set_postfix({
            "Loss": loss.item(), 
            "lr": optimizer.param_groups[0]["lr"]
        })

    # Calculating metrics
    results = evaluate(
        train_loss, 
        total_train, 
        correct_train, 
        all_labels, 
        all_logits,  
        metrics,
        prefix="train_",
        batch_times=batch_times
    )
    results["train_lr"] = optimizer.param_groups[0]['lr']

    return results

def test_epoch(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch_info: tuple[int, int],
    validation: bool = False,
    metrics: list[str] = ["accuracy"]
) -> None:
    """One validation epoch

    Args:
        model (torch.nn.Module): Model to train
        loss_fn (torch.nn.Module): Used loss function
        val_loader (torch.utils.data.DataLoader): Train data loader
        device (torch.device): Device to train on
        epoch_info (tuple): Tuple containing (current epoch, total epochs),
        metrics (list): List of requirem metrics. Might be ["accuracy", "precission", "recall", "f1", "batch_time"]
    """
    # Switching to evaluation mode
    model.eval()
    val_loss = 0.0
    correct_val= 0
    total_val = 0
    all_logits = []
    all_probabilities = []
    all_labels = []
    batch_times = []

    # Validation loop
    with torch.inference_mode():
        pbar = tqdm(
            test_loader, desc=f"Epoch {epoch_info[0] + 1:3d}/{epoch_info[1]} │ " + "Validation" if validation else "Testing", leave=False)
        for batch_idx, (features, labels) in enumerate(pbar):
            batch_start = time.time()
            features, labels = features.to(device), labels.to(device)

            # Forward pass
            logits = model(features)
            loss = loss_fn(logits, labels)

            # Checking loss
            if torch.isnan(loss):
                raise Exception(f"Warning: NaN loss encountered at batch {batch_idx}")

            # Collecting metrics
            batch_size = labels.size(0)
            val_loss += loss.item() * batch_size
            total_val += batch_size
            probabilities = torch.softmax(logits, dim=1)
            predicted = torch.argmax(logits, dim=1)
            correct_val += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

            # Measuring batch time
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)

            # Updating progress bar
            pbar.set_postfix({
                "Loss": loss.item()
            })

    # Calculating metrics
    results = evaluate(
        val_loss, 
        total_val, 
        correct_val, 
        all_labels, 
        all_logits,
        metrics,
        prefix="val_" if validation else "test_",
        batch_times=batch_times,
        all_probs=all_probabilities
    )

    return results

if __name__ == "__main__":
    # Model params
    NUM_CLASSES = 1001
    DIM_MODEL = 256
    DIM_FF = 512
    NUM_ENCODERS = 6
    NUM_HEADS = 8
    DROPOUT = 0.5

    # Training params
    CHECKPOINT_PATH = "./data/models/best_model.pth"
    INIT_LEARNING_RATE = 5e-5
    MAX_LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 1e-2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = 100
    PATIENCE = 10
    BATCH_SIZE = 256

    # Dataoaders
    train_loader, val_loader = get_dataloaders("./data/processed/train.parquet", "./data/processed/test.parquet", BATCH_SIZE)
    test_loader = val_loader # TODO Fix

    # Model
    model = GestureTransformer(
        num_classes=NUM_CLASSES,
        d_model=DIM_MODEL,
        d_ff=DIM_FF,
        num_encoders=NUM_ENCODERS,
        nheads=NUM_HEADS,       
        dropout=DROPOUT
    ).to(DEVICE)

    optimizer = optim.AdamW(
        params=model.parameters(),
        lr=INIT_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    loss_fn = nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler()

    # Training variables
    best_metrics = {
        "accuracy": 0
    }
    no_improvements = 0

    # Loading model checkpoint if exists
    try:
        checkpoint = torch.load(CHECKPOINT_PATH)
        best_metrics = checkpoint["best_metrics"]
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optim_state_dict"])
        print(f"Loaded checkpoint with accuracy {best_metrics["accuracy"]:.4f}")
    except:
        print("No checkpoint found. Skipping")

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=MAX_LEARNING_RATE,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        div_factor=10,
        final_div_factor=100
    )

    with open("./data/models/model_summary.txt", "w", encoding="utf-8") as f:
        f.write(str(summary(model, input_size=(1, 1, 84))))
    
    with open("./data/models/model_configuration.json", "w", encoding="utf-8") as f:
        NUM_CLASSES = 1001
        DIM_MODEL = 256
        DIM_FF = 512
        NUM_ENCODERS = 6
        NUM_HEADS = 8
        DROPOUT = 0.5
        json.dump({
            "num_classes": 1001,
            "dim_model": 256,
            "dim_ff": 512,
            "num_encoders": 6,
            "num_heads": 8,
            "dropout": 0.5
        }, f)

    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    with mlflow.start_run(log_system_metrics=True) as run:
        # Logging 
        mlflow.log_params({
            "max_lr": MAX_LEARNING_RATE,
            "init_lr": INIT_LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "device": DEVICE,
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "batch_size": BATCH_SIZE
        })

        mlflow.log_artifact("./data/models/model_summary.txt")
        mlflow.log_artifact("./data/models/model_configuration.json")

        # Training and validation
        model.to(DEVICE)
        for epoch in range(EPOCHS):
            # Training
            train_metrics = train_epoch(
                model=model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                train_loader=train_loader,
                device=DEVICE,
                epoch_info=(epoch, EPOCHS),
                scaler=scaler,
                scheduler=scheduler,
                metrics=["avg_loss", "accuracy", "f1"]
            )

            # Validating
            val_metrics = test_epoch(
                model=model,
                loss_fn=loss_fn,
                test_loader=val_loader,
                device=DEVICE,
                epoch_info=(epoch, EPOCHS),
                validation=True,
                metrics=["avg_loss", "accuracy", "f1"]
            )

            # Saving model checkpoint
            if val_metrics["val_accuracy"] > best_metrics["accuracy"]:
                best_metrics["accuracy"] = val_metrics["val_accuracy"]
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optim_state_dict": optimizer.state_dict(),
                    "best_metrics": best_metrics
                }, CHECKPOINT_PATH)
                no_improvements = 0

                print(f"Epoch {epoch+1:3d}/{EPOCHS} │ Saved new best model with accuracy: { best_metrics["accuracy"]}")
            else:
                no_improvements += 1

            # Logging metrics to console
            log_metrics([train_metrics, val_metrics], epoch_info=(epoch, EPOCHS))

            mlflow.log_metrics(train_metrics)
            mlflow.log_metrics(val_metrics)

            # Checking for early stopping
            if no_improvements >= PATIENCE:
                print(f"Epoch {epoch+1:3d}/{EPOCHS} │ No improvements for {PATIENCE} epochs. Stopping")
                break
        
        # Testing
        test_metrics = test_epoch(
            model=model,
            loss_fn=loss_fn,
            test_loader=test_loader,
            device=DEVICE,
            epoch_info=(epoch, EPOCHS),
            validation=False,
            metrics=["avg_loss", "accuracy", "f1", "conf_mat"]
        )

        log_metrics([test_metrics], epoch_info=(EPOCHS+1, EPOCHS))

        mlflow.log_metrics(test_metrics)

        fig = plt.figure(figsize=(20, 20))
        sns.heatmap(test_metrics["test_conf_mat"], fmt='d', cmap='Blues', annot=False)
        plt.xlabel('Predicted')
        plt.ylabel('True')

        mlflow.log_figure(fig)
        
