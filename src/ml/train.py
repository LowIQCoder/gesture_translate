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
from src.ml.model import GestureModel, model_example_input
from torchinfo import summary
import json
import random


def seed_random(seed: int = 42):
    """
    Seed all random number generators for reproducibility.
    
    Args:
        seed (int): Random seed value. Default is 42.
    """
    # Python random
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # PyTorch CuDNN deterministic settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
            if metric.split("_")[-1] in ["mat", "curve", "lr"]:
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
) -> tuple[dict, dict]:
    """Evaluation function

    Args:
        loss (float): All measurements losses
        total (int): Total measurements
        correct (int): All total classifications
        all_labels (list): All true labels
        all_predicted (list): All predicted labels
        all_probs (list): All predicted probabilities
        batch_times (list): All batches times
        metrics (list): List of needed metrics
        prefix (str): Prefix for metric names (e.g., "train_", "val_")

    Returns:
        tuple[dict, dict]: Tuple of (logged_metrics, non_logged_metrics) dictionaries
    """
    logging_result_metrics = {}
    nonlogging_result_metrics = {}
    
    all_labels_np = np.array(all_labels)
    all_predicted_np = np.array(all_predicted)
    
    if "avg_loss" in metrics:
        logging_result_metrics[prefix + "avg_loss"] = loss / total
    
    if "batch_time" in metrics and batch_times:
        logging_result_metrics[prefix + "batch_time"] = np.mean(batch_times)
    
    if "accuracy" in metrics:
        logging_result_metrics[prefix + "accuracy"] = correct / total
    
    if "precision" in metrics:
        logging_result_metrics[prefix + "precision"] = precision_score(
            all_labels_np, all_predicted_np, average="macro", zero_division=0
        )
    
    if "recall" in metrics:
        logging_result_metrics[prefix + "recall"] = recall_score(
            all_labels_np, all_predicted_np, average="macro", zero_division=0
        )
    
    if "f1" in metrics:
        logging_result_metrics[prefix + "f1"] = f1_score(
            all_labels_np, all_predicted_np, average="macro", zero_division=0
        )

    if "conf_mat" in metrics:
        conf_matrix = confusion_matrix(all_labels_np, all_predicted_np)

        fig = plt.figure(figsize=(20, 20))
        sns.heatmap(conf_matrix, fmt='d', cmap='Blues', annot=False)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')

        nonlogging_result_metrics[prefix + "conf_mat"] = fig
        logging_result_metrics[prefix + "conf_mat_accuracy"] = np.trace(conf_matrix) / np.sum(conf_matrix)
    
    return logging_result_metrics, nonlogging_result_metrics

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
    results[0]["train_lr"] = optimizer.param_groups[0]['lr']

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
    params = {
        "num_classes": 35,
        "d_model": 84,
        "d_hidden": 512,
        "dropout": 0.5
    }

    # Training params
    CHECKPOINT_PATH = "./data/models/best_model.pth"
    MAX_LEARNING_RATE = 5e-5
    DIV_FACTOR = 10
    WEIGHT_DECAY = 1e-2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = 5
    PATIENCE = 10
    BATCH_SIZE = 512
    SEED = 42

    # Fixing random
    seed_random(SEED)

    # Dataoaders
    train_loader, val_loader, test_loader = get_dataloaders("./data/processed/", BATCH_SIZE)

    # Model
    model = GestureModel(**params).to(DEVICE)

    # Loss function
    loss_fn = nn.CrossEntropyLoss()

    # Optimize
    optimizer = optim.AdamW(
        params=model.parameters(),
        lr=MAX_LEARNING_RATE / DIV_FACTOR,
        weight_decay=WEIGHT_DECAY
    )
    
    # Grad scaler
    scaler = torch.amp.GradScaler()

    # Defining scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=MAX_LEARNING_RATE,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.2,
        div_factor=DIV_FACTOR,
        final_div_factor=100
    )

    # Training variables
    best_metrics = {
        "accuracy": 0,
        "loss": 1000
    }
    patience = 0

    # Loading model checkpoint if exists
    try:
        checkpoint = torch.load(CHECKPOINT_PATH)
        best_metrics = checkpoint["best_metrics"]
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optim_state_dict"])
        print(f"Loaded checkpoint with accuracy {best_metrics["accuracy"]:.4f}")
    except:
        print("No checkpoint found. Skipping")

    # Setting up mlflow
    mlflow.set_tracking_uri("http://localhost:5000")

    with mlflow.start_run(log_system_metrics=True) as run:
        # Logging experement parameters
        mlflow.log_params({
            "max_lr": MAX_LEARNING_RATE,
            "div_factor": DIV_FACTOR,
            "weight_decay": WEIGHT_DECAY,
            "device": DEVICE,
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "batch_size": BATCH_SIZE,
            "seed": SEED
        })

        # Logging model information
        model.to("cpu")
        signature = mlflow.models.infer_signature(model_example_input.numpy().astype(np.float32))
        model_info = mlflow.pytorch.log_model(
            model,
            name="model",
            registered_model_name="GestureModel",
            pip_requirements="requirements.txt",
            await_registration_for=0,
            params=params,
            signature=signature
        )
        model.to(DEVICE)

        # Saving model architecture
        with open("./data/models/model_summary.txt", "w", encoding="utf-8") as f:
            f.write(str(summary(model, input_data=model_example_input, device=DEVICE, verbose=0)))
        mlflow.log_artifact("./data/models/model_summary.txt")

        # SAving model configuration
        with open("./data/models/model_configuration.json", "w", encoding="utf-8") as f:
            json.dump(params, f)
        mlflow.log_artifact("./data/models/model_configuration.json")

        # Training and validation
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
                metrics=["avg_loss", "accuracy", "f1", "batch_time"]
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
            if val_metrics[0]["val_avg_loss"] < best_metrics["loss"] or val_metrics[0]["val_accuracy"] > best_metrics["accuracy"]:
                best_metrics["loss"] = val_metrics[0]["val_avg_loss"]
                best_metrics["accuracy"] = val_metrics[0]["val_accuracy"]
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optim_state_dict": optimizer.state_dict(),
                    "best_metrics": best_metrics
                }, CHECKPOINT_PATH)
                patience = 0

                print(f"Epoch {epoch+1:3d}/{EPOCHS} │ Saved new best model with accuracy: { best_metrics["accuracy"]}")
            else:
                patience += 1

            # Logging metrics
            log_metrics([train_metrics[0], val_metrics[0]], epoch_info=(epoch, EPOCHS))
            mlflow.log_metrics(train_metrics[0], step=epoch)
            mlflow.log_metrics(val_metrics[0], step=epoch)

            # Checking for early stopping
            if patience >= PATIENCE:
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

        # Export to ONNX
        torch.onnx.export(
            model,
            model_example_input.to(DEVICE),
            "./data/models/gesture_transformer.onnx",
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output']
        )

        # Logging test metrics
        log_metrics([test_metrics[0]], epoch_info=(EPOCHS+1, EPOCHS))
        mlflow.log_metrics(test_metrics[0])

        # Saving best model
        mlflow.log_artifact("./data/models/best_model.pth")
        mlflow.log_artifact("./data/models/gesture_transformer.onnx")

        # Confussion matrix
        mlflow.log_figure(test_metrics[1]["test_conf_mat"], "confusion_matrix.png")
