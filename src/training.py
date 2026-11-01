from pathlib import Path
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
import wandb

from torch.utils.data import DataLoader

from src.models import RNNModel, WordEmbeddingModel
from src.preprocessing import collate_batch


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in train_loader:
        labels = batch["label"].to(device)

        batch = {
            key: value.to(device)
            for key, value in batch.items()
            if key != "label"
        }

        optimizer.zero_grad()

        logits = model(batch)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def validate(model, valid_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in valid_loader:
        labels = batch["label"].to(device)

        batch = {
            key: value.to(device)
            for key, value in batch.items()
            if key != "label"
        }

        logits = model(batch)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def run_experiment(
    train_encoded,
    valid_encoded,
    vocab,
    project_root,
    device,
    batch_size=256,
    learning_rate=0.001,
    epochs=60,
    patience=5,
    seed=42,
    dropout=0.0,
    weight_decay=0.01,
    use_cache=True,
    model_type="word_embeddings"
):

    # Unterstützte Modelltypen: Kürzel für den W&B-Run-Namen und Klassenname
    model_types = {
        "word_embeddings": ("WE", "WordEmbeddingModel"),
        "rnn": ("RNN", "RNNModel"),
    }

    if model_type not in model_types:
        raise ValueError(
            f"Unbekannter Modelltyp: {model_type!r}. "
            f"Erlaubt sind: {', '.join(model_types)}"
        )

    run_prefix, model_name = model_types[model_type]

    # 0. Gespeichertes Ergebnis verwenden, falls das Experiment bereits durchgeführt wurde.
    #    Dadurch wird beim erneuten Ausführen des Notebooks weder trainiert
    #    noch ein neuer W&B-Run erstellt.
    #    Der Modelltyp steht am Anfang der ID ("word_embeddings_..." bzw. "rnn_..."),
    #    dadurch bleiben die bisherigen Word-Embedding-IDs unverändert.
    experiment_id = (
        f"{model_type}_bs{batch_size}_"
        f"lr{learning_rate:g}_do{dropout:g}_"
        f"wd{weight_decay:g}_ep{epochs}_p{patience}_s{seed}"
    )

    result_path = Path(project_root) / "outputs" / "results" / f"{experiment_id}.json"

    if use_cache and result_path.exists():
        try:
            cached = json.loads(result_path.read_text(encoding="utf-8"))
            result = cached["result"]
            checkpoint = Path(result["checkpoint_path"])

            # Relative Pfade beziehen sich auf das Projektverzeichnis
            if not checkpoint.is_absolute():
                checkpoint = Path(project_root) / checkpoint

            if checkpoint.is_file():
                print(cached["log"], end="")
                print(f"\n(Gespeichertes Ergebnis geladen: {result_path.name})")
                return result

            print("Checkpoint fehlt – Experiment wird erneut trainiert.")

        except (OSError, ValueError, KeyError, TypeError):
            print("Cache ungültig – Experiment wird erneut trainiert.")

    log_lines = []

    def log(line=""):
        print(line)
        log_lines.append(line)

    # 1. Zufallswerte setzen
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 2. DataLoader erstellen
    generator = torch.Generator().manual_seed(seed)

    experiment_train_loader = DataLoader(
        train_encoded,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        generator=generator
    )

    experiment_valid_loader = DataLoader(
        valid_encoded,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch
    )

    # 3. Modell, Loss und Optimizer
    if model_type == "rnn":
        experiment_model = RNNModel(
            vocab_size=len(vocab),
            embedding_dim=50,
            hidden_dim=64,
            dropout=dropout
        ).to(device)
    else:
        experiment_model = WordEmbeddingModel(
            vocab_size=len(vocab),
            embedding_dim=50,
            dropout=dropout
        ).to(device)

    criterion = nn.CrossEntropyLoss()

    experiment_optimizer = torch.optim.AdamW(
        experiment_model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # 4. W&B-Run starten
    run = wandb.init(
        project="NLP-Project-1",
        name=(
            f"{run_prefix} | BS{batch_size} | LR{learning_rate:g} | "
            f"DO{dropout:g} | WD{weight_decay:g} | "
            f"EP{epochs} | P{patience} | S{seed}"
        ),
        config={
            "model": model_name,
            "model_type": model_type,
            "batch_size": batch_size,
            "epochs": epochs,
            "patience": patience,
            "embedding_dim": 50,
            "hidden_dim": 64,
            "learning_rate": learning_rate,
            "dropout": dropout,
            "weight_decay": weight_decay,
            "seed": seed
        }
    )

    # 5. Checkpoint-Pfad
    checkpoint_dir = Path(project_root) / "outputs" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"{experiment_id}.pt"

    best_valid_acc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0

    start_time = time.perf_counter()

    # 6. Training
    try:
        for epoch in range(epochs):

            train_loss, train_acc = train_one_epoch(
                experiment_model,
                experiment_train_loader,
                experiment_optimizer,
                criterion,
                device
            )

            valid_loss, valid_acc = validate(
                experiment_model,
                experiment_valid_loader,
                criterion,
                device
            )

            # Bestes Modell speichern
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_epoch = epoch + 1
                epochs_without_improvement = 0

                torch.save({
                    "epoch": best_epoch,
                    "model_state_dict": experiment_model.state_dict(),
                    "optimizer_state_dict": experiment_optimizer.state_dict(),
                    "valid_accuracy": best_valid_acc,
                    "model_type": model_type,
                    "embedding_dim": 50,
                    "hidden_dim": 64,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "dropout": dropout,
                    "weight_decay": weight_decay,
                    "vocab": vocab
                }, checkpoint_path)

            else:
                epochs_without_improvement += 1

            # Metriken protokollieren
            run.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "valid_loss": valid_loss,
                "valid_accuracy": valid_acc
            })

            log(
                f"BS {batch_size} | DO {dropout} | "
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Acc: {train_acc:.2%} | "
                f"Valid Acc: {valid_acc:.2%}"
            )

            # Early Stopping
            if epochs_without_improvement >= patience:
                log(f"Early Stopping nach Epoche {epoch+1}")
                break

        elapsed = time.perf_counter() - start_time

        run.summary["best_valid_accuracy"] = best_valid_acc
        run.summary["best_epoch"] = best_epoch
        run.summary["training_time_seconds"] = elapsed

        log()
        log(f"Beste Validation Accuracy: {best_valid_acc:.2%}")
        log(f"Beste Epoche: {best_epoch}")
        log(f"Trainingszeit: {elapsed:.1f} Sekunden")

        result = {
            "model_type": model_type,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "dropout": dropout,
            "weight_decay": weight_decay,
            "best_valid_accuracy": best_valid_acc,
            "best_epoch": best_epoch,
            "training_time_seconds": elapsed,
            "checkpoint_path": str(checkpoint_path)
        }

        # Ergebnis und Trainingsverlauf für spätere Notebook-Ausführungen speichern
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(
                {"result": result, "log": "\n".join(log_lines) + "\n"},
                indent=2,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )

        return result

    finally:
        run.finish()