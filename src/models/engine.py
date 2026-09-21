import torch
from src import logger


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device
):
    model.train()

    total_loss = 0.0

    for X_batch, y_batch in dataloader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        logits = model(X_batch)

        loss = criterion(
            logits,
            y_batch
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item() * X_batch.size(0)
        )

    return total_loss / len(dataloader.dataset)


def validate(
    model,
    dataloader,
    criterion,
    device
):
    model.eval()

    total_loss = 0.0

    with torch.no_grad():

        for X_batch, y_batch in dataloader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)

            loss = criterion(
                logits,
                y_batch
            )

            total_loss += (
                loss.item() * X_batch.size(0)
            )

    return total_loss / len(dataloader.dataset)


def fit(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs,
    checkpoint_path
):

    best_val_loss = float("inf")

    logger.info(
        "Starting model training for %d epochs",
        epochs
    )

    for epoch in range(epochs):

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss = validate(
            model,
            val_loader,
            criterion,
            device
        )

        logger.info(
            "Epoch %d/%d | Train Loss: %.4f | Val Loss: %.4f",
            epoch + 1,
            epochs,
            train_loss,
            val_loss
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": model.network[0].in_features,
                    "val_loss": val_loss
                },
                checkpoint_path
            )

            logger.info(
                "New best model saved | Val Loss: %.4f",
                val_loss
            )

    logger.info("Training completed")


def predict(
    model,
    dataloader,
    device
):

    model.eval()

    predictions = []
    probabilities = []

    with torch.no_grad():

        for X_batch, _ in dataloader:

            X_batch = X_batch.to(device)

            logits = model(X_batch)

            probs = torch.sigmoid(logits)

            preds = (
                probs >= 0.5
            ).int()

            probabilities.extend(
                probs.cpu().numpy().flatten()
            )

            predictions.extend(
                preds.cpu().numpy().flatten()
            )

    return predictions, probabilities