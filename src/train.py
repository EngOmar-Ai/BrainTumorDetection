from model import WARMUP_EPOCHS, MAIN_EPOCHS, model, optimizer, criterion, device, torch, path, scheduler
from data import train_loader, test_loader

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
import numpy as np

def train() -> None:
    """
    Train and fine-tune the model for the configured number of epochs.

    The function first attempts to restore a previously saved checkpoint and
    then performs the complete training loop. For each epoch, the model
    processes all training batches, calculates the loss, performs
    backpropagation, and updates its parameters using the configured optimizer.

    After each epoch, the learning-rate scheduler is advanced, validation loss
    is calculated, and a comprehensive evaluation is performed. The user is
    also given the option to save the current training state as a checkpoint.

    The total number of training epochs is determined by the combined warmup
    and main training phases.

    Returns:
        None
    """

    print(f"Initializing Training Session... ")

    load()
    model.train()

    for epoch in range(WARMUP_EPOCHS + MAIN_EPOCHS):

        train_loss = 0
        for images, labels in train_loader:

            images, labels = images.to(device), labels.to(device)

            output = model(images)

            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        scheduler.step()

        train_loss = train_loss / len(train_loader)
        validation_loss = validate()

        print(f"Epoch: ({epoch + 1}/{WARMUP_EPOCHS + MAIN_EPOCHS})| Train Loss: {train_loss}| Validation Loss: {validation_loss}")

        evaluate()

        user = input("Do You Want To Save The Model? (y/n) ").strip().lower()
        if user == "y":
            save()

        print(f"============================================================")

def validate() -> float:
    """
    Calculate the average validation loss over the evaluation dataset.

    The model is temporarily switched to evaluation mode and gradient
    computation is disabled to reduce unnecessary memory usage and
    computational overhead. The loss is calculated for every batch in the
    test loader and averaged across the entire dataset.

    The model is returned to training mode before the function completes.

    Returns:
        float: The average cross-entropy loss across all validation batches.
    """

    model.eval()

    validation_loss = 0
    with torch.no_grad():
        for images, labels in test_loader:

            images, labels = images.to(device), labels.to(device)

            output = model(images)
            loss = criterion(output, labels)

            validation_loss += loss.item()

    validation_loss = validation_loss / len(test_loader)

    model.train()

    return validation_loss

def evaluate() -> None:
    """
    Evaluate the model using comprehensive multi-class classification metrics.

    The model is evaluated on the test dataset with gradient computation
    disabled. During evaluation, the function collects predicted classes,
    true labels, and predicted class probabilities for all samples.

    The following metrics are calculated and printed:

    - Average test loss
    - Classification accuracy
    - Macro-averaged precision
    - Macro-averaged recall
    - Macro-averaged F1-score
    - Multi-class AUC-ROC using the One-vs-Rest strategy
    - Per-class precision, recall, F1-score, and sample support
    - Confusion matrix

    Macro averaging gives equal importance to each class regardless of the
    number of samples in that class.

    After evaluation, the model is returned to training mode.

    Returns:
        None
    """

    model.eval()

    test_loss = 0.0
    all_predictions, all_labels, all_probabilities = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:

            images, labels = images.to(device), labels.to(device)

            output = model(images)

            loss = criterion(output, labels)
            test_loss += loss.item()

            probabilities = torch.softmax(output, dim=1)
            all_probabilities.append(probabilities.cpu().numpy())

            predictions = probabilities.argmax(dim=1)
            all_predictions.append(predictions.cpu().numpy())

            all_labels.append(labels.cpu().numpy())

    test_loss = test_loss / len(test_loader)

    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    all_probabilities = np.concatenate(all_probabilities)

    accuracy = accuracy_score(all_labels, all_predictions)

    precision, recall, f1, support = precision_recall_fscore_support(all_labels, all_predictions, average=None, zero_division=0)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(all_labels, all_predictions, average="macro", zero_division=0)

    matrix = confusion_matrix(all_labels, all_predictions)

    auc = roc_auc_score(all_labels, all_probabilities, multi_class="ovr", average="macro")

    print(f"==================== Performance Report ====================")

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Accuracy:  {(accuracy * 100):.4f}%")

    print()

    print(f"Macro Precision: {precision_macro:.4f}")
    print(f"Macro F1: {f1_macro:.4f}")
    print(f"Macro Recall: {recall_macro:.4f}")

    print()

    print(f"AUC-ROC: {auc:.4f}")

    print()

    print(f"Per-class breakdown:")
    print()
    for i in range(len(precision)):
        print(f"  Class {i}: P={precision[i]:.4f}  R={recall[i]:.4f} F1={f1[i]:.4f}  n={support[i]} ")

    print()

    print(f"Confusion Matrix:\n{matrix}")
    print(f"============================================================")

    model.train()

def save() -> None:
    """
    Save the current training state as a checkpoint.

    The checkpoint includes the model parameters, optimizer state, and
    learning-rate scheduler state. Saving these components together allows
    training to be resumed from the same state at a later time.

    Returns:
        None
    """

    print("Saving Model...")

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }

    torch.save(checkpoint, path)

def load() -> None:
    """
    Load a previously saved training checkpoint if one exists.

    The function checks whether the configured checkpoint path exists. If a
    checkpoint is found, it restores the model parameters, optimizer state,
    and learning-rate scheduler state. If no checkpoint is available, the
    current model and training configuration remain unchanged.

    The checkpoint is loaded onto the configured computation device.

    Returns:
        None
    """

    if path.exists():
        print(f"Found checkpoint at {path}, Loading model...")

        data = torch.load(path, map_location=device)

        model.load_state_dict(data['model_state_dict'])
        optimizer.load_state_dict(data['optimizer_state_dict'])
        scheduler.load_state_dict(data['scheduler_state_dict'])

    else:
        print(f"No checkpoint found at {path}, Initializing Default Values")

    print(f"============================================================")

if __name__ == "__main__":
    ...