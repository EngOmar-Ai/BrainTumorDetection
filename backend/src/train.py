from model import WARMUP_EPOCHS, MAIN_EPOCHS, model, optimizer, criterion, device, torch, path, scheduler
from data import train_loader, test_loader

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
import numpy as np

def train():
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

def validate():
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

def save():
    print("Saving Model...")

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }

    torch.save(checkpoint, path)

def load():
    if path.exists():
        print(f"Found checkpoint at {path}, Loading model...")

        data = torch.load(path, map_location=device)

        model.load_state_dict(data['model_state_dict'])
        optimizer.load_state_dict(data['optimizer_state_dict'])
        scheduler.load_state_dict(data['scheduler_state_dict'])

    else:
        print(f"No checkpoint found at {path}, Initializing Default Values")

    print(f"============================================================")

def evaluate():

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

if __name__ == "__main__":
    evaluate()
    train()