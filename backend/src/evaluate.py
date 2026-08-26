from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
from train import load, model, torch, test_loader, device, criterion

import numpy as np


def evaluate():

    load()

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
    print(f"Accuracy:  {accuracy:.4f}")
    
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
    print() 
    print(f"============================================================")

    model.train()

if __name__ == "__main__":
    ...