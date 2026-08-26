from model import WARMUP_EPOCHS, MAIN_EPOCHS, model, optimizer, criterion, device, torch, path, scheduler
from data import train_loader, test_loader

def train():
    print(f"Initializing Training Session...")

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

        user = input("Do You Want To Save The Model? (y/n) ").strip().lower()
        if user == "y":
            save()

        print(f"============================================================\n")

def validate():
    model.eval()

    validation_loss = 0
    with torch.no_grad():
        for images, labels in test_loader:

            images, labels = images.to(device), labels.to(device)

            output = model(images)
            loss = criterion(output, labels)

            validation_loss += loss.item()

    validation_loss = validation_loss / len(train_loader)

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

if __name__ == "__main__":
    ...