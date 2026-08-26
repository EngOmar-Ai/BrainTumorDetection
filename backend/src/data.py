from torch.utils.data import DataLoader
from torchvision import datasets, transforms

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root='../../data/train', transform=train_transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

test_dataset = datasets.ImageFolder(root='../../data/test', transform=test_transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True)

if __name__ == "__main__":
    ...