import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import os
from tqdm import tqdm 

print(torch.cuda.is_available())

image_size = 96
batch_size = 32
num_epochs = 50
num_classes = 2
lr = 1e-4
patience = 5
device = torch.device("cuda" if torch.cuda.is_available() else  "cpu")

print(device)

train_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

val_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_dataset = datasets.ImageFolder(
    r'dataset\train',
    transform=train_transform
)


val_dataset = datasets.ImageFolder(
    r'dataset\val',
    transform=val_transform
)
if __name__ == "__main__":

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    
    model = timm.create_model(
        'convnextv2_tiny.fcmae_ft_in1k',
        pretrained=True,
        num_classes=num_classes
    )
    
    model.stem[0] = nn.Conv2d(1, model.stem[0].out_channels, kernel_size=4, stride=2, padding=0, bias=True)

    model.stages[3].downsample = nn.Sequential(
        nn.Conv2d(384, 768, kernel_size=1),
        nn.BatchNorm2d(768)
    )
    
    
    model.to(device).to(memory_format=torch.channels_last)
    print(next(model.parameters()).device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW([
        {'params': model.stem.parameters(), 'lr': lr*5},
        {'params': [p for n, p in model.named_parameters() if "stem" not in n], 'lr': lr}
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    
    scaler = torch.cuda.amp.GradScaler()
    
    best_val_acc = 0.0
    epochs_no_improve = 0
    
    os.makedirs("models", exist_ok=True)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss, correct = 0.0, 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [TRAIN]", leave=False)
        
        for inputs, targets in loop:
            inputs, targets = inputs.to(device).to(memory_format=torch.channels_last), targets.to(device)
            optimizer.zero_grad()
    
            # --- Mixed precision ---
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
    
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    
            train_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == targets).sum().item()
            loop.set_postfix(loss=loss.item())
    
        train_loss /= len(train_dataset)
        train_acc = correct / len(train_dataset)
    
        model.eval()
        val_loss, val_correct = 0.0, 0
        loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [VAL]", leave=False)
        with torch.no_grad():
            for inputs, targets in loop:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
    
                val_loss += loss.item() * inputs.size(0)
                val_correct += (outputs.argmax(1) == targets).sum().item()
    
        val_loss /= len(val_dataset)
        val_acc = val_correct / len(val_dataset)
    
        scheduler.step(val_acc)
    
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        
        
            
        with torch.no_grad():
            stem_norm = 0.0
            for p in model.stem.parameters():
                stem_norm += torch.norm(p).item()
            
            
            print(f"Norme des poids du stem : {stem_norm:.4f}")
            
            
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), "models/best_convnextv2_tiny_vehicle_classifier_mod.pth")
            print(f"Nouveau meilleur modèle sauvegardé avec Val Acc={val_acc:.4f}")
        else:
            epochs_no_improve += 1
            print(f"Pas d'amélioration ({epochs_no_improve}/{patience})")
    
        if epochs_no_improve >= patience:
            print("Early stopping déclenché !")
            break
    
