import argparse
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np

# --- 1. Dataset Class ---
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.images = sorted(glob.glob(os.path.join(image_dir, '*')))
        self.masks = sorted(glob.glob(os.path.join(mask_dir, '*')))
        
        if len(self.images) != len(self.masks):
            print(f"Warning: Found {len(self.images)} images and {len(self.masks)} masks.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.images[idx]).convert("RGB")
            mask = Image.open(self.masks[idx])
            
            # Resize
            resize = transforms.Resize((256, 256)) 
            image = resize(image)
            mask = resize(mask)

            # Convert
            image = transforms.ToTensor()(image)
            mask = torch.from_numpy(np.array(mask)).long() 
            
            return image, mask
        except Exception as e:
            print(f"Error loading file index {idx}: {e}")
            return torch.zeros(3, 256, 256), torch.zeros(256, 256).long()

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--train_annotation', type=str, default=os.environ.get('SM_CHANNEL_TRAIN_ANNOTATION'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--validation_annotation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION_ANNOTATION'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    
    args, _ = parser.parse_known_args()

    device = torch.device("cpu")
    print(f"Training on device: {device}")
    
    # --- 2. Load Data ---
    train_dataset = SegmentationDataset(args.train, args.train_annotation)
    val_dataset = SegmentationDataset(args.validation, args.validation_annotation)
    
    # drop_last=True ensures we never have a batch of size 1
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        drop_last=True  
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        drop_last=True
    )

    # --- 3. Load MobileNetV3 (num_classes=4) ---
    print(f"Loading MobileNetV3 with num_classes=3...")
    model = models.segmentation.deeplabv3_mobilenet_v3_large(weights=None, num_classes=4)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # --- 4. Training Loop ---
    for epoch in range(args.epochs):
        model.train()
        print(f"--- Epoch {epoch+1}/{args.epochs} ---")
        
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)['out']
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 5 == 0:
                print(f"Batch {batch_idx}: Loss = {loss.item():.4f}")

        # Check if loader was empty (edge case for very small datasets)
        if len(train_loader) > 0:
            epoch_loss = running_loss / len(train_loader)
            print(f"Epoch {epoch+1} Avg Loss: {epoch_loss:.4f}")
        else:
            print("Epoch skipped (Dataset smaller than batch size?)")

    # --- 5. Save Model ---
    path = os.path.join(args.model_dir, 'model.pth')
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

if __name__ == '__main__':
    train()
