import argparse
from pathlib import Path
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision import models
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report

# --- Configuration ---
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.0001  # Lowered for fine-tuning unfrozen layers
RANDOM_STATE = 42
PATIENCE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = Path("D:/ESS_AI_Project/ESS_Final/Model_Save/final_efficientnet_b0_cv.pth")
REPORT_SAVE_PATH = Path("D:/ESS_AI_Project/ESS_Final/Model_Saves/final_classification_report_cv.txt")
CLASS_WEIGHTS = torch.tensor([1.0, 1.5]).to(DEVICE)  # Higher weight for HG

# --- Patch Dataset ---
class PatchDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.valid_indices = []
        # Pre-check valid images
        for idx in range(len(df)):
            img_path = str(df.iloc[idx]['path'])
            img = cv2.imread(img_path)
            if img is not None:
                self.valid_indices.append(idx)
            else:
                print(f"⚠️ Skipping invalid image: {img_path}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        img_path = str(self.df.iloc[actual_idx]['path'])
        label = self.df.iloc[actual_idx]['label']
        
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

# --- The State-of-the-Art Model: EfficientNet ---
def create_efficientnet_model(num_classes=2):
    """Loads a pre-trained EfficientNet-B0 and adapts it for our task."""
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    
    # Unfreeze last two feature blocks
    for param in model.features[-2:].parameters():
        param.requires_grad = True
        
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    print("✅ Loaded EfficientNet-B0 with partial unfreezing.")
    return model

# --- Training Logic per Fold ---
def train_model(train_loader, test_loader, model, criterion, optimizer, epochs, patience, device):
    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
        val_loss /= len(test_loader.dataset)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    return model

# --- Main Training Logic with 5-Fold Cross-Validation ---
def main():
    # Create directories if they don't exist
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # --- 1. Load and Balance Data with Oversampling ---
    print("🔎 Finding all preprocessed image patches...")
    # Use preprocessed data directory
    DATA_DIR = Path("D:/ESS_AI_Project/ESS_Final/Dataset/preprocessed_ess")
    filepaths = list(DATA_DIR.glob("**/*.jpg")) + list(DATA_DIR.glob("**/*.png")) + list(DATA_DIR.glob("**/*.tif"))
    label_map = {"HG": 1, "LG": 0}
    data = [{"path": path, "label": label_map[path.parent.name]} for path in filepaths if path.parent.name in label_map]
    df = pd.DataFrame(data)
    print(f"📁 Using preprocessed data from: {DATA_DIR}")

    # Oversample HG class
    print("⚖️ Balancing dataset with oversampling...")
    hg_df = df[df['label'] == 1]  # HG samples
    lg_df = df[df['label'] == 0]  # LG samples
    hg_count = len(hg_df)
    lg_count = len(lg_df)
    if hg_count < lg_count:
        oversample_count = lg_count - hg_count
        hg_oversampled = hg_df.sample(n=oversample_count, replace=True, random_state=RANDOM_STATE)
        balanced_df = pd.concat([df, hg_oversampled], ignore_index=True)
    else:
        balanced_df = df  # No oversampling needed if HG >= LG

    print(f"Found {len(df)} total patches. After oversampling: {len(balanced_df)} patches.")

    # --- 2. Define Data Augmentation and Datasets ---
    train_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(20),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 5-Fold Cross-Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_accuracies = []
    all_preds, all_labels = [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(balanced_df), 1):
        print(f"\n🔄 Starting Fold {fold}/5")
        train_df = balanced_df.iloc[train_idx].reset_index(drop=True)
        test_df = balanced_df.iloc[test_idx].reset_index(drop=True)

        train_dataset = PatchDataset(train_df, transform=train_transform)
        test_dataset = PatchDataset(test_df, transform=test_transform)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # --- 3. Initialize Model and Train ---
        model = create_efficientnet_model().to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        print(f"🏋️ Training Patch Classifier Model for Fold {fold}...")
        model = train_model(train_loader, test_loader, model, criterion, optimizer, EPOCHS, PATIENCE, DEVICE)

        # --- 4. Evaluate the Model ---
        print(f"📊 Evaluating model on test set for Fold {fold}...")
        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    # Aggregate and Report
    report = classification_report(all_labels, all_preds, target_names=["LG (Class 0)", "HG (Class 1)"])
    print("\n📊 5-Fold Cross-Validation Results:")
    print(report)
    
    # Save the classification report
    with open(REPORT_SAVE_PATH, 'w') as f:
        f.write(report)
    print(f"✅ Classification report saved to {REPORT_SAVE_PATH}")

    # Save model from last fold (for simplicity)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ Model saved to {MODEL_SAVE_PATH} (last fold)")

if __name__ == "__main__":
    main()