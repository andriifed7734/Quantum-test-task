import os
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms


class Config:
    SUMMER_DIR = "./prepared_dataset/summer"
    WINTER_DIR = "./prepared_dataset/winter"
    BATCH_SIZE = 16
    EPOCHS = 10
    LEARNING_RATE = 0.0005
    MARGIN = 2.0  # Margin for Contrastive Loss
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SeasonalPatchDataset(Dataset):
    """
    Custom Dataset for loading pairs of Sentinel-2 images (Summer & Winter).
    Generates positive pairs (same location) and negative pairs (different locations).
    """
    def __init__(self, summer_dir, winter_dir, transform=None):
        self.summer_dir = summer_dir
        self.winter_dir = winter_dir
        self.transform = transform
        self.filenames = sorted(os.listdir(summer_dir))
        self.num_samples = len(self.filenames)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 50% chance to create a positive pair, 50% for a negative pair
        is_positive = random.random() > 0.5
        
        img1_name = self.filenames[idx]
        img1_path = os.path.join(self.summer_dir, img1_name)
        
        if is_positive:
            img2_name = img1_name
            label = torch.tensor(0.0, dtype=torch.float32) # 0 means identical
        else:
            # Optimized negative sampling (O(1) instead of O(N))
            idx2 = random.randint(0, self.num_samples - 1)
            while idx2 == idx:
                idx2 = random.randint(0, self.num_samples - 1)
            img2_name = self.filenames[idx2]
            label = torch.tensor(1.0, dtype=torch.float32) # 1 means different

        img2_path = os.path.join(self.winter_dir, img2_name)

        # Read images (Grayscale for structural focus)
        img1 = cv2.imdecode(np.fromfile(img1_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imdecode(np.fromfile(img2_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label


class SiameseNetwork(nn.Module):
    """
    Siamese Network using ResNet18 as the backbone for feature extraction.
    """
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # Load pre-trained ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Modify first layer to accept grayscale images (1 channel instead of 3)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove the classification head (fc layer)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Add custom fully connected layers for feature embedding
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )

    def forward_once(self, x):
        output = self.backbone(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def forward(self, input1, input2):
        # Extract features for both images using the shared backbone
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss function.
    Pulls positive pairs closer, pushes negative pairs apart up to a margin.
    """
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Calculate Euclidean distance
        euclidean_distance = nn.functional.pairwise_distance(output1, output2, keepdim=True)
        
        # Loss calculation
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +
            (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive


def train():
    print(f"Starting training on device: {Config.DEVICE}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256), antialias=True) # Downscale for faster training
    ])

    dataset = SeasonalPatchDataset(Config.SUMMER_DIR, Config.WINTER_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0)

    model = SiameseNetwork().to(Config.DEVICE)
    criterion = ContrastiveLoss(margin=Config.MARGIN)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    model.train()
    loss_history = []

    for epoch in range(Config.EPOCHS):
        epoch_loss = 0.0
        
        for i, data in enumerate(dataloader, 0):
            img1, img2, label = data
            img1, img2, label = img1.to(Config.DEVICE), img2.to(Config.DEVICE), label.to(Config.DEVICE)

            optimizer.zero_grad()
            
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, label)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch+1}/{Config.EPOCHS}] - Average Loss: {avg_loss:.4f}")

    # Save model weights
    torch.save(model.state_dict(), "siamese_model_weights.pth")
    print("Training complete. Model weights saved to 'siamese_model_weights.pth'.")


if __name__ == "__main__":
    # Ensure dataset directories exist before starting
    if os.path.exists(Config.SUMMER_DIR) and os.path.exists(Config.WINTER_DIR):
        train()
    else:
        print("Dataset directories not found. Please run dataset extraction first.")