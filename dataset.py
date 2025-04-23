import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class COVIDXrayDataset(Dataset):
    def __init__(self, base_dir, txt_file, transform=None, is_train=True):
        """
        Args:
            base_dir (string): Base directory for the data
            txt_file (string): Path to the text file with annotations
            transform (callable, optional): Optional transform to be applied on images
            is_train (bool): Whether this is training data or validation data
        """
        self.base_dir = base_dir
        self.is_train = is_train
        self.transform = transform
        
        # Parse the data file
        self.data = []
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 3:  # Ensure there are at least patient_id, filename, and class
                patient_id, filename, covid_class, *rest = parts
                
                # COVID-19 positive if 'positive', otherwise normal
                label = 1 if covid_class.lower() == 'positive' else 0
                
                img_path = os.path.join(base_dir, 'train' if is_train else 'val', filename)
                if os.path.exists(img_path):
                    self.data.append({
                        'patient_id': patient_id,
                        'filename': filename,
                        'label': label,
                        'path': img_path
                    })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item['path']
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
        
        # Prepare label text for Llama model
        label_text = "COVID-19 positive" if item['label'] == 1 else "Normal patient"
        
        return {
            'image': image,
            'label': torch.tensor(item['label'], dtype=torch.long),
            'patient_id': item['patient_id'],
            'filename': item['filename'],
            'label_text': label_text
        }

def get_data_loaders(base_dir, batch_size=16, num_workers=4):
    # Define paths
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    train_txt = os.path.join(base_dir, 'train.txt')
    val_txt = os.path.join(base_dir, 'val.txt')
    
    # Define transformations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = COVIDXrayDataset(base_dir, train_txt, transform=train_transform, is_train=True)
    val_dataset = COVIDXrayDataset(base_dir, val_txt, transform=val_transform, is_train=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader 