import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from sklearn.model_selection import KFold

class SinogramDatasetCV(Dataset):
    def __init__(self, data_dir, fold_idx=0, num_folds=6, is_train=True, transform=None):
        """
        Cross-validation Sinogram Dataset
        
        Args:
            data_dir: Base directory containing train and test folders
            fold_idx: Current fold index (0 to num_folds-1)
            num_folds: Total number of folds (default: 6)
            is_train: If True, load training set for current fold, otherwise load validation set
            transform: Optional transforms to apply
        """
        self.data_dir = data_dir
        self.fold_idx = fold_idx
        self.num_folds = num_folds
        self.is_train = is_train
        self.transform = transform
        
        # Define all possible groups (combine train and test)
        train_groups = range(1, 171)  # 1 to 170 from train folder
        test_groups = range(1, 37)    # 1 to 36 from test folder
        all_groups = list(train_groups) + list(test_groups)
        
        # Create fold assignments
        random.seed(42)  # For reproducibility
        all_groups_shuffled = all_groups.copy()
        random.shuffle(all_groups_shuffled)
        
        # Calculate fold size
        fold_size = len(all_groups_shuffled) // num_folds
        
        # Assign groups to folds
        folds = []
        for i in range(num_folds):
            if i < num_folds - 1:
                fold = all_groups_shuffled[i * fold_size:(i + 1) * fold_size]
            else:
                # Last fold might have extra elements
                fold = all_groups_shuffled[i * fold_size:]
            folds.append(fold)
        
        # Get current fold's train and validation groups
        val_groups = folds[fold_idx]
        train_groups = []
        for i in range(num_folds):
            if i != fold_idx:
                train_groups.extend(folds[i])
        
        # Select groups based on train/validation flag
        self.groups = train_groups if is_train else val_groups
        
        # Map original group indices to folder paths
        self.group_paths = []
        for group in self.groups:
            if group <= 170:
                # From original train folder
                folder = os.path.join(data_dir, 'train')
                group_idx = group
            else:
                # From original test folder
                folder = os.path.join(data_dir, 'test')
                group_idx = group - 170  # Adjust index for test folder
            self.group_paths.append((folder, group_idx))
        
        # J range is the same for all groups
        self.j_range = range(1, 1765)  # 1 to 1764
        
        # Create all possible (group_path, i, j) pairs
        self.pairs = []
        for folder, i in self.group_paths:
            for j in self.j_range:
                self.pairs.append((folder, i, j))
        
        # Preload all data into memory
        print(f"Preloading {'training' if is_train else 'validation'} data for fold {fold_idx+1}/{num_folds} into memory...")
        self.incomplete_data = {}
        self.complete_data = {}
        
        for folder, i, j in tqdm(self.pairs):
            # Define file paths
            incomplete_path = os.path.join(folder, f"incomplete_{i}_{j}.npy")
            complete_path = os.path.join(folder, f"complete_{i}_{j}.npy")
            
            # Load data as float16 to save memory during preloading
            try:
                self.incomplete_data[(folder, i, j)] = np.load(incomplete_path).astype(np.float16)
                self.complete_data[(folder, i, j)] = np.load(complete_path).astype(np.float16)
            except FileNotFoundError as e:
                print(f"Warning: File not found: {e}")
                continue
        
        print(f"Successfully preloaded {len(self.incomplete_data)} pairs of sinograms")
        
        # Update pairs to only include successfully loaded files
        self.pairs = list(self.incomplete_data.keys())
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        folder, i, j = self.pairs[idx]
        # Determine position in group (0-indexed): 0 means first image in a group, 41 means last image in a group
        pos_in_group = (j - 1) % 42

        # --- For the incomplete sinogram ---
        # Load current sinogram and ensure channel dimension exists
        current_incomplete = torch.from_numpy(self.incomplete_data[(folder, i, j)].astype(np.float32))
        if current_incomplete.dim() == 2:
            current_incomplete = current_incomplete.unsqueeze(0)
        
        # Left neighbor: if first in group, use current image itself; otherwise load j-1
        if pos_in_group == 0:
            left_incomplete = current_incomplete.clone()
        else:
            left_key = (folder, i, j - 1)
            if left_key in self.incomplete_data:
                left_incomplete = torch.from_numpy(self.incomplete_data[left_key].astype(np.float32))
                if left_incomplete.dim() == 2:
                    left_incomplete = left_incomplete.unsqueeze(0)
            else:
                left_incomplete = current_incomplete.clone()
        
        # Right neighbor: if last in group, use current image itself; otherwise load j+1
        if pos_in_group == 41:
            right_incomplete = current_incomplete.clone()
        else:
            right_key = (folder, i, j + 1)
            if right_key in self.incomplete_data:
                right_incomplete = torch.from_numpy(self.incomplete_data[right_key].astype(np.float32))
                if right_incomplete.dim() == 2:
                    right_incomplete = right_incomplete.unsqueeze(0)
            else:
                right_incomplete = current_incomplete.clone()
        
        # Stack to create a 3-channel tensor (3, H, W)
        incomplete_3ch = torch.cat([left_incomplete, current_incomplete, right_incomplete], dim=0)
        
        # --- For the complete sinogram ---
        current_complete = torch.from_numpy(self.complete_data[(folder, i, j)].astype(np.float32))
        if current_complete.dim() == 2:
            current_complete = current_complete.unsqueeze(0)
        
        if pos_in_group == 0:
            left_complete = current_complete.clone()
        else:
            left_key = (folder, i, j - 1)
            if left_key in self.complete_data:
                left_complete = torch.from_numpy(self.complete_data[left_key].astype(np.float32))
                if left_complete.dim() == 2:
                    left_complete = left_complete.unsqueeze(0)
            else:
                left_complete = current_complete.clone()
        
        if pos_in_group == 41:
            right_complete = current_complete.clone()
        else:
            right_key = (folder, i, j + 1)
            if right_key in self.complete_data:
                right_complete = torch.from_numpy(self.complete_data[right_key].astype(np.float32))
                if right_complete.dim() == 2:
                    right_complete = right_complete.unsqueeze(0)
            else:
                right_complete = current_complete.clone()
        
        complete_3ch = torch.cat([left_complete, current_complete, right_complete], dim=0)
        
        # Apply transforms if provided
        if self.transform:
            incomplete_3ch = self.transform(incomplete_3ch)
            complete_3ch = self.transform(complete_3ch)
        
        return incomplete_3ch, complete_3ch, (folder, i, j)

# Create cross-validation dataloaders
def create_cv_dataloaders(data_dir, fold_idx=0, num_folds=6, batch_size=8, num_workers=4, transform=None):
    """
    Create dataloaders for cross-validation
    
    Args:
        data_dir: Base directory with train and test folders
        fold_idx: Current fold index (0 to num_folds-1)
        num_folds: Total number of folds
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes
        transform: Optional transforms to apply
        
    Returns:
        train_loader, val_loader: DataLoader objects for training and validation
    """
    # Create datasets
    train_dataset = SinogramDatasetCV(data_dir, fold_idx, num_folds, is_train=True, transform=transform)
    val_dataset = SinogramDatasetCV(data_dir, fold_idx, num_folds, is_train=False, transform=transform)
    
    # Create dataloaders
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