import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
import glob
from pathlib import Path

# Import the dataset class and model
from model import UNet, LighterUNet

def load_model(checkpoint_path, device, model_type='unet', light=1, attention=True):
    """
    Load a trained model from checkpoint
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        model_type: Type of model ('unet' or 'lighterunet')
        light: Level of model lightness (for LighterUNet)
        attention: Whether to use attention in the model
        
    Returns:
        Loaded model
    """
    # Create the model instance based on model type
    if model_type.lower() == 'unet':
        model = UNet(n_channels=3, n_classes=3, bilinear=False, attention=attention, pretrain=False)
    elif model_type.lower() == 'lighterunet':
        model = LighterUNet(n_channels=3, n_classes=3, bilinear=False, attention=attention, pretrain=False, light=light)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Assuming the checkpoint dictionary contains 'model_state_dict'
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def merge_cv_predictions(predictions_dir, output_dir):
    """
    Merge predictions from all cross-validation folds into one directory
    
    Args:
        predictions_dir: Base directory containing fold prediction directories
        output_dir: Directory to save merged predictions
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all fold prediction directories
    fold_dirs = sorted(glob.glob(os.path.join(predictions_dir, "fold_*")))
    
    if not fold_dirs:
        raise ValueError(f"No fold prediction directories found in {predictions_dir}")
    
    print(f"Found {len(fold_dirs)} fold prediction directories")
    
    # Keep track of which files we've already processed
    processed_files = set()
    
    # Process each fold directory
    for fold_dir in fold_dirs:
        print(f"Processing {fold_dir}...")
        
        # Find all prediction files in this fold
        prediction_files = glob.glob(os.path.join(fold_dir, "incomplete_*.npy"))
        
        for file_path in tqdm(prediction_files, desc=f"Processing {os.path.basename(fold_dir)}"):
            file_name = os.path.basename(file_path)
            
            # Skip if we've already processed this file
            if file_name in processed_files:
                continue
            
            # Load prediction
            prediction = np.load(file_path)
            
            # Save to output directory
            output_path = os.path.join(output_dir, file_name)
            np.save(output_path, prediction)
            
            # Mark as processed
            processed_files.add(file_name)
    
    print(f"Merged {len(processed_files)} unique predictions into {output_dir}")

def main():
    parser = argparse.ArgumentParser(
        description="Merge predictions from cross-validation folds"
    )
    parser.add_argument('--predictions_dir', type=str, required=True,
                        help="Base directory containing fold prediction directories")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Directory to save merged predictions")
    args = parser.parse_args()
    
    # Merge predictions
    merge_cv_predictions(args.predictions_dir, args.output_dir)

if __name__ == '__main__':
    main()