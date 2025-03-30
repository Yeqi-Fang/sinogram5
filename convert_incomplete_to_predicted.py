import os
import argparse
import numpy as np
import torch
from tqdm import tqdm

# Import the dataset class and model
from dataset import SinogramDataset
from model import UNet, LighterUNet
from torch.utils.data import DataLoader

def load_model(checkpoint_path, device, model_type='unet', attention=True):
    # Create the model instance based on model type
    if model_type.lower() == 'unet':
        model = UNet(n_channels=3, n_classes=3, bilinear=False, attention=attention, pretrain=False)
    elif model_type.lower() == 'lighterunet':
        model = LighterUNet(n_channels=3, n_classes=3, bilinear=False, attention=attention, pretrain=False, light=1)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Assuming the checkpoint dictionary contains 'model_state_dict'
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def convert_incomplete_to_predicted(data_dir, subset, checkpoint_path, output_dir, device, 
                                    batch_size=32, num_workers=4, model_type='unet', attention=True):
    # data_dir should be the base folder containing subfolders (e.g., 'train' and 'test')
    subset_dir = os.path.join(data_dir, subset)
    
    # Instantiate the dataset
    is_train = True if subset.lower() == 'train' else False
    dataset = SinogramDataset(subset_dir, is_train=is_train, transform=None, test=False)
    
    # Create a DataLoader with the specified batch size
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the model
    model = load_model(checkpoint_path, device, model_type, attention)
    
    # Process batches
    print(f"Converting incomplete sinograms to predicted complete sinograms using batch size {batch_size}...")
    
    batch_idx = 0
    with torch.no_grad(), torch.cuda.amp.autocast():  # Use mixed precision for faster inference
        for incomplete_batch, _ in tqdm(data_loader, desc="Processing batches"):
            # Move batch to device
            incomplete_batch = incomplete_batch.to(device, non_blocking=True)
            
            # Get model predictions
            outputs = model(incomplete_batch)
            
            # Process each sample in the batch
            for i in range(outputs.shape[0]):
                # Get the corresponding file index
                idx = batch_idx * batch_size + i
                if idx >= len(dataset.pairs):
                    break  # In case the last batch is not full
                
                # Get file identification
                data_i, data_j = dataset.pairs[idx]
                
                # Extract the predicted complete image (middle channel)
                predicted_complete = outputs[i, 1].cpu().numpy()
                
                # Save the predicted image
                output_path = os.path.join(output_dir, f"incomplete_{data_i}_{data_j}.npy")
                np.save(output_path, predicted_complete)
            
            # Increment batch index
            batch_idx += 1
            
            # Optional: Clear cache periodically to avoid memory issues
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
    
    print(f"Conversion complete. Predicted images are saved in {output_dir}")

def main():
    parser = argparse.ArgumentParser(
        description="Convert all incomplete sinogram images into predicted complete images."
    )
    parser.add_argument('--data_dir', type=str, required=True,
                        help="Path to the base data directory (should contain subfolders like 'train' and 'test').")
    parser.add_argument('--subset', type=str, default='test', choices=['train', 'test'],
                        help="Which subset to process ('train' or 'test').")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help="Path to the trained model checkpoint (should include 'model_state_dict').")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Directory to save the predicted complete images.")
    parser.add_argument('--device', type=str, default='cuda',
                        help="Device to use (e.g., 'cuda' or 'cpu').")
    parser.add_argument('--batch_size', type=int, default=24,
                        help="Batch size for processing.")
    parser.add_argument('--num_workers', type=int, default=4,
                        help="Number of worker processes for data loading.")
    parser.add_argument('--model_type', type=str, default='unet', choices=['unet', 'lighterunet'],
                        help="Type of model to use.")
    parser.add_argument('--attention', type=bool, default=True,
                        help="Whether to use attention in the model.")
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    convert_incomplete_to_predicted(
        args.data_dir, 
        args.subset, 
        args.checkpoint, 
        args.output_dir, 
        device,
        args.batch_size,
        args.num_workers,
        args.model_type,
        args.attention
    )

if __name__ == '__main__':
    main()