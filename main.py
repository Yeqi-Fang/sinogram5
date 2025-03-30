import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
import shutil
from pathlib import Path
import time
import csv

# Import our modules
from dataset import SinogramDatasetCV, create_cv_dataloaders
from model import UNet, LighterUNet
from training import train_model
from evaluation import evaluate_model

def get_timestamp():
    """Create a timestamp string for directory naming"""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def convert_incomplete_to_predicted(model, val_dataset, output_dir, device):
    """
    Convert all validation set incomplete sinograms to predicted complete sinograms
    
    Args:
        model: Trained model
        val_dataset: Validation dataset
        output_dir: Directory to save predicted sinograms
        device: Device to run inference on
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Process all validation samples
    with torch.no_grad(), torch.cuda.amp.autocast():
        for i in tqdm(range(len(val_dataset)), desc="Generating predictions"):
            # Get sample
            incomplete_3ch, _, (folder, data_i, data_j) = val_dataset[i]
            
            # Add batch dimension and move to device
            incomplete_batch = incomplete_3ch.unsqueeze(0).to(device)
            
            # Get prediction
            output = model(incomplete_batch)
            
            # Extract the predicted complete image (middle channel)
            predicted_complete = output[0, 1].cpu().numpy()
            
            # Save the predicted image
            output_path = os.path.join(output_dir, f"incomplete_{data_i}_{data_j}.npy")
            np.save(output_path, predicted_complete)
            
            # Optional: Clear cache periodically to avoid memory issues
            if i % 100 == 0:
                torch.cuda.empty_cache()

def count_model_parameters(model):
    """Count the total number of parameters in a model"""
    return sum(p.numel() for p in model.parameters())

def log_fold_results(args, fold, metrics, model_params, training_time, log_file='cv_results_log.csv'):
    """
    Log the results of a fold to a CSV file
    
    Args:
        args: Command line arguments
        fold: Current fold number
        metrics: Dictionary containing evaluation metrics
        model_params: Dictionary containing model parameters
        training_time: Total training time in seconds
        log_file: Path to the CSV log file
    """
    # Check if the file exists to determine if we need to write headers
    file_exists = os.path.isfile(log_file)
    
    # Prepare the row data
    row_data = {
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'run_id': get_timestamp(),
        'fold': fold,
        'data_dir': args.data_dir,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'attention': bool(args.attention),
        'light_model': args.light,
        'mse': metrics['mse'],
        'psnr': metrics['psnr'],
        'ssim': metrics['ssim'],
        'total_parameters': model_params['total_params'],
        'training_time_hours': training_time / 3600
    }
    
    # Open the file in append mode
    with open(log_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=row_data.keys())
        
        # Write header if file doesn't exist
        if not file_exists:
            writer.writeheader()
        
        # Write the data row
        writer.writerow(row_data)
    
    print(f"Results logged to {log_file}")

def main():
    parser = argparse.ArgumentParser(description='Sinogram Restoration with 6-fold Cross-Validation')
    parser.add_argument('--data_dir', type=str, required=True, help='Base data directory containing train and test folders')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs per fold')
    parser.add_argument('--models_dir', type=str, default='cv_models', help='Directory to save model checkpoints')
    parser.add_argument('--log_dir', type=str, default='cv_logs', help='Base directory for logs and visualizations')
    parser.add_argument('--predictions_dir', type=str, default='predictions', help='Directory to save predicted sinograms')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--attention', type=bool, default=True, help='Use attention in the model')
    parser.add_argument('--pretrain', type=bool, default=False, help='Use pretrained model')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--light', type=int, default=1, help='Use lighter model (0=no, 1=medium, 2=very light)')
    parser.add_argument('--num_folds', type=int, default=6, help='Number of folds for cross-validation')
    parser.add_argument('--start_fold', type=int, default=0, help='Starting fold (0-indexed)')
    parser.add_argument('--end_fold', type=int, default=5, help='Ending fold (0-indexed, inclusive)')
    parser.add_argument('--results_log', type=str, default='cv_results.csv', help='CSV file to log all fold results')
    args = parser.parse_args()
    
    # Check for valid fold range
    if args.start_fold < 0 or args.end_fold >= args.num_folds or args.start_fold > args.end_fold:
        raise ValueError(f"Invalid fold range: start_fold={args.start_fold}, end_fold={args.end_fold}, num_folds={args.num_folds}")
    
    # Set device
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    # Create base directories
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Run cross-validation for specified folds
    for fold_idx in range(args.start_fold, args.end_fold + 1):
        print(f"\n{'-' * 80}")
        print(f"Starting Fold {fold_idx + 1}/{args.num_folds}")
        print(f"{'-' * 80}\n")
        
        # Record start time for this fold
        fold_start_time = time.time()
        
        # Create dataloaders for this fold
        train_loader, val_loader = create_cv_dataloaders(
            args.data_dir, 
            fold_idx=fold_idx, 
            num_folds=args.num_folds,
            batch_size=args.batch_size,
            num_workers=4
        )
        
        # Create model
        if not args.light:
            model = UNet(n_channels=3, n_classes=3, bilinear=False, attention=args.attention, pretrain=args.pretrain)
        else:
            model = LighterUNet(n_channels=3, n_classes=3, bilinear=False, attention=args.attention, pretrain=args.pretrain, light=args.light)
        
        # Count model parameters
        total_params = count_model_parameters(model)
        print(f"Total model parameters: {total_params:,}")
        
        # Create fold-specific directories
        fold_timestamp = get_timestamp()
        fold_log_dir = os.path.join(args.log_dir, f"fold_{fold_idx+1}_{fold_timestamp}")
        fold_model_path = os.path.join(args.models_dir, f"model_fold_{fold_idx+1}_{fold_timestamp}.pth")
        fold_predictions_dir = os.path.join(args.predictions_dir, f"fold_{fold_idx+1}_{fold_timestamp}")
        
        os.makedirs(fold_log_dir, exist_ok=True)
        os.makedirs(fold_predictions_dir, exist_ok=True)
        
        # Save fold parameters
        with open(os.path.join(fold_log_dir, 'fold_params.txt'), 'w') as f:
            f.write(f"Fold: {fold_idx+1}/{args.num_folds}\n")
            for arg, value in vars(args).items():
                f.write(f"{arg}: {value}\n")
        
        # Train model for this fold
        model, history = train_model(
            model, 
            train_loader, 
            val_loader, 
            num_epochs=args.num_epochs,
            device=device, 
            save_path=fold_model_path,
            vis_dir=fold_log_dir,
            lr=args.lr
        )
        
        # Evaluate model and save results
        mse, psnr_val, ssim_val = evaluate_model(model, val_loader, device, output_dir=fold_log_dir)
        
        # Save final model with metrics
        final_model_path = os.path.join(fold_log_dir, 'final_model.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'fold': fold_idx,
            'metrics': {
                'mse': mse,
                'psnr': psnr_val,
                'ssim': ssim_val
            }
        }, final_model_path)
        print(f"Final model for fold {fold_idx+1} saved at {final_model_path}")
        
        # Generate predictions for validation set
        print("Generating predictions for validation set...")
        convert_incomplete_to_predicted(
            model, 
            val_loader.dataset, 
            fold_predictions_dir, 
            device
        )
        print(f"Predictions saved to {fold_predictions_dir}")
        
        # Calculate total training time for this fold
        fold_training_time = time.time() - fold_start_time
        
        # Log results for this fold
        metrics = {
            'mse': mse,
            'psnr': psnr_val,
            'ssim': ssim_val
        }
        
        model_params = {
            'total_params': total_params
        }
        
        log_fold_results(args, fold_idx+1, metrics, model_params, fold_training_time, log_file=args.results_log)
        
        # Print summary for this fold
        print(f"\nFold {fold_idx+1} Summary:")
        print(f"Training time: {fold_training_time/3600:.2f} hours ({fold_training_time:.2f} seconds)")
        print(f"MSE: {mse:.6f}")
        print(f"PSNR: {psnr_val:.4f} dB")
        print(f"SSIM: {ssim_val:.4f}")
        
        # Clean up to free memory
        del model, train_loader, val_loader
        torch.cuda.empty_cache()
    
    print("\nCross-validation complete!")

if __name__ == '__main__':
    main()