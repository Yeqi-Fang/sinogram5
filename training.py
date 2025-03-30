import torch.optim as optim
import time
from tqdm import tqdm
import torch.nn as nn
import torch
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import os
import random
import numpy as np
import math
import csv
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


class DynamicWeightedLoss(nn.Module):
    def __init__(self, initial_alpha=0.1, max_alpha=0.8, epochs=100, schedule='linear'):
        """
        参数:
            initial_alpha: MAE的初始权重
            max_alpha: MAE的最大权重
            epochs: 总训练周期数
            schedule: 权重增长方式 ('linear', 'exp', 'step')
        """
        super().__init__()
        self.initial_alpha = initial_alpha
        self.max_alpha = max_alpha
        self.epochs = epochs
        self.schedule = schedule
        self.mae_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, pred, target):
        mae = self.mae_loss(pred, target)
        mse = self.mse_loss(pred, target)
        return self.current_alpha * mae + (1 - self.current_alpha) * mse
    
    def update_alpha(self, epoch):
        """根据当前周期更新alpha权重"""
        if self.schedule == 'linear':
            # 线性增长
            self.current_alpha = self.initial_alpha + (self.max_alpha - self.initial_alpha) * (epoch / self.epochs)
        elif self.schedule == 'exp':
            # 指数增长
            self.current_alpha = self.initial_alpha + (self.max_alpha - self.initial_alpha) * (1 - math.exp(-5 * epoch / self.epochs))
        elif self.schedule == 'step':
            # 阶梯增长
            milestone = self.epochs // 3
            if epoch < milestone:
                self.current_alpha = self.initial_alpha
            elif epoch < 2 * milestone:
                self.current_alpha = (self.initial_alpha + self.max_alpha) / 2
            else:
                self.current_alpha = self.max_alpha
        
        # 确保alpha在有效范围内
        self.current_alpha = max(self.initial_alpha, min(self.max_alpha, self.current_alpha))
        return self.current_alpha


def calculate_metrics(output_batch, target_batch):
    """
    Calculate SSIM, PSNR, and MAE metrics between output and target batches
    
    Args:
        output_batch: Model output batch tensor (already on CPU and converted to numpy)
        target_batch: Target batch tensor (already on CPU and converted to numpy)
    
    Returns:
        Dictionary containing average SSIM, PSNR, and MAE values
    """
    batch_size = output_batch.shape[0]
    ssim_val = 0.0
    psnr_val = 0.0
    mae_val = 0.0
    
    for i in range(batch_size):
        # Extract the central channel (index 1) containing the main prediction
        output_img = output_batch[i, 1]
        target_img = target_batch[i, 1]
        
        # Calculate data range for normalized metrics
        data_range = target_img.max() - target_img.min()
        
        # Calculate SSIM
        ssim_val += ssim(target_img, output_img, data_range=data_range)
        
        # Calculate PSNR
        psnr_val += psnr(target_img, output_img, data_range=data_range)
        
        # Calculate MAE
        mae_val += np.mean(np.abs(output_img - target_img))
    
    # Return the averages
    return {
        'ssim': ssim_val / batch_size,
        'psnr': psnr_val / batch_size,
        'mae': mae_val / batch_size
    }


def log_epoch_metrics(metrics, epoch, log_dir):
    """
    Log epoch metrics to a CSV file
    
    Args:
        metrics: Dictionary containing metric values
        epoch: Current epoch number
        log_dir: Directory to save log files
    """
    csv_path = os.path.join(log_dir, 'epoch_metrics.csv')
    
    # Check if file exists and create header if it doesn't
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, mode='a', newline='') as file:
        fieldnames = ['epoch', 'train_loss', 'val_loss', 'ssim', 'psnr', 'mae', 'learning_rate', 'timestamp']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            'epoch': epoch + 1,
            'train_loss': metrics['train_loss'],
            'val_loss': metrics['val_loss'],
            'ssim': metrics['ssim'],
            'psnr': metrics['psnr'],
            'mae': metrics['mae'],
            'learning_rate': metrics['learning_rate'],
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        })


def train_model(model, train_loader, test_loader, num_epochs=50, start_epoch=0, device='cuda', 
                save_path='model.pth', vis_dir='visualizations', optimizer_state=None, 
                scaler_state=None, best_loss=float('inf'), scheduler_state=None, 
                random_state=None, vis_data=None, lr=1e-4):
    """
    Train the model with support for resuming from checkpoints
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        test_loader: DataLoader for validation data
        num_epochs: Total number of epochs to train
        start_epoch: Epoch to start training from (for resuming)
        device: Device to train on (cuda or cpu)
        save_path: Path to save the best model
        vis_dir: Directory to save visualizations
        optimizer_state: State dict of optimizer (for resuming)
        scaler_state: State dict of gradient scaler (for resuming)
        best_loss: Best validation loss so far (for resuming)
        scheduler_state: State dict of learning rate scheduler (for resuming)
        random_state: Dictionary of random states (for resuming)
        vis_data: Visualization samples from previous run (for resuming)
        
    Returns:
        tuple: (Trained model, Dict with training history and metrics)
    """
    # Create directory for visualizations if it doesn't exist
    os.makedirs(vis_dir, exist_ok=True)
    
    # Create a checkpoint directory within visualizations
    checkpoint_dir = os.path.join(vis_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Restore random states if resuming
    if random_state:
        torch.set_rng_state(random_state['torch'])
        np.random.set_state(random_state['numpy'])
        random.setstate(random_state['python'])
    
    # Move model to device
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    # criterion = DynamicWeightedLoss(initial_alpha=0.1, max_alpha=0.8, epochs=num_epochs)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Load optimizer state if resuming
    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)
        # Move optimizer state to the correct device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler('cuda')
    
    # Load scaler state if resuming
    if scaler_state:
        scaler.load_state_dict(scaler_state)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.3)
    
    # Load scheduler state if resuming
    if scheduler_state:
        scheduler.__dict__.update(scheduler_state)
    
    # Get visualization samples
    if vis_data:
        # Use the same visualization samples as before
        vis_incomplete, vis_complete = vis_data
    else:
        # Get new samples for visualization
        vis_dataloader = torch.utils.data.DataLoader(test_loader.dataset, batch_size=4, shuffle=True)
        vis_batch = next(iter(vis_dataloader))
        vis_incomplete, vis_complete = vis_batch
    
    # Save initial visualizations for reference if starting fresh
    if start_epoch == 0:
        with torch.no_grad():
            model.eval()
            with autocast(device_type='cuda'):
                vis_outputs = model(vis_incomplete.to(device))
            
            save_visualizations(vis_incomplete, vis_outputs, vis_complete, 
                              os.path.join(vis_dir, 'initial_state.png'),
                              title="Initial Model State")
    
    # Store training history and metrics
    history = {
        'train_loss': [],
        'val_loss': [],
        'ssim': [],
        'psnr': [],
        'mae': [],
        'learning_rate': [],
        'epoch_times': []
    }
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for incomplete, complete in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} (Train)'):
            # Move tensors to device
            incomplete = incomplete.to(device)
            complete = complete.to(device)
            
            # Forward pass with mixed precision
            optimizer.zero_grad()
            
            with autocast(device_type='cuda'):
                outputs = model(incomplete)
                loss = criterion(outputs, complete)
            
            # Backward pass and optimize with scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Update statistics
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        # For calculating metrics
        all_ssim = 0.0
        all_psnr = 0.0
        all_mae = 0.0
        
        with torch.no_grad():
            for incomplete, complete in tqdm(test_loader, desc=f'Epoch {epoch+1}/{num_epochs} (Test)'):
                # Move tensors to device
                incomplete = incomplete.to(device)
                complete = complete.to(device)
                
                # Forward pass with mixed precision
                with autocast(device_type='cuda'):
                    outputs = model(incomplete)
                    loss = criterion(outputs[:, 1:2, :, :], complete[:, 1:2, :, :])
                
                # Calculate metrics for this batch
                outputs_np = outputs.cpu().numpy()
                complete_np = complete.cpu().numpy()
                batch_metrics = calculate_metrics(outputs_np, complete_np)
                
                # Update metrics totals
                all_ssim += batch_metrics['ssim'] * len(incomplete)
                all_psnr += batch_metrics['psnr'] * len(incomplete)
                all_mae += batch_metrics['mae'] * len(incomplete)
                
                # Update loss statistics
                val_loss += loss.item()
                val_batches += 1
        
        # Calculate averages
        avg_val_loss = val_loss / val_batches
        avg_ssim = all_ssim / len(test_loader.dataset)
        avg_psnr = all_psnr / len(test_loader.dataset)
        avg_mae = all_mae / len(test_loader.dataset)
        current_lr = optimizer.param_groups[0]["lr"]
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Calculate epoch duration
        epoch_time = time.time() - epoch_start_time
        
        # Store history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['ssim'].append(avg_ssim)
        history['psnr'].append(avg_psnr)
        history['mae'].append(avg_mae)
        history['learning_rate'].append(current_lr)
        history['epoch_times'].append(epoch_time)
        
        # Print progress with additional metrics
        print(f'Epoch {epoch+1}/{num_epochs}: '
              f'Train Loss: {avg_train_loss:.6f}, '
              f'Val Loss: {avg_val_loss:.6f}, '
              f'SSIM: {avg_ssim:.4f}, '
              f'PSNR: {avg_psnr:.4f}dB, '
              f'MAE: {avg_mae:.6f}, '
              f'LR: {current_lr:.6f}, '
              f'Time: {epoch_time:.2f}s')
        
        # Log epoch metrics to CSV
        epoch_metrics = {
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'ssim': avg_ssim,
            'psnr': avg_psnr,
            'mae': avg_mae,
            'learning_rate': current_lr
        }
        log_epoch_metrics(epoch_metrics, epoch, vis_dir)
        
        # Save checkpoint every epoch for safety
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        
        # Capture current random states
        random_states = {
            'torch': torch.get_rng_state(),
            'numpy': np.random.get_state(),
            'python': random.getstate()
        }
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state': scheduler.__dict__,  # Save full scheduler state
            'scaler': scaler.state_dict(),
            'loss': avg_val_loss,
            'best_loss': best_loss,
            'random_state': random_states,
            'vis_data': (vis_incomplete, vis_complete),  # Save visualization data
            'metrics': {
                'ssim': avg_ssim,
                'psnr': avg_psnr,
                'mae': avg_mae
            }
        }, checkpoint_path)
        
        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state': scheduler.__dict__,
                'scaler': scaler.state_dict(),
                'loss': best_loss,
                'random_state': random_states,
                'vis_data': (vis_incomplete, vis_complete),
                'metrics': {
                    'ssim': avg_ssim,
                    'psnr': avg_psnr,
                    'mae': avg_mae
                }
            }, save_path)
            
            # Also save to the vis directory with a clear name
            best_model_path = os.path.join(vis_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state': scheduler.__dict__,
                'scaler': scaler.state_dict(),
                'loss': best_loss,
                'random_state': random_states,
                'vis_data': (vis_incomplete, vis_complete),
                'metrics': {
                    'ssim': avg_ssim,
                    'psnr': avg_psnr,
                    'mae': avg_mae
                }
            }, best_model_path)
            
            print(f'Model saved at epoch {epoch+1} with validation loss {best_loss:.6f}')
        
        # Generate and save visualizations for this epoch
        with torch.no_grad():
            model.eval()
            with autocast(device_type='cuda'):
                vis_outputs = model(vis_incomplete.to(device))
            
            vis_path = os.path.join(vis_dir, f'epoch_{epoch+1:03d}.png')
            save_visualizations(vis_incomplete, vis_outputs, vis_complete, vis_path,
                              title=f'Epoch {epoch+1} - Val Loss: {avg_val_loss:.6f}, SSIM: {avg_ssim:.4f}, PSNR: {avg_psnr:.4f}dB')
            print(f"Visualization saved to {vis_path}")
    
    # Load best model
    checkpoint = torch.load(save_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Return model and history
    return model, history

def save_visualizations(incomplete, outputs, complete, filepath, title="Visualization"):
    """Helper function to save visualizations"""
    # Create a figure to visualize results
    fig, axes = plt.subplots(min(4, len(incomplete)), 3, figsize=(16, 15), gridspec_kw={"width_ratios": [1, 1, 1.0675]})
    
    # Handle the case where there's only one sample
    if min(4, len(incomplete)) == 1:
        axes = axes.reshape(1, 3)
    
    for i in range(min(4, len(incomplete))):
        # Get the images
        input_img = incomplete[i, 1].cpu().numpy()
        output_img = outputs[i, 1].cpu().numpy()
        target_img = complete[i, 1].cpu().numpy()
        
        # Determine global min and max for consistent colormap scaling
        vmin = min(input_img.min(), output_img.min(), target_img.min())
        vmax = max(input_img.max(), output_img.max(), target_img.max())
        
        # Plot input
        im = axes[i, 0].imshow(input_img, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[i, 0].set_title('Input (Incomplete)')
        axes[i, 0].axis('off')
        
        # Plot output
        im = axes[i, 1].imshow(output_img, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[i, 1].set_title('Output (Predicted)')
        axes[i, 1].axis('off')
        
        # Plot ground truth
        im = axes[i, 2].imshow(target_img, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[i, 2].set_title('Ground Truth (Complete)')
        axes[i, 2].axis('off')
        
        # Add a colorbar to the last image in each row
        plt.colorbar(im, ax=axes[i, 2], fraction=0.04, pad=0.02)
    
    # Add title information
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the figure
    plt.savefig(filepath, dpi=600)
    plt.close(fig)