import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np

def evaluate_model(model, test_loader, device='cuda', num_samples=5, output_dir="."):
    model.eval()
    test_mse = 0.0
    test_psnr = 0.0
    test_ssim = 0.0
    test_mae = 0.0
    
    samples = []
    count = 0
    
    with torch.no_grad():
        for incomplete, complete in tqdm(test_loader, desc='Evaluating'):
            incomplete = incomplete.to(device)
            complete = complete.to(device)
            
            output = model(incomplete)
            
            # Calculate metrics
            output_np = output.cpu().numpy()
            complete_np = complete.cpu().numpy()
            incomplete_np = incomplete.cpu().numpy()
            
            batch_size = output.shape[0]
            for i in range(batch_size):
                # MSE (already normalized data)
                mse = ((output_np[i, 1] - complete_np[i, 1]) ** 2).mean()
                test_mse += mse
                
                # MAE
                mae = np.abs(output_np[i, 1] - complete_np[i, 1]).mean()
                test_mae += mae
                
                data_range = complete_np[i, 1].max() - complete_np[i, 1].min()
                # PSNR
                test_psnr += psnr(complete_np[i, 1], output_np[i, 1], data_range=data_range)
                
                # SSIM
                test_ssim += ssim(complete_np[i, 1], output_np[i, 1], data_range=data_range)
                
                # Save samples for visualization
                if count < num_samples:
                    samples.append((incomplete_np[i, 1], output_np[i, 1], complete_np[i, 1]))
                    count += 1
    
    # Average metrics
    num_items = len(test_loader.dataset)
    avg_mse = test_mse / num_items
    avg_psnr = test_psnr / num_items
    avg_ssim = test_ssim / num_items
    avg_mae = test_mae / num_items
    
    print(f"Test Results: MSE: {avg_mse:.6f}, PSNR: {avg_psnr:.4f} dB, SSIM: {avg_ssim:.4f}, MAE: {avg_mae:.6f}")
    
    # Visualize samples
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 3*num_samples))
    
    for i, (incomplete, output, complete) in enumerate(samples):
        axes[i, 0].imshow(incomplete, cmap='viridis')
        axes[i, 0].set_title('Incomplete Sinogram')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(output, cmap='viridis')
        axes[i, 1].set_title('Restored Sinogram')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(complete, cmap='viridis')
        axes[i, 2].set_title('Ground Truth')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sinogram_results.pdf', dpi=600)
    plt.close()
    
    # Save metrics to a text file
    with open(f'{output_dir}/evaluation_metrics.txt', 'w') as f:
        f.write(f"MSE: {avg_mse:.6f}\n")
        f.write(f"PSNR: {avg_psnr:.4f} dB\n")
        f.write(f"SSIM: {avg_ssim:.4f}\n")
        f.write(f"MAE: {avg_mae:.6f}\n")
    
    return avg_mse, avg_psnr, avg_ssim