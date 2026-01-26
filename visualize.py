import torch
import matplotlib.pyplot as plt

from model import Autoencoder
from data import get_device, get_test_images, denormalize


def plot_losses(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150)
    plt.close()


def load_model(checkpoint_path, device, encoder_name, decoder_name, bottleneck_name, 
               latent_dim=256, **kwargs):
    model = Autoencoder(
        encoder_name=encoder_name,
        decoder_name=decoder_name,
        bottleneck_name=bottleneck_name,
        latent_dim=latent_dim,
        **kwargs
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


@torch.no_grad()
def visualize_reconstructions(model, device, n=8, save_path='reconstructions.png',
                               dataset_name='cifar10', data_dir=None, img_size=None):
    images = get_test_images(n, dataset_name=dataset_name, data_dir=data_dir, img_size=img_size).to(device)
    recons, z, _ = model(images)
    
    images = denormalize(images).cpu()
    recons = denormalize(recons).cpu()
    
    is_grayscale = images.shape[1] == 1
    
    fig, axes = plt.subplots(2, n, figsize=(n * 2, 4))
    
    for i in range(n):
        if is_grayscale:
            axes[0, i].imshow(images[i].squeeze(0), cmap='gray')
            axes[1, i].imshow(recons[i].squeeze(0), cmap='gray')
        else:
            axes[0, i].imshow(images[i].permute(1, 2, 0))
            axes[1, i].imshow(recons[i].permute(1, 2, 0))
        
        axes[0, i].axis('off')
        axes[1, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original')
            axes[1, i].set_title('Reconstructed')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    import yaml
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    device = get_device()
    checkpoint_path = args.checkpoint or f"checkpoints/{config['name']}/best_model.pt"
    
    model = load_model(
        checkpoint_path, device,
        encoder_name=config['model']['encoder'],
        decoder_name=config['model']['decoder'],
        bottleneck_name=config['model']['bottleneck'],
        latent_dim=config['model'].get('latent_dim', 256),
    )
    
    visualize_reconstructions(
        model, device,
        dataset_name=config['dataset']['name'],
        data_dir=config['dataset'].get('val_dir'),
        img_size=config['dataset']['img_size']
    )
