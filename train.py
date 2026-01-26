import torch
import torch.nn.functional as F
import os
import time
import yaml
import argparse
from tqdm import tqdm

from model import Autoencoder
from data import get_device, get_dataloader, get_test_images, denormalize
from visualize import visualize_reconstructions, plot_losses


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_epoch(model, dataloader, optimizer, device, kl_weight=0.001, epoch=None, total_epochs=None):
    model.train()
    total_loss = 0
    num_samples = 0
    start_time = time.time()
    
    desc = f"Epoch {epoch}/{total_epochs}" if epoch else "Training"
    pbar = tqdm(dataloader, desc=desc, leave=False)
    
    for x, _ in pbar:
        x = x.to(device)
        optimizer.zero_grad()
        x_recon, z, kl_loss = model(x)
        recon_loss = F.mse_loss(x_recon, x)
        loss = recon_loss + kl_weight * kl_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_samples += x.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    elapsed = time.time() - start_time
    return total_loss / len(dataloader), elapsed, num_samples


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    
    for x, _ in dataloader:
        x = x.to(device)
        x_recon, z, kl_loss = model(x)
        loss = F.mse_loss(x_recon, x)
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }, path)


def train(config):
    device = get_device()
    
    name = config['name']
    dataset_cfg = config['dataset']
    model_cfg = config['model']
    train_cfg = config['training']
    
    print(f"Experiment: {name}")
    print(f"Using device: {device}")
    print(f"Dataset: {dataset_cfg['name']} ({dataset_cfg['img_size']}x{dataset_cfg['img_size']}x{dataset_cfg['channels']})")
    print(f"Encoder: {model_cfg['encoder']}, Decoder: {model_cfg['decoder']}, Bottleneck: {model_cfg['bottleneck']}")
    
    run_dir = os.path.join('checkpoints', name)
    output_dir = os.path.join('outputs', name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    model = Autoencoder(
        encoder_name=model_cfg['encoder'],
        decoder_name=model_cfg['decoder'],
        bottleneck_name=model_cfg['bottleneck'],
        latent_dim=model_cfg.get('latent_dim', 256),
        encoder=model_cfg.get('encoder_kwargs', {}),
        decoder=model_cfg.get('decoder_kwargs', {}),
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg['lr'])
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    train_dir = dataset_cfg.get('train_dir')
    val_dir = dataset_cfg.get('val_dir')
    
    train_loader = get_dataloader(
        batch_size=train_cfg['batch_size'],
        train=True,
        dataset_name=dataset_cfg['name'],
        data_dir=train_dir,
        img_size=dataset_cfg['img_size']
    )
    val_loader = get_dataloader(
        batch_size=train_cfg['batch_size'],
        train=False,
        dataset_name=dataset_cfg['name'],
        data_dir=val_dir,
        img_size=dataset_cfg['img_size']
    )
    
    print(f"Train samples: {len(train_loader.dataset):,}, batches: {len(train_loader)}")
    print(f"Val samples: {len(val_loader.dataset):,}, batches: {len(val_loader)}")
    print(f"Batch size: {train_cfg['batch_size']}, Epochs: {train_cfg['epochs']}")
    print("-" * 60)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    total_start_time = time.time()
    
    for epoch in range(1, train_cfg['epochs'] + 1):
        train_loss, epoch_time, num_samples = train_epoch(
            model, train_loader, optimizer, device, train_cfg.get('kl_weight', 0.001),
            epoch=epoch, total_epochs=train_cfg['epochs']
        )
        val_loss = validate(model, val_loader, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        samples_per_sec = num_samples / epoch_time
        print(f"Epoch {epoch}/{train_cfg['epochs']}: train={train_loss:.4f}, val={val_loss:.4f}, "
              f"time={epoch_time:.1f}s, {samples_per_sec:.0f} samples/s")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss, 
                          os.path.join(run_dir, 'best_model.pt'))
            print(f"  -> Saved new best model")
        
        if epoch % train_cfg.get('visualize_every', 5) == 0:
            visualize_reconstructions(
                model, device,
                save_path=os.path.join(output_dir, f'recon_epoch_{epoch}.png'),
                dataset_name=dataset_cfg['name'],
                data_dir=val_dir,
                img_size=dataset_cfg['img_size']
            )
            plot_losses(train_losses, val_losses, 
                       save_path=os.path.join(output_dir, 'loss_curve.png'))
    
    save_checkpoint(model, optimizer, train_cfg['epochs'], train_loss, val_loss,
                   os.path.join(run_dir, 'final_model.pt'))
    plot_losses(train_losses, val_losses, 
               save_path=os.path.join(output_dir, 'loss_curve.png'))
    
    total_time = time.time() - total_start_time
    print(f"\nTraining complete! Total time: {total_time/60:.1f} min")
    print(f"Best val_loss: {best_val_loss:.4f}")
    
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True, help='Path to config YAML')
    args = parser.parse_args()
    
    config = load_config(args.config)
    train(config)
