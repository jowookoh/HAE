import torch
from model import HypersphericalAutoencoder
from data import get_device

device = get_device()
model = HypersphericalAutoencoder().to(device)

batch_sizes = [128, 256, 512, 1024, 2048]

for bs in batch_sizes:
    try:
        x = torch.randn(bs, 3, 32, 32).to(device)
        x_recon, z, kl = model(x)
        loss = torch.nn.functional.mse_loss(x_recon, x)
        loss.backward()
        
        if device.type == 'mps':
            mem = torch.mps.current_allocated_memory() / 1024**3
            print(f"Batch {bs}: OK - {mem:.2f} GB allocated")
        else:
            print(f"Batch {bs}: OK")
        
        del x, x_recon, z, loss
        torch.mps.empty_cache() if device.type == 'mps' else None
        
    except RuntimeError as e:
        print(f"Batch {bs}: FAILED - {e}")
        break
