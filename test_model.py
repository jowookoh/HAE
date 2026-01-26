import torch
from model import HypersphericalAutoencoder

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

model = HypersphericalAutoencoder().to(device)

x = torch.randn(4, 3, 32, 32).to(device)  # batch of 4 images

x_recon, z, kl_loss = model(x)

print(f"Input shape: {x.shape}")
print(f"Latent shape: {z.shape}")
print(f"Latent norm (should be ~1.0): {z.norm(dim=-1)}")
print(f"Output shape: {x_recon.shape}")

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
