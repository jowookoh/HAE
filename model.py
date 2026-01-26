import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# =============================================================================
# Bottlenecks
# =============================================================================

class HypersphereBottleneck(nn.Module):
    def forward(self, z):
        return F.normalize(z, p=2, dim=-1), torch.tensor(0.0, device=z.device)


class VAEBottleneck(nn.Module):
    def forward(self, z):
        mu, log_var = z.chunk(2, dim=-1)
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z_sampled = mu + std * eps
        else:
            z_sampled = mu
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return z_sampled, kl_loss


# =============================================================================
# CIFAR Encoders/Decoders (32x32x3)
# =============================================================================

class CIFAREncoder(nn.Module):
    def __init__(self, latent_dim=256, bottleneck_type='hypersphere'):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)   # 32 -> 16
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # 16 -> 8
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1) # 8 -> 4
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1) # 4 -> 2
        
        out_dim = latent_dim * 2 if bottleneck_type == 'vae' else latent_dim
        self.proj = nn.Linear(512 * 2 * 2, out_dim)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.flatten(1)
        return self.proj(x)


class CIFARCNNDecoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.proj = nn.Linear(latent_dim, 512 * 2 * 2)
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)
    
    def forward(self, z):
        x = self.proj(z).view(-1, 512, 2, 2)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        return self.deconv4(x)


class CIFARTransformerDecoder(nn.Module):
    def __init__(self, latent_dim=256, embed_dim=256, depth=4, num_heads=8, patch_size=4):
        super().__init__()
        self.img_size = 32
        self.patch_size = patch_size
        self.num_patches = (32 // patch_size) ** 2
        
        self.latent_proj = nn.Linear(latent_dim, embed_dim)
        self.patch_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=depth)
        self.head = nn.Linear(embed_dim, patch_size * patch_size * 3)
    
    def forward(self, z):
        batch_size = z.shape[0]
        memory = self.latent_proj(z).unsqueeze(1)
        tgt = (self.patch_embed + self.pos_embed).expand(batch_size, -1, -1)
        decoded = self.transformer(tgt, memory)
        patches = self.head(decoded)
        h = w = self.img_size // self.patch_size
        return rearrange(patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                        h=h, w=w, p1=self.patch_size, p2=self.patch_size, c=3)


# =============================================================================
# QuickDraw Encoders/Decoders (256x256x1)
# =============================================================================

class QuickDrawEncoder(nn.Module):
    def __init__(self, latent_dim=256, bottleneck_type='hypersphere'):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)    # 256 -> 128
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)   # 128 -> 64
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # 64 -> 32
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1) # 32 -> 16
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1) # 16 -> 8
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1) # 8 -> 4
        
        out_dim = latent_dim * 2 if bottleneck_type == 'vae' else latent_dim
        self.proj = nn.Linear(512 * 4 * 4, out_dim)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = x.flatten(1)
        return self.proj(x)


class QuickDrawCNNDecoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.proj = nn.Linear(latent_dim, 512 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)  # 4 -> 8
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)  # 8 -> 16
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # 16 -> 32
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)   # 32 -> 64
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)    # 64 -> 128
        self.deconv6 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)     # 128 -> 256
    
    def forward(self, z):
        x = self.proj(z).view(-1, 512, 4, 4)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))
        x = F.relu(self.deconv5(x))
        return self.deconv6(x)


class QuickDrawTransformerDecoder(nn.Module):
    def __init__(self, latent_dim=256, embed_dim=384, depth=6, num_heads=8, patch_size=16):
        super().__init__()
        self.img_size = 256
        self.patch_size = patch_size
        self.num_patches = (256 // patch_size) ** 2  # 256 patches for patch_size=16
        
        self.latent_proj = nn.Linear(latent_dim, embed_dim)
        self.patch_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=depth)
        self.head = nn.Linear(embed_dim, patch_size * patch_size * 1)
    
    def forward(self, z):
        batch_size = z.shape[0]
        memory = self.latent_proj(z).unsqueeze(1)
        tgt = (self.patch_embed + self.pos_embed).expand(batch_size, -1, -1)
        decoded = self.transformer(tgt, memory)
        patches = self.head(decoded)
        h = w = self.img_size // self.patch_size
        return rearrange(patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                        h=h, w=w, p1=self.patch_size, p2=self.patch_size, c=1)


# =============================================================================
# Registry
# =============================================================================

ENCODERS = {
    'cifar': CIFAREncoder,
    'quickdraw': QuickDrawEncoder,
}

DECODERS = {
    'cifar_cnn': CIFARCNNDecoder,
    'cifar_transformer': CIFARTransformerDecoder,
    'quickdraw_cnn': QuickDrawCNNDecoder,
    'quickdraw_transformer': QuickDrawTransformerDecoder,
}

BOTTLENECKS = {
    'hypersphere': HypersphereBottleneck,
    'vae': VAEBottleneck,
}


# =============================================================================
# Main Autoencoder (assembled from registry)
# =============================================================================

class Autoencoder(nn.Module):
    def __init__(self, encoder_name, decoder_name, bottleneck_name, latent_dim=256, **kwargs):
        super().__init__()
        
        encoder_cls = ENCODERS[encoder_name]
        decoder_cls = DECODERS[decoder_name]
        bottleneck_cls = BOTTLENECKS[bottleneck_name]
        
        self.encoder = encoder_cls(latent_dim=latent_dim, bottleneck_type=bottleneck_name, **kwargs.get('encoder', {}))
        self.bottleneck = bottleneck_cls()
        self.decoder = decoder_cls(latent_dim=latent_dim, **kwargs.get('decoder', {}))
    
    def encode(self, x):
        h = self.encoder(x)
        z, kl_loss = self.bottleneck(h)
        return z, kl_loss
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z, kl_loss = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z, kl_loss
