# HAE Project Notes

## Current Architecture (Stage 1)
- CNN Encoder → Bottleneck (Hypersphere or VAE) → Transformer/CNN Decoder
- Dataset: CIFAR-10 (32x32, 50k train, 10k test)
- Latent dim: 256

## Experiments to Run
- [x] Hypersphere + Transformer
- [ ] Hypersphere + CNN
- [ ] VAE + Transformer
- [ ] VAE + CNN

## Ideas to Try (After 4-way comparison)

### Interpolation (Priority!)
- [ ] Implement SLERP (spherical) for hypersphere bottleneck
- [ ] Implement LERP (linear) for VAE bottleneck
- [ ] Visualize interpolation between random image pairs
- [ ] Visualize 2D grid interpolation (4 corners)
- [ ] Compare smoothness across all 4 setups
- [ ] Test many image pairs, not just one

### Regularization
- [ ] Uniformity loss - prevents clustering on hypersphere

### Encoder Improvements
- [ ] Add ResNet-style skip connections (within encoder only)
- [ ] Replace ReLU with SiLU/GELU
- [ ] Add BatchNorm or LayerNorm

### Training Improvements
- [ ] Switch Adam → AdamW
- [ ] Add learning rate scheduler (CosineAnnealingLR or ReduceLROnPlateau)
- [ ] Try warmup + decay schedule

### Loss Function
- [ ] Add perceptual loss (VGG features)
- [ ] Try MSE + L1 combination

### Architecture Scaling
- [ ] Increase latent_dim (256 → 512)
- [ ] More transformer depth/heads
- [ ] Try pretrained ResNet encoder (for larger images)

## Predictions (Reconstruction Loss)
1. Hypersphere + CNN (best)
2. Hypersphere + Transformer
3. VAE + CNN
4. VAE + Transformer (worst)

Note: Interpolation quality might rank differently - VAE could be smoother

## Stage 2 (Future)
- Latent Mapper conditioned on CLIP text
- Freeze Stage 1, train new transformer to navigate the manifold
- Input: image + text → Output: new point on sphere → decode to edited image

## Key Insights from Discussion
- Hypersphere: norm=1, use SLERP for interpolation
- VAE: Gaussian space, use LERP for interpolation
- Encoder-to-decoder skips (U-Net) hurt manifold quality - avoid for interpolation
- Transformer decoder needs more epochs/data to outperform CNN
