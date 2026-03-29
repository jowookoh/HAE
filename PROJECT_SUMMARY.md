# Hyperspherical Autoencoder (HAE) Project Summary

## Project Goal
Build a two-stage system for image editing:
- **Stage 1**: Learn a well-structured latent manifold (hypersphere) that enables smooth interpolation
- **Stage 2** (future): Text-conditioned navigation on the manifold using CLIP

## Key Theoretical Concepts

### Why Hypersphere?
- Standard VAE: Gaussian latent space, uses LERP (linear interpolation)
- Hyperspherical: All points have norm=1, uses SLERP (spherical interpolation)
- SLERP travels along geodesic (surface), avoiding "empty" interior regions
- Results in semantically meaningful intermediate points during interpolation

### SLERP vs LERP
- **LERP** (Linear): `z = (1-t)*z1 + t*z2` — cuts through sphere interior
- **SLERP** (Spherical): Follows arc on sphere surface — stays on manifold

### Architecture Discussion
- **CNN Encoder**: Good for extracting spatial features efficiently
- **Transformer Decoder**: Global attention helps with coherent reconstruction
- **Hybrid approach**: CNN encodes local patterns → hypersphere bottleneck → Transformer decodes with global context

## What We Built

### Stage 1 Components
1. **Encoders**: CIFAREncoder (32x32x3), QuickDrawEncoder (256x256x1)
2. **Decoders**: CNN and Transformer variants for each dataset
3. **Bottlenecks**: Hypersphere (L2 normalization) and VAE (reparameterization trick)
4. **Registry system**: Easy to mix/match components via YAML configs

### Training Infrastructure
- Config-driven experiments (YAML files)
- Loss curve plotting
- Reconstruction visualization every N epochs
- Timing/throughput metrics
- tqdm progress bars

### Experiments Run
- CIFAR-10: 4 combinations (hypersphere/vae × transformer/cnn)
- QuickDraw: Same 4 combinations at 256x256

## Key Findings

### Reconstruction Quality
- At 50 epochs on CIFAR, loss ~0.035-0.04
- Reconstructions capture shapes and colors but remain blurry
- MSE loss tends to produce averaged/soft outputs

### Performance
- CIFAR 32x32: Fast (~1-2 min/epoch on M3 Pro)
- QuickDraw 256x256: Slow (~10 min/epoch on M3 Pro)
- Batch size 64-512 works on 36GB unified memory

## Ideas Tried (Some Reverted)
- ✅ Visualizations (kept)
- ✅ Loss curve plotting (kept)
- ✅ Timing metrics (kept)
- ❌ ResBlocks with skip connections (reverted - slower convergence)
- ❌ AdamW + CosineAnnealingLR (reverted - needed more tuning)
- ❌ Large batch size without LR scaling (reverted)

## Future Plans

### Short Term
- [ ] Implement SLERP interpolation visualization
- [ ] Implement LERP interpolation for VAE comparison
- [ ] 2D grid interpolation (4 corners)
- [ ] Test multiple image pairs for interpolation quality
- [ ] Add uniformity loss (prevents clustering on hypersphere)

### Improvements to Try
- [ ] Increase latent_dim (512 → 1024)
- [ ] Perceptual loss (VGG features) instead of pure MSE
- [ ] L1 + MSE combination
- [ ] Reduce QuickDraw to 128x128 for faster iteration
- [ ] Learning rate warmup + decay (with proper tuning)

### Stage 2 (Text-Conditioned Editing)
- Freeze Stage 1 encoder/decoder
- Train new transformer that takes:
  - Image → encoded to point on hypersphere
  - CLIP text embedding
- Outputs: New point on hypersphere → decode to edited image
- The manifold becomes the "law" that Stage 2 must obey

**Diffusion approach (discussed):**
- Instead of direct point prediction (tends to produce "averaged" results)
- Use **Latent Diffusion** on the hypersphere:
  1. Take point z on sphere (encoded image)
  2. Add noise (also on sphere)
  3. Train transformer to denoise, conditioned on CLIP text
  4. Iterative denoising "walks" the point toward the text-described target
- Benefits: More diverse outputs, avoids mode collapse, better detail
- Alternative: **Flow Matching** - learns velocity vectors on sphere (cleaner math for spherical geometry)

### Dataset Ideas
- CelebA faces (128x128) - good for face morphing demos
- Larger QuickDraw subsets
- Custom video frames (original goal)

## Architecture Details

### Current CIFAR Model (~6.4M params)
```
CNNEncoder: 3→64→128→256→512 (stride 2 each), proj to latent_dim
Hypersphere: L2 normalize
TransformerDecoder: 4 layers, 8 heads, 64 patches (4x4 patch size)
```

### Current QuickDraw Model
```
CNNEncoder: 1→32→64→128→256→512→512, proj to latent_dim  
Hypersphere: L2 normalize
TransformerDecoder: 6 layers, 8 heads, 256 patches (16x16 patch size)
```

## File Structure
```
HAE/
├── configs/           # YAML experiment configs
├── data.py           # Dataset loading (CIFAR, QuickDraw)
├── model.py          # Encoders, decoders, bottlenecks, registry
├── train.py          # Training loop
├── visualize.py      # Reconstruction & loss visualization
├── run_experiments.py # Batch experiment runner
├── checkpoints/      # Saved models
└── outputs/          # Visualizations
```

## Hardware
- **Development**: MacBook Pro M3 Pro, 36GB unified memory
- **Production training**: RTX 3090 (24GB VRAM) available for larger experiments
