# HAE Project Tasks

## Phase 1: Solidify Image Autoencoder

- [x] 1.1 Basic CNN encoder
- [x] 1.2 Hypersphere bottleneck (L2 normalization)
- [x] 1.3 Transformer decoder
- [x] 1.4 CNN decoder alternative
- [x] 1.5 VAE bottleneck alternative
- [x] 1.6 Training loop with visualization
- [x] 1.7 Loss curve plotting
- [x] 1.8 Config-driven experiments (YAML)
- [x] 1.9 CIFAR-10 dataset support
- [x] 1.10 QuickDraw dataset support
- [ ] 1.11 Run all 4 CIFAR experiments to completion
- [ ] 1.12 Run all 4 QuickDraw experiments to completion
- [ ] 1.13 Compare final losses across all 8 experiments

## Phase 2: Interpolation & Manifold Quality

- [ ] 2.1 Implement SLERP function (spherical interpolation)
- [ ] 2.2 Implement LERP function (linear interpolation for VAE comparison)
- [ ] 2.3 Visualize interpolation between 2 images (1D strip)
- [ ] 2.4 Visualize 2D grid interpolation (4 corners)
- [ ] 2.5 Compare SLERP vs LERP smoothness
- [ ] 2.6 Test interpolation on many random pairs
- [ ] 2.7 Add uniformity loss to prevent clustering on sphere
- [ ] 2.8 Evaluate latent space coverage (are points spread evenly?)

## Phase 3: Improve Reconstruction Quality

- [ ] 3.1 Increase latent_dim (256 → 512 → 1024)
- [ ] 3.2 Try L1 loss instead of MSE
- [ ] 3.3 Try L1 + MSE combination
- [ ] 3.4 Try perceptual loss (VGG features)
- [ ] 3.5 Add learning rate scheduler (with proper tuning)
- [ ] 3.6 Try AdamW with lower weight decay
- [ ] 3.7 Experiment with larger decoder (more layers/heads)
- [ ] 3.8 Reduce QuickDraw to 128x128 for faster iteration

## Phase 4: Better Datasets

- [ ] 4.1 Add CelebA faces dataset (good for face morphing demos)
- [ ] 4.2 Preprocess CelebA to 128x128 aligned faces
- [ ] 4.3 Train on CelebA with best config from Phase 1-3
- [ ] 4.4 Visualize face interpolations (more visually compelling)
- [ ] 4.5 Prepare video dataset loader (frame pairs)

## Phase 5: Video - Basic Temporal Model

- [ ] 5.1 Create video frame pair dataset loader
- [ ] 5.2 Design video encoder (takes frame T and T+1)
- [ ] 5.3 Design masked prediction task (mask 75% of T+1)
- [ ] 5.4 Implement video autoencoder
- [ ] 5.5 Train on small video clips (64x64, 2 frames)
- [ ] 5.6 Verify temporal consistency in latent space
- [ ] 5.7 Test: do nearby frames map to nearby points on sphere?

## Phase 6: Video - Interpolation & Evaluation

- [ ] 6.1 SLERP between video latents
- [ ] 6.2 Does interpolation produce valid intermediate motion?
- [ ] 6.3 Extend to longer clips (4-8 frames)
- [ ] 6.4 Increase resolution (128x128)
- [ ] 6.5 Evaluate reconstruction quality on video

## Phase 7: Stage 2 - Text Conditioning (Research)

- [ ] 7.1 Study CLIP architecture and embeddings
- [ ] 7.2 Implement CLIP text encoder integration
- [ ] 7.3 Design Stage 2 architecture (latent mapper)
- [ ] 7.4 Decide: direct prediction vs diffusion vs flow matching
- [ ] 7.5 Implement chosen approach

## Phase 8: Stage 2 - Training & Evaluation

- [ ] 8.1 Create image-text pair dataset (or use BLIP for auto-captioning)
- [ ] 8.2 Train Stage 2 with frozen Stage 1
- [ ] 8.3 Test text-to-edit on images
- [ ] 8.4 Evaluate: does text move latent in meaningful direction?

## Phase 9: Stage 2 - Video Editing (Final Goal)

- [ ] 9.1 Extend Stage 2 to video latents
- [ ] 9.2 Text-conditioned video editing
- [ ] 9.3 Evaluate temporal consistency of edits
- [ ] 9.4 Demo: "make person walk faster", "change weather", etc.

---

## Current Status

**Currently running:** QuickDraw experiments (1.12)
**Next up:** Phase 2 - Interpolation visualization (2.1-2.4)

## Quick Wins (Can Do Anytime)

- [ ] Q.1 Reduce QuickDraw to 128x128 (faster training)
- [ ] Q.2 Add tqdm to validation loop
- [ ] Q.3 Save config.yaml copy to output folder
- [ ] Q.4 Add git commit hash to saved checkpoints
