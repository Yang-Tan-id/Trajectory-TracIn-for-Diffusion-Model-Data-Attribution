# ArtBench Data

Expected files:

- `latents/artbench256/train_latents.npz`
- `latents/artbench256/test_latents.npz`
- `hf_artbench/train/` only for the optional legacy Diffusers wrapper
- `indices/` for any ArtBench subset index files

Prompted and unprompted JAX runs share the raw images, autoencoder, and latent
cache. Conditioning is disabled only in the unprompted diffusion model.
