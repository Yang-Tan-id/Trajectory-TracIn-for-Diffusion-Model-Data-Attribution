# CelebA local data

Prepared data is written to `10000/` by
`../../celeba/script/prepare_celeba.py`. Raw CelebA archives are not committed.

The prepared contract contains:

- `dataset.npz`: 10,000 RGB uint8 images at 64x64 plus source filenames;
- `celeba_10k_source_filenames.npy`: exact seed-42 selection from split 0;
- `attribution_5k_indices.npy`: exact seed-42 attribution universe;
- `attribution_5k_mask.npy`: length-10,000 boolean version of that universe;
- `metadata.json`: source and preprocessing provenance.
