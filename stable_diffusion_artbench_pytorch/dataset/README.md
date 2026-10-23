# ArtBench data location

Place the original ArtBench-10 metadata and 256 px ImageFolder dataset here:

```text
dataset/
  metadata/
    ArtBench-10.csv
  images/
    artbench-10-imagefolder/
      post_impressionism/
        <image files referenced by ArtBench-10.csv>
      ukiyo_e/
        <image files referenced by ArtBench-10.csv>
      ...other ArtBench classes are optional...
  indices/
    lds-val/
```

The CSV must contain at least these columns:

- `name`: image filename;
- `label`: e.g. `post_impressionism` or `ukiyo_e`;
- `split`: `train` or `test`.

`python -m sd_artbench.prepare_data` creates all files under `indices/`.
Dataset files and generated pickle files are ignored by Git.

If the images already exist elsewhere, symlink the directory instead of
duplicating it:

```bash
ln -s /absolute/path/to/artbench-10-imagefolder \
  dataset/images/artbench-10-imagefolder
```

