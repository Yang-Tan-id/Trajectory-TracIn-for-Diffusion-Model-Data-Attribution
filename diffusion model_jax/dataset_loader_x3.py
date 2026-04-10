import csv
import colorsys
from typing import List, Tuple, Iterator, Optional, Sequence

import numpy as np
import jax
import jax.numpy as jnp


# Fixed saturation and value (brightness) for all colors
FIXED_SATURATION = 0.9
FIXED_VALUE = 0.9
TILE_SIZE = 3  # 3x3 pixel grid


class ColorGridDatasetJAX:
    def __init__(
        self,
        csv_path: str,
        grid_size: int = 3,
        fixed_s: float = 0.9,
        fixed_v: float = 0.9,
        label_start: Optional[int] = None,
        row_indices: Optional[Sequence[int]] = None,
        subset_ranges: Optional[Sequence[Tuple[int, int]]] = None,
    ):
        self.csv_path = csv_path
        self.grid_size = int(grid_size)
        self.num_tiles = self.grid_size * self.grid_size
        self.fixed_s = fixed_s
        self.fixed_v = fixed_v

        # columns: [0]=id, [1..num_tiles]=tiles, [label_start..]=labels
        self.label_start = (1 + self.num_tiles) if label_start is None else int(label_start)

        # Only one of row_indices or subset_ranges should be used
        if row_indices is not None and subset_ranges is not None and len(subset_ranges) > 0:
            raise ValueError("Use only one of row_indices or subset_ranges, not both.")

        # None or empty => use all rows
        if row_indices is None or len(row_indices) == 0:
            self.row_indices = None
        else:
            self.row_indices = [int(i) for i in row_indices]

        # None or empty => use all rows
        if subset_ranges is None or len(subset_ranges) == 0:
            self.subset_ranges = None
        else:
            self.subset_ranges = [(int(start), int(count)) for start, count in subset_ranges]

        self.rows: List[List[str]] = []
        self.vocab = {}
        self.id_to_label = {}
        self._load()

    def _expand_subset_ranges(self, n: int) -> List[int]:
        """
        Convert subset_ranges like [(1000, 100), (2000, 50)] into a flat list of row indices.

        Important:
        - overlaps are preserved, so duplicated rows appear multiple times
        - if a range exceeds dataset size, it is clipped to the end
        - if start is invalid or count <= 0, raise an error
        """
        assert self.subset_ranges is not None

        out: List[int] = []
        for start, count in self.subset_ranges:
            if count <= 0:
                raise ValueError(f"subset_ranges contains non-positive count: {(start, count)}")

            if start < 0:
                raise ValueError(f"subset_ranges contains negative start: {(start, count)}")

            if start >= n:
                raise IndexError(
                    f"subset_ranges start {start} is out of range for dataset of size {n}"
                )

            end = start + count
            if end > n:
                end = n

            out.extend(range(start, end))

        return out

    def _load(self):
        all_rows: List[List[str]] = []
        with open(self.csv_path, newline="") as fh:
            reader = csv.reader(fh)
            for row in reader:
                if not row or row[0].lower() == "id":
                    continue
                all_rows.append(row)

        n = len(all_rows)

        if self.row_indices is not None:
            bad = [i for i in self.row_indices if i < 0 or i >= n]
            if bad:
                raise IndexError(
                    f"row_indices contain out-of-range values: {bad[:10]} "
                    f"(dataset has {n} rows)"
                )
            self.rows = [all_rows[i] for i in self.row_indices]

        elif self.subset_ranges is not None:
            expanded_indices = self._expand_subset_ranges(n)
            self.rows = [all_rows[i] for i in expanded_indices]

        else:
            self.rows = all_rows

        labels = set()
        for row in self.rows:
            for label in row[self.label_start:]:
                if label:
                    labels.add(label)

        labels = sorted(labels)
        self.vocab = {lab: i for i, lab in enumerate(labels)}
        self.id_to_label = {i: lab for lab, i in self.vocab.items()}

    def __len__(self):
        return len(self.rows)

    def _hues_from_row(self, row: List[str]) -> List[float]:
        hues = []
        for i in range(1, 1 + self.num_tiles):
            val = row[i]
            try:
                hue = float(val)
            except Exception:
                hue = float(val.strip('"'))
            hues.append(hue)
        return hues

    def _image_from_hues(self, hues: List[float]) -> np.ndarray:
        # create H,W,C then transpose to C,H,W
        H = W = self.grid_size
        img = np.zeros((H, W, 3), dtype=np.float32)

        k = 0
        for i in range(H):
            for j in range(W):
                h = hues[k]
                r, g, b = colorsys.hsv_to_rgb(h, self.fixed_s, self.fixed_v)
                img[i, j, :] = (r, g, b)
                k += 1

        return np.transpose(img, (2, 0, 1))  # C,H,W

    def _cond_vector_from_row(self, row: List[str]) -> np.ndarray:
        vec = np.zeros(len(self.vocab), dtype=np.float32)
        for label in row[self.label_start:]:
            if label and label in self.vocab:
                vec[self.vocab[label]] = 1.0
        return vec

    def __getitem__(self, idx: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        row = self.rows[idx]
        hues = self._hues_from_row(row)
        img = self._image_from_hues(hues)
        cond = self._cond_vector_from_row(row)
        return jnp.array(img, dtype=jnp.float32), jnp.array(cond, dtype=jnp.float32)

    def get_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Materialize the whole dataset as NumPy arrays:
          x: (N, C, H, W)
          y: (N, vocab_size)
        """
        xs = []
        ys = []
        for i in range(len(self)):
            x, y = self[i]
            xs.append(np.array(x, dtype=np.float32))
            ys.append(np.array(y, dtype=np.float32))
        return np.stack(xs), np.stack(ys)

    def get_jax(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Materialize the whole dataset as JAX arrays:
          x: (N, C, H, W)
          y: (N, vocab_size)
        """
        x_np, y_np = self.get_numpy()
        return jnp.array(x_np), jnp.array(y_np)

    def batch_iterator(
        self,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
        indices = np.arange(len(self))
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            end = start + batch_size
            if end > len(indices) and drop_last:
                break

            batch_idx = indices[start:end]
            xs = []
            ys = []
            for i in batch_idx:
                x, y = self[i]
                xs.append(np.array(x, dtype=np.float32))
                ys.append(np.array(y, dtype=np.float32))

            yield jnp.array(np.stack(xs)), jnp.array(np.stack(ys))


def quick_load(
    csv_path: str,
    row_indices: Optional[Sequence[int]] = None,
    subset_ranges: Optional[Sequence[Tuple[int, int]]] = None,
) -> ColorGridDatasetJAX:
    """Helper to build dataset and return it."""
    return ColorGridDatasetJAX(
        csv_path,
        row_indices=row_indices,
        subset_ranges=subset_ranges,
    )


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print('Usage: python3 dataset_loader_jax.py generated_database/<seed>_<n>.csv')
        raise SystemExit(1)

    path = sys.argv[1]

    # Example: use whole dataset
    ds = quick_load(path)

    # Example: use selected contiguous subsets
    # ds = quick_load(path, subset_ranges=[(1000, 100), (2000, 50)])

    # Example: overlapping subsets -> overlapping rows appear twice
    # ds = quick_load(path, subset_ranges=[(1000, 100), (1050, 20)])

    print(f'Loaded {len(ds)} samples, vocab size {len(ds.vocab)}')

    img, cond = ds[0]
    print('Image tensor shape:', img.shape)   # (C, H, W)
    print('Cond vector sum (num active labels):', int(cond.sum()))

    for x_batch, y_batch in ds.batch_iterator(batch_size=4, shuffle=True, seed=42):
        print('Batch x shape:', x_batch.shape)
        print('Batch y shape:', y_batch.shape)
        break