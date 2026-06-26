import os
import pickle
from typing import Iterable, Optional, Sequence, Dict, List, Tuple, Iterator

import numpy as np
import jax
import jax.numpy as jnp


def unpickle(file_path: str):
    with open(file_path, "rb") as fo:
        data = pickle.load(fo, encoding="bytes")
    return data


def _decode_if_bytes(x):
    return x.decode("utf-8") if isinstance(x, bytes) else x


class CIFAR10DiffusionDataset:
    """
    CIFAR-10 loader for diffusion model training.

    Features:
    - load selected training batches, e.g. ["data_batch_1", "data_batch_3"]
    - optionally filter to a subset of classes
    - optionally exclude row ranges from specific CIFAR batches
    - returns images as float32 in either [-1, 1] or [0, 1]
    - labels can be returned as integer class ids or one-hot vectors
    """

    def __init__(
        self,
        root: str,
        batch_names: Optional[Sequence[str]] = None,
        train: bool = True,
        class_names: Optional[Sequence[str]] = None,
        class_ids: Optional[Sequence[int]] = None,
        normalize: str = "minus_one_to_one",   # "zero_to_one" or "minus_one_to_one"
        channels_last: bool = True,            # True -> (H,W,C), False -> (C,H,W)
        one_hot_labels: bool = False,
        exclude_ranges: Optional[Sequence[Tuple[int, int, int]]] = None,
    ):
        """
        Args:
            root:
                Path to CIFAR-10 extracted directory.
            batch_names:
                Example: ["data_batch_1", "data_batch_3"].
                If None and train=True, loads all five train batches.
                If None and train=False, loads test_batch.
            train:
                Whether to load train or test by default.
            class_names:
                Optional subset of classes by name, e.g. ["cat", "dog"].
            class_ids:
                Optional subset of classes by id, e.g. [3, 5].
            normalize:
                "zero_to_one" or "minus_one_to_one".
            channels_last:
                Whether images are (H,W,C) or (C,H,W).
            one_hot_labels:
                Whether to return labels as one-hot vectors.
            exclude_ranges:
                List of tuples (batch_id, start_index, count).
                Example:
                    [(3, 1500, 500), (2, 200, 30)]
                means:
                    exclude rows [1500, 2000) from data_batch_3
                    exclude rows [200, 230) from data_batch_2

                If the batch is not selected, nothing happens.
                If start_index >= batch size, print a warning and skip.
                If start_index + count exceeds batch size, exclude until the end and print a warning.
        """
        self.root = root
        self.train = train
        self.normalize = normalize
        self.channels_last = channels_last
        self.one_hot_labels = one_hot_labels
        self.exclude_ranges = list(exclude_ranges) if exclude_ranges else []

        self.label_names = self._load_label_names()
        self.name_to_id = {name: i for i, name in enumerate(self.label_names)}

        if batch_names is None:
            if train:
                batch_names = [
                    "data_batch_1",
                    "data_batch_2",
                    "data_batch_3",
                    "data_batch_4",
                    "data_batch_5",
                ]
            else:
                batch_names = ["test_batch"]

        self.batch_names = list(batch_names)

        allowed_ids = None
        if class_names is not None and class_ids is not None:
            raise ValueError("Use only one of class_names or class_ids, not both.")

        if class_names is not None:
            allowed_ids = sorted(self.name_to_id[name] for name in class_names)

        if class_ids is not None:
            allowed_ids = sorted(int(x) for x in class_ids)

        images, labels = self._load_batches(self.batch_names)

        if allowed_ids is not None:
            mask = np.isin(labels, np.array(allowed_ids))
            images = images[mask]
            labels = labels[mask]

        self.images = self._preprocess_images(images)
        self.labels = labels.astype(np.int32)

        self.num_classes = 10
        self.class_subset_ids = allowed_ids

    def _load_label_names(self) -> List[str]:
        meta_path = os.path.join(self.root, "batches.meta")
        meta = unpickle(meta_path)
        raw = meta[b"label_names"]
        return [_decode_if_bytes(x) for x in raw]

    def _load_one_batch(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        d = unpickle(path)

        x = d[b"data"]
        y = np.array(d[b"labels"], dtype=np.int32)

        if not isinstance(x, np.ndarray):
            x = np.array(x, dtype=np.uint8)

        return x, y

    def _apply_exclusions_to_batch(
        self,
        batch_name: str,
        x: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply exclusions specified in self.exclude_ranges to one CIFAR batch.

        batch_name is expected to look like 'data_batch_3'.
        """
        if not self.exclude_ranges:
            return x, y

        if not batch_name.startswith("data_batch_"):
            # test_batch or any other nonstandard batch name: no-op
            return x, y

        try:
            batch_id = int(batch_name.split("_")[-1])
        except Exception:
            return x, y

        n = len(y)
        keep_mask = np.ones(n, dtype=bool)

        for ex_batch_id, start_idx, count in self.exclude_ranges:
            if int(ex_batch_id) != batch_id:
                continue

            start_idx = int(start_idx)
            count = int(count)

            if count <= 0:
                print(
                    f"[exclude_ranges warning] batch {batch_id}: "
                    f"count={count} is not positive, skipping."
                )
                continue

            if start_idx < 0:
                print(
                    f"[exclude_ranges warning] batch {batch_id}: "
                    f"start_idx={start_idx} < 0, clamping to 0."
                )
                start_idx = 0

            if start_idx >= n:
                print(
                    f"[exclude_ranges warning] batch {batch_id}: "
                    f"start_idx={start_idx} exceeds batch size {n}. Nothing excluded."
                )
                continue

            end_idx = start_idx + count
            if end_idx > n:
                print(
                    f"[exclude_ranges warning] batch {batch_id}: "
                    f"requested exclusion [{start_idx}, {end_idx}) exceeds batch size {n}. "
                    f"Excluding only [{start_idx}, {n})."
                )
                end_idx = n

            keep_mask[start_idx:end_idx] = False
            print(
                f"[exclude_ranges] batch {batch_id}: excluded rows [{start_idx}, {end_idx}) "
                f"({end_idx - start_idx} samples)."
            )

        return x[keep_mask], y[keep_mask]

    def _load_batches(self, batch_names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        xs = []
        ys = []

        for batch_name in batch_names:
            path = os.path.join(self.root, batch_name)
            x, y = self._load_one_batch(path)

            # apply per-batch row exclusions before concatenation
            x, y = self._apply_exclusions_to_batch(batch_name, x, y)

            xs.append(x)
            ys.append(y)

        x = np.concatenate(xs, axis=0)
        y = np.concatenate(ys, axis=0)
        return x, y

    def _preprocess_images(self, flat_images: np.ndarray) -> np.ndarray:
        x = flat_images.reshape(-1, 3, 32, 32).astype(np.float32)

        if self.channels_last:
            x = np.transpose(x, (0, 2, 3, 1))  # (N,32,32,3)

        if self.normalize == "zero_to_one":
            x = x / 255.0
        elif self.normalize == "minus_one_to_one":
            x = x / 127.5 - 1.0
        else:
            raise ValueError("normalize must be 'zero_to_one' or 'minus_one_to_one'.")

        return x.astype(np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def _format_label(self, y: int):
        if self.one_hot_labels:
            return jax.nn.one_hot(y, self.num_classes, dtype=jnp.float32)
        return jnp.asarray(y, dtype=jnp.int32)

    def __getitem__(self, idx: int):
        x = jnp.asarray(self.images[idx], dtype=jnp.float32)
        y = self._format_label(int(self.labels[idx]))
        return x, y

    def get_all(self):
        x = jnp.asarray(self.images, dtype=jnp.float32)

        if self.one_hot_labels:
            y = jax.nn.one_hot(jnp.asarray(self.labels), self.num_classes, dtype=jnp.float32)
        else:
            y = jnp.asarray(self.labels, dtype=jnp.int32)

        return x, y

    def batch_iterator(
        self,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
        n = len(self)
        indices = np.arange(n)

        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)

        for start in range(0, n, batch_size):
            end = start + batch_size
            if end > n and drop_last:
                break

            idx = indices[start:end]
            x = jnp.asarray(self.images[idx], dtype=jnp.float32)

            if self.one_hot_labels:
                y = jax.nn.one_hot(
                    jnp.asarray(self.labels[idx]),
                    self.num_classes,
                    dtype=jnp.float32,
                )
            else:
                y = jnp.asarray(self.labels[idx], dtype=jnp.int32)

            yield x, y

    def class_counts(self) -> Dict[str, int]:
        counts = {name: 0 for name in self.label_names}
        uniq, cnt = np.unique(self.labels, return_counts=True)
        for i, c in zip(uniq.tolist(), cnt.tolist()):
            counts[self.label_names[i]] = c
        return counts


if __name__ == "__main__":
    root = "./cifar-10-batches-py"

    # Example exclusions:
    # - from data_batch_3, exclude rows [1500, 2000)
    # - from data_batch_2, exclude rows [200, 230)
    exclude_ranges = [
        (3, 1500, 500),
        (2, 200, 30),
    ]

    # Example 1: use only batch 1 and batch 3, with exclusions
    ds = CIFAR10DiffusionDataset(
        root=root,
        batch_names=["data_batch_1", "data_batch_3"],
        train=True,
        normalize="minus_one_to_one",
        channels_last=True,
        exclude_ranges=exclude_ranges,
    )

    print("dataset size:", len(ds))
    print("class counts:", ds.class_counts())

    x0, y0 = ds[0]
    print("single sample image shape:", x0.shape)
    print("single sample label:", y0)

    # Example 2: only airplane and automobile from batch 1 + batch 3, with exclusions
    ds_subset = CIFAR10DiffusionDataset(
        root=root,
        batch_names=["data_batch_1", "data_batch_3"],
        class_names=["airplane", "automobile"],
        normalize="minus_one_to_one",
        channels_last=True,
        exclude_ranges=exclude_ranges,
    )

    print("subset size:", len(ds_subset))
    print("subset class counts:", ds_subset.class_counts())

    # Example 3: iterate mini-batches
    for xb, yb in ds_subset.batch_iterator(batch_size=64, shuffle=True, seed=42):
        print("batch x:", xb.shape)
        print("batch y:", yb.shape)
        break

    # Example 4: exclusion range exceeds batch size
    ds_overflow = CIFAR10DiffusionDataset(
        root=root,
        batch_names=["data_batch_3"],
        normalize="minus_one_to_one",
        channels_last=True,
        exclude_ranges=[(3, 9800, 500)],  # will exclude [9800, 10000) and print warning
    )

    print("overflow example size:", len(ds_overflow))

    # Example 5: exclusion refers to a batch that is not selected
    ds_ignore = CIFAR10DiffusionDataset(
        root=root,
        batch_names=["data_batch_1"],
        normalize="minus_one_to_one",
        channels_last=True,
        exclude_ranges=[(3, 1500, 500)],  # batch 3 not loaded, so nothing happens
    )

    print("ignore example size:", len(ds_ignore))