import os
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import jax
import jax.numpy as jnp
from PIL import Image, ImageOps


ExcludeRangeKey = Union[str, int]


def _normalize_image(x: np.ndarray, normalize: str) -> np.ndarray:
    x = x.astype(np.float32)
    if normalize == "zero_to_one":
        return x / 255.0
    if normalize == "minus_one_to_one":
        return x / 127.5 - 1.0
    raise ValueError("normalize must be 'zero_to_one' or 'minus_one_to_one'.")


def _decode_relpath(x):
    return x.decode("utf-8") if isinstance(x, bytes) else str(x)


class ArtBenchImageFolderLatentDataset:
    """
    ArtBench image-folder loader for latent-diffusion pipelines.

    Features:
    - train/test split loading
    - optional class subset by name or id
    - optional per-class range exclusions
    - optional exact per-class index exclusions
    - optional exact file exclusions by relative path
    - returns images as float32 in either [-1, 1] or [0, 1]
    - labels can be returned as integer class ids or one-hot vectors
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        class_names: Optional[Sequence[str]] = None,
        class_ids: Optional[Sequence[int]] = None,
        normalize: str = "minus_one_to_one",
        channels_last: bool = True,
        one_hot_labels: bool = False,
        image_size: int = 256,
        resize_mode: str = "shortest_center_crop",
        file_extensions: Sequence[str] = (".jpg", ".jpeg", ".png", ".webp"),
        exclude_ranges: Optional[Sequence[Tuple[ExcludeRangeKey, int, int]]] = None,
        exclude_indices: Optional[Dict[ExcludeRangeKey, Sequence[int]]] = None,
        exclude_files: Optional[Iterable[str]] = None,
    ):
        self.root = root
        self.split = split
        self.split_root = os.path.join(root, split)
        self.normalize = normalize
        self.channels_last = channels_last
        self.one_hot_labels = one_hot_labels
        self.image_size = int(image_size)
        self.resize_mode = resize_mode
        self.file_extensions = tuple(ext.lower() for ext in file_extensions)
        self.exclude_ranges = list(exclude_ranges) if exclude_ranges else []
        self.exclude_indices = {
            k: [int(vv) for vv in v]
            for k, v in (exclude_indices.items() if exclude_indices else [])
        }
        self.exclude_files = {str(x).replace("\\", "/") for x in (exclude_files or [])}

        if not os.path.isdir(self.split_root):
            raise ValueError(f"Split directory does not exist: {self.split_root}")

        self.label_names = self._discover_label_names()
        self.name_to_id = {name: i for i, name in enumerate(self.label_names)}

        if class_names is not None and class_ids is not None:
            raise ValueError("Use only one of class_names or class_ids, not both.")

        allowed_names = None
        if class_names is not None:
            allowed_names = [str(name) for name in class_names]
        elif class_ids is not None:
            allowed_names = [self.label_names[int(i)] for i in class_ids]

        if allowed_names is not None:
            missing = [name for name in allowed_names if name not in self.name_to_id]
            if missing:
                raise ValueError(f"Requested classes not found: {missing}")
            self.label_names = allowed_names
            self.name_to_id = {name: i for i, name in enumerate(self.label_names)}

        self.samples = self._index_samples()
        self.num_classes = len(self.label_names)

    def _discover_label_names(self) -> List[str]:
        out = []
        for name in sorted(os.listdir(self.split_root)):
            path = os.path.join(self.split_root, name)
            if os.path.isdir(path) and not name.startswith("."):
                out.append(name)
        if not out:
            raise ValueError(f"No class folders found under {self.split_root}")
        return out

    def _key_matches_class(self, key: ExcludeRangeKey, class_name: str, class_id: int) -> bool:
        if isinstance(key, str):
            return key == class_name
        return int(key) == class_id

    def _apply_class_exclusions(
        self,
        class_name: str,
        class_id: int,
        relpaths: List[str],
    ) -> List[str]:
        n = len(relpaths)
        keep_mask = np.ones(n, dtype=bool)

        for key, start_idx, count in self.exclude_ranges:
            if not self._key_matches_class(key, class_name, class_id):
                continue
            start_idx = int(start_idx)
            count = int(count)

            if count <= 0:
                print(
                    f"[exclude_ranges warning] class {class_name}: "
                    f"count={count} is not positive, skipping."
                )
                continue

            if start_idx < 0:
                print(
                    f"[exclude_ranges warning] class {class_name}: "
                    f"start_idx={start_idx} < 0, clamping to 0."
                )
                start_idx = 0

            if start_idx >= n:
                print(
                    f"[exclude_ranges warning] class {class_name}: "
                    f"start_idx={start_idx} exceeds class size {n}. Nothing excluded."
                )
                continue

            end_idx = min(n, start_idx + count)
            if end_idx < start_idx + count:
                print(
                    f"[exclude_ranges warning] class {class_name}: "
                    f"requested exclusion [{start_idx}, {start_idx + count}) exceeds class size {n}. "
                    f"Excluding only [{start_idx}, {end_idx})."
                )

            keep_mask[start_idx:end_idx] = False
            print(
                f"[exclude_ranges] class {class_name}: excluded rows [{start_idx}, {end_idx}) "
                f"({end_idx - start_idx} samples)."
            )

        for key, indices in self.exclude_indices.items():
            if not self._key_matches_class(key, class_name, class_id):
                continue
            raw_idx = np.asarray(indices, dtype=np.int64)
            valid_mask = (raw_idx >= 0) & (raw_idx < n)
            invalid = raw_idx[~valid_mask]
            valid_idx = np.unique(raw_idx[valid_mask])
            if invalid.size > 0:
                print(
                    f"[exclude_indices warning] class {class_name}: "
                    f"{invalid.size} indices out of range [0, {n - 1}] were ignored."
                )
            if valid_idx.size > 0:
                keep_mask[valid_idx] = False
                print(
                    f"[exclude_indices] class {class_name}: excluded {valid_idx.size} exact rows."
                )

        out = []
        for i, relpath in enumerate(relpaths):
            relpath_norm = relpath.replace("\\", "/")
            if relpath_norm in self.exclude_files:
                keep_mask[i] = False

        for i, relpath in enumerate(relpaths):
            if keep_mask[i]:
                out.append(relpath)
        return out

    def _index_samples(self) -> List[Tuple[str, int, str]]:
        samples: List[Tuple[str, int, str]] = []
        for class_name in self.label_names:
            class_id = self.name_to_id[class_name]
            class_dir = os.path.join(self.split_root, class_name)
            relpaths = []
            for dirpath, _, filenames in os.walk(class_dir):
                filenames = sorted(filenames)
                for filename in filenames:
                    if filename.startswith("."):
                        continue
                    if not filename.lower().endswith(self.file_extensions):
                        continue
                    full_path = os.path.join(dirpath, filename)
                    relpath = os.path.relpath(full_path, self.split_root).replace("\\", "/")
                    relpaths.append(relpath)

            relpaths = self._apply_class_exclusions(class_name, class_id, sorted(relpaths))
            for relpath in relpaths:
                samples.append((os.path.join(self.split_root, relpath), class_id, relpath))
        return samples

    def _resize_image(self, img: Image.Image) -> Image.Image:
        target = self.image_size
        if self.resize_mode == "resize":
            return img.resize((target, target), Image.Resampling.BICUBIC)

        if self.resize_mode == "shortest_center_crop":
            w, h = img.size
            scale = target / min(w, h)
            new_w = max(target, round(w * scale))
            new_h = max(target, round(h * scale))
            img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
            left = (img.width - target) // 2
            top = (img.height - target) // 2
            return img.crop((left, top, left + target, top + target))

        if self.resize_mode == "identity_or_center_crop":
            if img.width == target and img.height == target:
                return img
            if img.width < target or img.height < target:
                return img.resize((target, target), Image.Resampling.BICUBIC)
            left = (img.width - target) // 2
            top = (img.height - target) // 2
            return img.crop((left, top, left + target, top + target))

        raise ValueError(
            "resize_mode must be 'resize', 'shortest_center_crop', or 'identity_or_center_crop'."
        )

    def _load_image(self, path: str) -> np.ndarray:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            img = self._resize_image(img)
            arr = np.asarray(img, dtype=np.float32)

        arr = _normalize_image(arr, self.normalize)
        if not self.channels_last:
            arr = np.transpose(arr, (2, 0, 1))
        return arr.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return len(self.samples)

    def _format_label(self, y: int):
        if self.one_hot_labels:
            return jax.nn.one_hot(y, self.num_classes, dtype=jnp.float32)
        return jnp.asarray(y, dtype=jnp.int32)

    def __getitem__(self, idx: int):
        path, class_id, _ = self.samples[idx]
        x = jnp.asarray(self._load_image(path), dtype=jnp.float32)
        y = self._format_label(int(class_id))
        return x, y

    def get_all(self):
        xs = [self._load_image(path) for path, _, _ in self.samples]
        x = jnp.asarray(np.stack(xs, axis=0), dtype=jnp.float32)
        ys = np.asarray([class_id for _, class_id, _ in self.samples], dtype=np.int32)
        if self.one_hot_labels:
            y = jax.nn.one_hot(jnp.asarray(ys), self.num_classes, dtype=jnp.float32)
        else:
            y = jnp.asarray(ys, dtype=jnp.int32)
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
            xb = [self._load_image(self.samples[i][0]) for i in idx]
            yb = np.asarray([self.samples[i][1] for i in idx], dtype=np.int32)

            x = jnp.asarray(np.stack(xb, axis=0), dtype=jnp.float32)
            if self.one_hot_labels:
                y = jax.nn.one_hot(jnp.asarray(yb), self.num_classes, dtype=jnp.float32)
            else:
                y = jnp.asarray(yb, dtype=jnp.int32)
            yield x, y

    def class_counts(self) -> Dict[str, int]:
        counts = {name: 0 for name in self.label_names}
        ys = np.asarray([class_id for _, class_id, _ in self.samples], dtype=np.int32)
        uniq, cnt = np.unique(ys, return_counts=True)
        for i, c in zip(uniq.tolist(), cnt.tolist()):
            counts[self.label_names[i]] = c
        return counts

    def relpaths(self) -> List[str]:
        return [relpath for _, _, relpath in self.samples]


class ArtBenchLatentDataset:
    """
    Loader for cached latent tensors saved as .npz.

    Expected arrays in the npz:
    - latents: (N, H, W, C) float32
    - labels: (N,) int32
    - relpaths: (N,) optional utf-8 strings
    - class_names: (K,) optional utf-8 strings
    """

    def __init__(
        self,
        npz_path: str,
        one_hot_labels: bool = False,
        class_names: Optional[Sequence[str]] = None,
        class_ids: Optional[Sequence[int]] = None,
        exclude_indices: Optional[Sequence[int]] = None,
        exclude_files: Optional[Iterable[str]] = None,
    ):
        payload = np.load(npz_path, allow_pickle=True)
        self.latents = np.asarray(payload["latents"], dtype=np.float32)
        self.labels = np.asarray(payload["labels"], dtype=np.int32)
        self.relpaths_arr = None
        if "relpaths" in payload:
            self.relpaths_arr = np.asarray([_decode_relpath(x) for x in payload["relpaths"]], dtype=object)
        if "class_names" in payload:
            self.label_names = [_decode_relpath(x) for x in payload["class_names"]]
        else:
            max_id = int(self.labels.max()) if len(self.labels) > 0 else -1
            self.label_names = [f"class_{i}" for i in range(max_id + 1)]

        self.one_hot_labels = one_hot_labels
        self.num_classes = len(self.label_names)

        keep = np.ones(len(self.labels), dtype=bool)

        if exclude_indices is not None:
            ex_idx = np.asarray(list(exclude_indices), dtype=np.int64)
            valid = ex_idx[(ex_idx >= 0) & (ex_idx < len(keep))]
            keep[np.unique(valid)] = False

        if exclude_files is not None and self.relpaths_arr is not None:
            exclude_files = {str(x).replace("\\", "/") for x in exclude_files}
            for i, relpath in enumerate(self.relpaths_arr.tolist()):
                if relpath.replace("\\", "/") in exclude_files:
                    keep[i] = False

        allowed_ids = None
        if class_names is not None and class_ids is not None:
            raise ValueError("Use only one of class_names or class_ids, not both.")
        if class_names is not None:
            name_to_id = {name: i for i, name in enumerate(self.label_names)}
            allowed_ids = sorted(name_to_id[str(name)] for name in class_names)
        elif class_ids is not None:
            allowed_ids = sorted(int(i) for i in class_ids)

        if allowed_ids is not None:
            keep &= np.isin(self.labels, np.asarray(allowed_ids, dtype=np.int32))

        self.latents = self.latents[keep]
        self.labels = self.labels[keep]
        if self.relpaths_arr is not None:
            self.relpaths_arr = self.relpaths_arr[keep]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        x = jnp.asarray(self.latents[idx], dtype=jnp.float32)
        y = jax.nn.one_hot(self.labels[idx], self.num_classes, dtype=jnp.float32) if self.one_hot_labels else jnp.asarray(self.labels[idx], dtype=jnp.int32)
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
            x = jnp.asarray(self.latents[idx], dtype=jnp.float32)
            if self.one_hot_labels:
                y = jax.nn.one_hot(jnp.asarray(self.labels[idx]), self.num_classes, dtype=jnp.float32)
            else:
                y = jnp.asarray(self.labels[idx], dtype=jnp.int32)
            yield x, y


def save_latent_dataset_npz(
    out_path: str,
    latents: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    relpaths: Optional[Sequence[str]] = None,
):
    payload = {
        "latents": np.asarray(latents, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.int32),
        "class_names": np.asarray(list(class_names), dtype=object),
    }
    if relpaths is not None:
        payload["relpaths"] = np.asarray(list(relpaths), dtype=object)
    np.savez(out_path, **payload)


if __name__ == "__main__":
    root = "./databases/artbench-10-imagefolder-split"

    # Example 1: exclude by per-class range
    # Meaning:
    # - in class "art_nouveau", exclude sorted rows [0, 100)
    # - in class id 1, exclude sorted rows [50, 70)
    exclude_ranges = [
        ("art_nouveau", 0, 100),
        (1, 50, 20),
    ]

    # Example 2: exclude by exact per-class indices
    # Meaning:
    # - in class "baroque", exclude the 0th, 10th, and 25th files
    # - in class id 3, exclude the 1st, 2nd, and 3rd files
    exclude_indices = {
        "baroque": [0, 10, 25],
        3: [1, 2, 3],
    }

    # Example 3: exclude by exact file set
    # Paths are relative to split_root, e.g. train/ is not included here.
    exclude_files = {
        "art_nouveau/a-y-jackson_algoma-in-november-1935.jpg",
        "baroque/some_painting.jpg",
    }

    ds = ArtBenchImageFolderLatentDataset(
        root=root,
        split="train",
        normalize="minus_one_to_one",
        channels_last=True,
        one_hot_labels=False,
        image_size=128,
        resize_mode="shortest_center_crop",
        exclude_ranges=exclude_ranges,
        exclude_indices=exclude_indices,
        exclude_files=exclude_files,
    )

    print("dataset size:", len(ds))
    print("class counts:", ds.class_counts())

    x0, y0 = ds[0]
    print("single sample image shape:", x0.shape)
    print("single sample label:", y0)

    # Example 4: class subset + exclude file set
    ds_subset = ArtBenchImageFolderLatentDataset(
        root=root,
        split="train",
        class_names=["art_nouveau", "baroque"],
        image_size=128,
        exclude_files={
            "art_nouveau/a-y-jackson_algoma-in-november-1935.jpg",
        },
    )

    print("subset size:", len(ds_subset))
    print("subset class counts:", ds_subset.class_counts())

    # Example 5: iterate mini-batches
    for xb, yb in ds_subset.batch_iterator(batch_size=8, shuffle=True, seed=42):
        print("batch x:", xb.shape)
        print("batch y:", yb.shape)
        break
