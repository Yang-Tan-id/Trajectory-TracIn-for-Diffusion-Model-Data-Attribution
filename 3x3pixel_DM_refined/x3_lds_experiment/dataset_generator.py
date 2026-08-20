import numpy as np
import colorsys
import random
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import os
import json
import csv

# 7 discrete color labels (evenly divided color wheel)
COLOR_LABELS = {
    'red': 0,
    'orange': 1,
    'yellow': 2,
    'green': 3,
    'cyan': 4,
    'blue': 5,
    'purple': 6
}

# Color wheel positions (hue values from 0 to 1)
# Centered on standard color positions with equal spacing
HUE_RANGES = {
    'red': (0.92, 0.08),          # 331-29 degrees (wraps around)
    'orange': (0.08, 0.15),       # 29-54 degrees
    'yellow': (0.15, 0.25),       # 54-90 degrees
    'green': (0.25, 0.42),        # 90-151 degrees
    'cyan': (0.42, 0.58),         # 151-209 degrees
    'blue': (0.58, 0.75),         # 209-270 degrees
    'purple': (0.75, 0.92)        # 270-331 degrees
}

# Fixed saturation and value (brightness) for all colors
FIXED_SATURATION = 0.9
FIXED_VALUE = 0.9


def hue_to_label(hue: float) -> str:
    """Convert hue (0-1) to discrete color label. Red wraps around (0.92-1.0 and 0.0-0.08)."""
    if hue >= 0.92 or hue < 0.08:
        return 'red'
    
    for color_name in ['orange', 'yellow', 'green', 'cyan', 'blue', 'purple']:
        hue_min, hue_max = HUE_RANGES[color_name]
        if hue_min <= hue < hue_max:
            return color_name
    
    return 'red'


def random_color() -> Tuple[float, float, float]:
    hue = np.random.random()  # Random hue between 0 and 1
    saturation = FIXED_SATURATION
    value = FIXED_VALUE
    rgb = colorsys.hsv_to_rgb(hue, saturation, value)
    return rgb


def get_color_label(hue: float) -> str:
    return hue_to_label(hue)


class ColoredGrid:
    def __init__(self, shape_color_rgb: Tuple[float, float, float],
                 background_color_rgb: Tuple[float, float, float],
                 shape_name: str):
        self.shape_color_rgb = shape_color_rgb
        self.background_color_rgb = background_color_rgb
        self.shape_name = shape_name
        
        # Extract hue from RGB colors
        shape_hsv = colorsys.rgb_to_hsv(*shape_color_rgb)
        bg_hsv = colorsys.rgb_to_hsv(*background_color_rgb)
        
        self.shape_color_label = hue_to_label(shape_hsv[0])
        self.background_color_label = hue_to_label(bg_hsv[0])
        
        # Generate the grid (9 tiles in 3x3 layout)
        self.grid = np.zeros((3, 3, 3), dtype=np.float32)  # 3x3 tiles, RGB per tile
        self._generate_grid()
    
    def _generate_grid(self):
        shape_func = SHAPE_FUNCTIONS.get(self.shape_name)
        if shape_func is None:
            raise ValueError(f"Unknown shape: {self.shape_name}")
        
        shape_mask = shape_func()
        
        # Fill grid: shape_color for True, background_color for False
        for i in range(3):
            for j in range(3):
                if shape_mask[i, j]:
                    self.grid[i, j] = self.shape_color_rgb
                else:
                    self.grid[i, j] = self.background_color_rgb
    
    def get_labels(self) -> Dict[str, str]:
        return {
            'shape_color': self.shape_color_label,
            'background_color': self.background_color_label,
            'shape': self.shape_name
        }
    
    def get_grid_array(self) -> np.ndarray:
        return self.grid


# Shape functions - each returns a 3x3 boolean mask where True = shape color

def shape_big_square() -> np.ndarray:
    """
    Big square: fills the entire 3x3 grid with the shape color.
    Grid positions:
    0 1 2
    3 4 5
    6 7 8
    """
    return np.ones((3, 3), dtype=bool)


def shape_emptiness() -> np.ndarray:
    """
    Emptiness: fills the entire 3x3 grid with the background color (no shape).
    """
    return np.zeros((3, 3), dtype=bool)


def shape_dot() -> np.ndarray:
    """
    Dot: only the middle tile (position 4) is the shape color,
    everything else is background color.
    """
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True
    return mask


def shape_ring() -> np.ndarray:
    """
    Ring: fills everything except the middle tile with the shape color.
    """
    mask = np.ones((3, 3), dtype=bool)
    mask[1, 1] = False
    return mask


def shape_h() -> np.ndarray:
    """
    H shape: positions 1 and 7 (second and eighth) are background color,
    all others are shape color.
    Grid positions:
    0 1 2
    3 4 5
    6 7 8
    """
    mask = np.ones((3, 3), dtype=bool)
    mask[0, 1] = False  # position 1
    mask[2, 1] = False  # position 7
    return mask


def shape_horizontal_h() -> np.ndarray:
    """
    Horizontal H: only positions 3 and 5 (fourth and sixth) are background color,
    all others are shape color.
    Grid positions:
    0 1 2
    3 4 5
    6 7 8
    """
    mask = np.ones((3, 3), dtype=bool)
    mask[1, 0] = False  # position 3
    mask[1, 2] = False  # position 5
    return mask


def shape_cross() -> np.ndarray:
    """
    Cross: positions 1, 3, 4, 5, 7 are shape color (center cross pattern).
    Grid positions:
    0 1 2
    3 4 5
    6 7 8
    """
    mask = np.zeros((3, 3), dtype=bool)
    mask[0, 1] = True  # position 1
    mask[1, 0] = True  # position 3
    mask[1, 1] = True  # position 4
    mask[1, 2] = True  # position 5
    mask[2, 1] = True  # position 7
    return mask

def shape_x() -> np.ndarray:
    """
    X shape: positions 0, 2, 4, 6, 8 are shape color (diagonal X pattern).
    Grid positions:
    0 1 2
    3 4 5
    6 7 8
    """
    mask = np.zeros((3, 3), dtype=bool)
    mask[0, 0] = True  # position 0
    mask[0, 2] = True  # position 2
    mask[1, 1] = True  # position 4
    mask[2, 0] = True  # position 6
    mask[2, 2] = True  # position 8
    return mask


def shape_diamond() -> np.ndarray:
    """
    Diamond shape: positions 1, 3, 5, 7 are shape color (diagonal X pattern).
    Grid positions:
    0 1 2
    3 4 5
    6 7 8
    """
    mask = np.zeros((3, 3), dtype=bool)
    mask[0, 1] = True  # position 1
    mask[1, 0] = True  # position 3
    mask[1, 2] = True  # position 5
    mask[2, 1] = True  # position 7
    return mask


# Dictionary mapping shape names to their functions
SHAPE_FUNCTIONS = {
    'big_square': shape_big_square,
    'emptiness': shape_emptiness,
    'dot': shape_dot,
    'ring': shape_ring,
    'h': shape_h,
    'horizontal_h': shape_horizontal_h,
    'cross': shape_cross,
    'x': shape_x,
    'diamond': shape_diamond
}


def generate_dataset(num_samples: int = 100, seed: int = 42) -> List[Dict]:
    """
    Generate a dataset of colored grids with different shapes.
    
    Args:
        num_samples: Number of samples to generate
        seed: Random seed for reproducibility
    
    Returns:
        List of dictionaries containing grid arrays and labels
    """
    np.random.seed(seed)
    dataset = []
    shapes = list(SHAPE_FUNCTIONS.keys())
    
    for _ in range(num_samples):
        # Randomly select a shape
        shape_name = np.random.choice(shapes)
        
        # Generate two random colors with same saturation and value
        shape_color_rgb = random_color()
        background_color_rgb = random_color()
        
        # Create the colored grid
        grid = ColoredGrid(shape_color_rgb, background_color_rgb, shape_name)
        
        # Collect the data
        sample = {
            'grid': grid.get_grid_array(),
            'labels': grid.get_labels(),
            'shape_color_rgb': shape_color_rgb,
            'background_color_rgb': background_color_rgb
        }
        
        dataset.append(sample)
    
    return dataset


def visualize_sample(sample: Dict, title: str = None) -> None:
    grid = sample['grid']
    labels = sample['labels']
    
    # Resize grid for visualization (each tile becomes 100x100 pixels)
    tile_size = 100
    img = np.zeros((3 * tile_size, 3 * tile_size, 3), dtype=np.float32)
    
    for i in range(3):
        for j in range(3):
            img[i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size] = grid[i, j]
    
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis('off')
    
    if title is None:
        title = f"Shape: {labels['shape']}, Color: {labels['shape_color']}, BG: {labels['background_color']}"
    
    plt.title(title)
    plt.tight_layout()
    plt.show()


def save_dataset(dataset: List[Dict], output_dir: str = 'generated_database', seed: int = 42, num_samples: int = 100) -> str:
    """
    Save dataset to CSV file in the generated_database folder.
    Each row: id, tile_color_1-9, shuffled_label_1, shuffled_label_2, shuffled_label_3
    Label columns show: shape_color_(value), background_color_(value), shape_(value)
    
    Args:
        dataset: List of samples to save
        output_dir: Output directory name
        seed: Random seed used to generate the dataset
        num_samples: Number of samples in the dataset
    
    Returns:
        Path to saved CSV file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename as (seed)_(num_samples).csv
    filename = f"{seed}_{num_samples}.csv"
    filepath = os.path.join(output_dir, filename)
    
    # Label names in order
    label_keys = ['shape_color', 'background_color', 'shape']
    
    # Save as CSV
    with open(filepath, 'w', newline='') as csvfile:
        # Column headers: id, tile_1-9, label_1, label_2, label_3
        fieldnames = ['id'] + [f'tile_{i}' for i in range(1, 10)] + ['label_1', 'label_2', 'label_3']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for idx, sample in enumerate(dataset):
            # Get the 9 tile colors from the grid
            grid = sample['grid']  # Shape: (3, 3, 3) - 3x3 tiles with RGB values
            tile_colors = []
            for i in range(3):
                for j in range(3):
                    rgb = grid[i, j]
                    # Extract hue only (ignore saturation and value/density)
                    hue, _, _ = colorsys.rgb_to_hsv(rgb[0], rgb[1], rgb[2])
                    # Store hue as a string with three decimal places
                    hue_str = f"{hue:.3f}"
                    tile_colors.append(hue_str)
            
            # Get the original labels
            labels = sample['labels']
            # Create label strings with format: key_value
            label_values = [f"{key}_{labels[key]}" for key in label_keys]
            
            # Shuffle the order of labels using random seed per sample
            sample_shuffle_rng = np.random.RandomState(seed + idx)
            shuffled_indices = sample_shuffle_rng.permutation(3)
            shuffled_labels = [label_values[i] for i in shuffled_indices]
            
            # Create row
            row = {'id': idx}
            for i, color in enumerate(tile_colors):
                row[f'tile_{i+1}'] = color
            for i, label in enumerate(shuffled_labels):
                row[f'label_{i+1}'] = label
            
            writer.writerow(row)
    
    print(f"Dataset saved to {filepath}")
    return filepath


def _save_subset_indices_csv(path: str, selected_indices: List[int]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx"])
        for i in selected_indices:
            w.writerow([int(i)])


def _save_subset_indices_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _read_csv_rows(csv_path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError(f"Empty CSV or missing header: {csv_path}")
        rows = list(reader)
    return rows, fieldnames


def _write_csv_rows(csv_path: str, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _row_labels(row: Dict[str, str]) -> List[str]:
    labs = []
    for k in ("label_1", "label_2", "label_3"):
        v = row.get(k, "")
        if v is not None and str(v).strip() != "":
            labs.append(str(v).strip())
    return labs


def _sample_without_replacement(rng: random.Random, pool: List[int], k: int) -> List[int]:
    if k <= 0:
        return []
    if k >= len(pool):
        return list(pool)
    return rng.sample(pool, k)


def generate_appended_dataset_from_base(
    base_csv_path: str,
    out_csv_path: str,
    num: int,
    mode: str,
    seed: int = 0,
    require_all_labels: Optional[List[str]] = None,
    balanced_labels: Optional[List[str]] = None,
    allow_duplicates_when_insufficient: bool = True,
    # NEW: where to store selected subset indices
    subset_save_path: Optional[str] = None,
    subset_save_format: str = "csv",  # "csv" | "json"
) -> str:
    """
    从 base dataset (CSV) 里抽取一个 size=num 的子集，然后 append 到 base 后面，生成新 CSV。
    同时可选把抽到的 subset indices 保存到 subset_save_path。
    """
    if num <= 0:
        raise ValueError("num must be > 0")
    mode = mode.lower().strip()
    subset_save_format = subset_save_format.lower().strip()
    if subset_save_format not in ("csv", "json"):
        raise ValueError("subset_save_format must be 'csv' or 'json'")

    rng = random.Random(seed)

    base_rows, fieldnames = _read_csv_rows(base_csv_path)
    N = len(base_rows)
    if N == 0:
        raise ValueError("Base dataset is empty.")

    labels_per_row = [_row_labels(r) for r in base_rows]
    label_sets = [set(labs) for labs in labels_per_row]

    selected: List[int] = []
    remaining = list(range(N))

    # For logging / saving metadata
    meta: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base_csv_path": base_csv_path,
        "out_csv_path": out_csv_path,
        "base_N": N,
        "requested_num": num,
        "mode": mode,
        "seed": seed,
        "require_all_labels": require_all_labels,
        "balanced_labels": balanced_labels,
        "allow_duplicates_when_insufficient": allow_duplicates_when_insufficient,
        "events": [],
    }

    def log_event(name: str, **kwargs):
        meta["events"].append({"event": name, **kwargs})

    def take(picked: List[int], note: str):
        nonlocal remaining
        picked_set = set(picked)
        remaining = [i for i in remaining if i not in picked_set]
        selected.extend(picked)
        log_event(note, picked=len(picked), remaining=len(remaining))

    # -------- mode handlers --------
    if mode == "random":
        picked = _sample_without_replacement(rng, remaining, num)
        take(picked, "pick_random")
        print(f"[append-ds] mode=random, requested={num}, picked={len(picked)}, base_N={N}")

    elif mode == "all":
        if not require_all_labels:
            raise ValueError("mode='all' requires require_all_labels (list of strings)")
        req = set(require_all_labels)
        cands = [i for i in remaining if req.issubset(label_sets[i])]
        picked = _sample_without_replacement(rng, cands, num)
        log_event("candidates_all", candidates=len(cands), require_all=sorted(req))
        take(picked, "pick_all")
        print(
            f"[append-ds] mode=all, require_all={sorted(req)}, candidates={len(cands)}, "
            f"requested={num}, picked={len(picked)}, base_N={N}"
        )

    elif mode == "balanced":
        if not balanced_labels or len(balanced_labels) == 0:
            raise ValueError("mode='balanced' requires balanced_labels (list of strings)")
        labs = [str(x) for x in balanced_labels]
        m = len(labs)
        per = num // m
        rem = num - per * m

        print(f"[append-ds] mode=balanced, labels={labs}, requested={num}, per_label={per}, remainder={rem}, base_N={N}")
        log_event("balanced_setup", labels=labs, per_label=per, remainder=rem)

        # per-label picks
        for lab in labs:
            cands = [i for i in remaining if lab in label_sets[i]]
            picked = _sample_without_replacement(rng, cands, per)
            log_event("candidates_label", label=lab, candidates=len(cands), requested=per, picked=len(picked))
            take(picked, f"pick_label_{lab}")
            if len(picked) < per:
                print(f"  [log] label='{lab}' insufficient: candidates={len(cands)}, requested={per}, picked={len(picked)}")

        # remainder picks from any of labels
        if rem > 0:
            labset = set(labs)
            cands = [i for i in remaining if len(label_sets[i].intersection(labset)) > 0]
            picked = _sample_without_replacement(rng, cands, rem)
            log_event("candidates_remainder", candidates=len(cands), requested=rem, picked=len(picked))
            take(picked, "pick_remainder")
            print(f"  [log] remainder picked={len(picked)}/{rem} from candidates={len(cands)}")

    else:
        raise ValueError("mode must be one of: 'random', 'all', 'balanced'")

    # -------- fill if insufficient (no-dup) --------
    if len(selected) < num:
        need = num - len(selected)
        filler = _sample_without_replacement(rng, remaining, need)
        take(filler, "fill_random_no_dup")
        print(f"[append-ds] fill_random (no-dup): need={need}, filled={len(filler)}, remaining_after={len(remaining)}")

    # -------- still short: allow duplicates? --------
    if len(selected) < num:
        need = num - len(selected)
        if not allow_duplicates_when_insufficient:
            log_event("warning_still_short", need=need, final=len(selected))
            print(
                f"[append-ds][WARNING] still short: requested={num}, got={len(selected)}; "
                f"allow_duplicates_when_insufficient=False so we stop."
            )
        else:
            dup_fill = [rng.randrange(N) for _ in range(need)]
            selected.extend(dup_fill)
            log_event("fill_with_duplicates", need=need)
            print(
                f"[append-ds][WARNING] base insufficient; filled remaining {need} WITH DUPLICATES. "
                f"final_selected={len(selected)} (requested={num})"
            )

    # -------- save selected subset indices (NEW) --------
    if subset_save_path is not None:
        if subset_save_format == "csv":
            _save_subset_indices_csv(subset_save_path, selected)
            print(f"[append-ds] subset indices saved (csv): {subset_save_path}")
        else:
            payload = dict(meta)
            payload["selected_indices"] = [int(i) for i in selected]
            _save_subset_indices_json(subset_save_path, payload)
            print(f"[append-ds] subset indices saved (json): {subset_save_path}")

    # -------- build appended rows --------
    appended_rows = [base_rows[i].copy() for i in selected]

    base_out_rows = [r.copy() for r in base_rows]
    start_id = len(base_out_rows)

    if "id" in fieldnames:
        for idx, r in enumerate(base_out_rows):
            r["id"] = str(idx)
        for j, r in enumerate(appended_rows):
            r["id"] = str(start_id + j)

    out_rows = base_out_rows + appended_rows

    print(f"[append-ds] output: base={len(base_out_rows)} + appended={len(appended_rows)} => total={len(out_rows)}")
    _write_csv_rows(out_csv_path, out_rows, fieldnames)
    print(f"[append-ds] saved to: {out_csv_path}")

    return out_csv_path



if __name__ == "__main__":
    # Example usage of generate_appended_dataset_from_base
    base_csv = "generated_database/49_100000.csv"  # Make sure this exists from previous generation
    out_csv = "generated_database/49_110000_rb.csv"
    subset_save = "generated_database/subsets/plus10000_49_110000_rb.csv"
    
    generate_appended_dataset_from_base(
    base_csv_path=base_csv,
    out_csv_path=out_csv,
    num=10000,
    mode="balanced",
    balanced_labels=["background_color_blue", "background_color_red"],
    seed=123,
    subset_save_path= subset_save,
    subset_save_format="csv",
    )



#if __name__ == '__main__':
#    # Configuration
#    num_samples = 10
#    seed = 4342
    
#    # Generate dataset
#    dataset = generate_dataset(num_samples=num_samples, seed=seed)
    
#    # Print information about the first few samples
#    for i, sample in enumerate(dataset[:3]):
#        print(f"\nSample {i}:")
#        print(f"  Labels: {sample['labels']}")
#        print(f"  Grid shape: {sample['grid'].shape}")
        
#        visualize_sample(sample)
    
#    # Save dataset to generated_database folder as CSV with format: seed_num_samples.csv
#    save_dataset(dataset, output_dir='generated_database', seed=seed, num_samples=num_samples)
