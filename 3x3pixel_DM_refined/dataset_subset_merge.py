import csv
import os
import random
from typing import Dict, List, Tuple


# ----------------------------
# Basic CSV helpers
# ----------------------------
def read_rows(csv_path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        fieldnames = r.fieldnames
        if not fieldnames:
            raise ValueError(f"Missing header in {csv_path}")
        rows = list(r)
    return rows, fieldnames


def write_rows(csv_path: str, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_idx_csv(csv_path: str, idxs: List[int], col: str = "idx") -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([col])
        for i in idxs:
            w.writerow([int(i)])


# ----------------------------
# Label logic
# ----------------------------
def get_row_labels(row: Dict[str, str], label_cols=("label_1", "label_2", "label_3")) -> List[str]:
    labs = []
    for c in label_cols:
        v = row.get(c, "")
        v = "" if v is None else str(v).strip()
        if v:
            labs.append(v)
    return labs


def has_label(row: Dict[str, str], label: str, label_cols=("label_1", "label_2", "label_3")) -> bool:
    target = str(label).strip()
    return target in set(get_row_labels(row, label_cols=label_cols))


# ----------------------------
# Core selection + splitting
# ----------------------------
def pick_k_indices_with_label(
    rows: List[Dict[str, str]],
    label: str,
    k: int,
    rng: random.Random,
    label_cols=("label_1", "label_2", "label_3"),
) -> List[int]:
    cands = [i for i, row in enumerate(rows) if has_label(row, label, label_cols=label_cols)]
    if len(cands) < k:
        raise ValueError(f"Not enough candidates for label={label}. Need {k}, found {len(cands)}.")
    return rng.sample(cands, k)


def split_half(idxs: List[int], rng: random.Random) -> Tuple[List[int], List[int]]:
    idxs = list(idxs)
    rng.shuffle(idxs)
    mid = len(idxs) // 2
    return idxs[:mid], idxs[mid:]


def build_appended_dataset(base_rows: List[Dict[str, str]], append_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [r.copy() for r in base_rows] + [r.copy() for r in append_rows]


def renumber_id_if_present(rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    if "id" in fieldnames:
        for new_id, r in enumerate(rows):
            r["id"] = str(new_id)

# ----------------------------
# Main pipeline you described
# ----------------------------
def make_six_datasets(
    base_csv_path: str,
    out_dir: str,
    seed: int = 0,
    n_per_color: int = 10_000,
    label_blue: str = "background_color_blue",
    label_yellow: str = "background_color_yellow",
    label_red: str = "background_color_red",
    label_cols=("label_1", "label_2", "label_3"),
) -> None:
    """
    (same docstring as before)
    """
    rng = random.Random(seed)

    base_rows, fieldnames = read_rows(base_csv_path)
    N = len(base_rows)
    print(f"[load] base={base_csv_path} rows={N}")

    # 1) pick indices
    idx_blue = pick_k_indices_with_label(base_rows, label_blue, n_per_color, rng, label_cols=label_cols)
    idx_yellow = pick_k_indices_with_label(base_rows, label_yellow, n_per_color, rng, label_cols=label_cols)
    idx_red = pick_k_indices_with_label(base_rows, label_red, n_per_color, rng, label_cols=label_cols)

    print(f"[pick] blue={len(idx_blue)} yellow={len(idx_yellow)} red={len(idx_red)} (each from base indices)")

    # Convert to rows
    rows_blue = [base_rows[i] for i in idx_blue]
    rows_yellow = [base_rows[i] for i in idx_yellow]
    rows_red = [base_rows[i] for i in idx_red]

    # 2) three appended datasets
    d1 = build_appended_dataset(base_rows, rows_blue)     # base + B
    d2 = build_appended_dataset(base_rows, rows_yellow)   # base + Y
    d3 = build_appended_dataset(base_rows, rows_red)      # base + R

    # 3) split into A / B halves
    idx_blue_A, idx_blue_B = split_half(idx_blue, rng)
    idx_yellow_A, idx_yellow_B = split_half(idx_yellow, rng)
    idx_red_A, idx_red_B = split_half(idx_red, rng)

    rows_blue_A = [base_rows[i] for i in idx_blue_A]
    rows_blue_B = [base_rows[i] for i in idx_blue_B]
    rows_yellow_A = [base_rows[i] for i in idx_yellow_A]
    rows_yellow_B = [base_rows[i] for i in idx_yellow_B]
    rows_red_A = [base_rows[i] for i in idx_red_A]
    rows_red_B = [base_rows[i] for i in idx_red_B]

    print(f"[split] blue_A={len(idx_blue_A)} blue_B={len(idx_blue_B)}")
    print(f"[split] yellow_A={len(idx_yellow_A)} yellow_B={len(idx_yellow_B)}")
    print(f"[split] red_A={len(idx_red_A)} red_B={len(idx_red_B)}")

    # 4) mixed appended datasets
    d4 = build_appended_dataset(base_rows, rows_blue_A + rows_yellow_B)  # base + blue_A + yellow_B
    d5 = build_appended_dataset(base_rows, rows_yellow_A + rows_red_B)   # base + yellow_A + red_B
    d6 = build_appended_dataset(base_rows, rows_red_A + rows_blue_B)     # base + red_A + blue_B

    # 5) Save everything
    os.makedirs(out_dir, exist_ok=True)

    # NEW: subsets go into out_dir/subset/
    subset_dir = os.path.join(out_dir, "subset")
    os.makedirs(subset_dir, exist_ok=True)

    # Save subsets (as CSV) and subset indices (as idx CSV)
    subset_map = {
        "blue_A": (rows_blue_A, idx_blue_A),
        "blue_B": (rows_blue_B, idx_blue_B),
        "yellow_A": (rows_yellow_A, idx_yellow_A),
        "yellow_B": (rows_yellow_B, idx_yellow_B),
        "red_A": (rows_red_A, idx_red_A),
        "red_B": (rows_red_B, idx_red_B),
    }
    for name, (srows, sidx) in subset_map.items():
        subset_csv = os.path.join(subset_dir, f"{name}.csv")
        subset_idx_csv = os.path.join(subset_dir, f"{name}_idx.csv")
        write_rows(subset_csv, [r.copy() for r in srows], fieldnames)
        write_idx_csv(subset_idx_csv, sidx, col="idx")
        print(f"[save] subset={name}: rows={len(srows)} -> {subset_csv} | idx -> {subset_idx_csv}")

    # Save 6 datasets + the appended-index lists used (still in out_dir/)
    dataset_map = {
        "dataset_1_base_plus_blue": (d1, idx_blue),
        "dataset_2_base_plus_yellow": (d2, idx_yellow),
        "dataset_3_base_plus_red": (d3, idx_red),
        "dataset_4_base_plus_blueA_yellowB": (d4, idx_blue_A + idx_yellow_B),
        "dataset_5_base_plus_yellowA_redB": (d5, idx_yellow_A + idx_red_B),
        "dataset_6_base_plus_redA_blueB": (d6, idx_red_A + idx_blue_B),
    }

    for name, (drows, used_idxs) in dataset_map.items():
        out_csv = os.path.join(out_dir, f"{name}.csv")
        out_used_idx_csv = os.path.join(out_dir, f"{name}_appended_idx.csv")

        drows_to_save = [r.copy() for r in drows]
        renumber_id_if_present(drows_to_save, fieldnames)

        write_rows(out_csv, drows_to_save, fieldnames)
        write_idx_csv(out_used_idx_csv, used_idxs, col="idx")

        print(f"[save] {name}: out_rows={len(drows_to_save)} -> {out_csv}")
        print(f"       appended idx count={len(used_idxs)} -> {out_used_idx_csv}")


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    make_six_datasets(
        base_csv_path="generated_database/49_100000.csv",
        out_dir="generated_database/RBY",
        seed=123,
        n_per_color=9900,
        label_blue="background_color_blue",
        label_yellow="background_color_yellow",
        label_red="background_color_red",
    )
