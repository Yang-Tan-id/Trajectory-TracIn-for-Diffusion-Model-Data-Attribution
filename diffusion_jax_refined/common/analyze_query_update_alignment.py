from __future__ import annotations

"""Summarize query-gradient alignment with actual checkpoint updates."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze query update alignment .npz.")
    parser.add_argument("--alignment-npz", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    with np.load(Path(args.alignment_npz).expanduser(), allow_pickle=False) as payload:
        alignment = np.asarray(payload["query_update_alignment_by_transition"], dtype=np.float64)
        cosine = np.asarray(payload["query_update_cosine_by_transition"], dtype=np.float64)
        qnorm = np.asarray(payload["query_grad_norm_by_transition"], dtype=np.float64)
        update_norm = np.asarray(payload["update_norm_by_transition"], dtype=np.float64)
        timesteps = np.asarray(payload["timesteps"], dtype=np.int32)
        positions = np.asarray(payload["snapshot_positions"], dtype=np.int32)

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    transition_rows = []
    for c in range(alignment.shape[0]):
        transition_rows.append(
            {
                "transition_index": c,
                "from_ckpt_index": c,
                "to_ckpt_index": c + 1,
                "from_epoch": 4 * (c + 1),
                "to_epoch": 4 * (c + 2),
                "alignment_mean": float(np.mean(alignment[c])),
                "alignment_median": float(np.median(alignment[c])),
                "alignment_positive_fraction": float(np.mean(alignment[c] > 0.0)),
                "cosine_mean": float(np.mean(cosine[c])),
                "cosine_median": float(np.median(cosine[c])),
                "cosine_positive_fraction": float(np.mean(cosine[c] > 0.0)),
                "query_grad_norm_mean": float(np.mean(qnorm[c])),
                "update_norm": float(update_norm[c]),
            }
        )
    write_csv(out_dir / "transition_alignment_summary.csv", transition_rows)

    snapshot_rows = []
    for s in range(alignment.shape[1]):
        snapshot_rows.append(
            {
                "snapshot_column": s,
                "snapshot_position": int(positions[s]) if s < len(positions) else s,
                "timestep": int(timesteps[s]) if s < len(timesteps) else -1,
                "alignment_mean": float(np.mean(alignment[:, s])),
                "alignment_median": float(np.median(alignment[:, s])),
                "alignment_positive_fraction": float(np.mean(alignment[:, s] > 0.0)),
                "cosine_mean": float(np.mean(cosine[:, s])),
                "cosine_median": float(np.median(cosine[:, s])),
                "cosine_positive_fraction": float(np.mean(cosine[:, s] > 0.0)),
                "query_grad_norm_mean": float(np.mean(qnorm[:, s])),
            }
        )
    write_csv(out_dir / "snapshot_alignment_summary.csv", snapshot_rows)

    summary = {
        "alignment_npz": str(Path(args.alignment_npz).expanduser()),
        "shape": list(alignment.shape),
        "alignment_mean": float(np.mean(alignment)),
        "alignment_median": float(np.median(alignment)),
        "alignment_positive_fraction": float(np.mean(alignment > 0.0)),
        "cosine_mean": float(np.mean(cosine)),
        "cosine_median": float(np.median(cosine)),
        "cosine_positive_fraction": float(np.mean(cosine > 0.0)),
        "num_transitions_alignment_positive_fraction_ge_0p75": int(
            sum(float(r["alignment_positive_fraction"]) >= 0.75 for r in transition_rows)
        ),
        "num_snapshots_alignment_positive_fraction_ge_0p75": int(
            sum(float(r["alignment_positive_fraction"]) >= 0.75 for r in snapshot_rows)
        ),
        "best_transitions": sorted(transition_rows, key=lambda r: float(r["alignment_positive_fraction"]), reverse=True)[:10],
        "worst_transitions": sorted(transition_rows, key=lambda r: float(r["alignment_positive_fraction"]))[:10],
        "best_snapshots": sorted(snapshot_rows, key=lambda r: float(r["alignment_positive_fraction"]), reverse=True)[:10],
        "worst_snapshots": sorted(snapshot_rows, key=lambda r: float(r["alignment_positive_fraction"]))[:10],
    }
    (out_dir / "query_update_alignment_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[saved] {out_dir}")
    print(
        f"alignment_positive_fraction={summary['alignment_positive_fraction']:.3f} | "
        f"cosine_positive_fraction={summary['cosine_positive_fraction']:.3f} | "
        f"alignment_mean={summary['alignment_mean']:.6g} | cosine_mean={summary['cosine_mean']:.6g}"
    )


if __name__ == "__main__":
    main()
