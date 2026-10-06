from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument(
        "--query-file",
        type=Path,
        default=SHAPES_ROOT / "queries_in_distribution_plus_zero_seed_100_219.json",
    )
    parser.add_argument("--query-ids", default=",".join(str(i) for i in range(100)))
    parser.add_argument("--expected-models", type=int, default=192)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    query_ids = [int(x) for x in args.query_ids.replace(",", " ").split()]
    queries = json.loads(args.query_file.read_text())["queries"]
    eval_root = SHAPES_ROOT / "result" / args.experiment / "eval" / "prompted_solo"
    out_dir = args.out_dir or (SHAPES_ROOT / "result" / args.experiment / "analysis" / "trajectory_deviation_quarters_q0_99")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int]] = []
    per_query: list[dict[str, float | int]] = []
    for qid in query_ids:
        seed = int(queries[qid].get("initial_seed", queries[qid].get("seed")))
        candidates = list(eval_root.glob(f"query_*/initial_seed_{seed}/lds_target_cache/ddim_eta0/*/trajectory_state_mse"))
        if len(candidates) != 1:
            raise RuntimeError(f"Q{qid}: expected one trajectory cache directory, found {len(candidates)}")
        files = sorted(candidates[0].glob("target_*.json"))
        if len(files) != args.expected_models:
            raise RuntimeError(f"Q{qid}: expected {args.expected_models} models, found {len(files)}")

        query_rows = []
        for model_id, path in enumerate(files):
            payload = json.loads(path.read_text())
            values = np.asarray(payload["target_details"]["per_snapshot_mean"], dtype=np.float64)
            if values.ndim != 1 or values.size < 4:
                raise ValueError(f"invalid per-snapshot values in {path}: shape={values.shape}")
            blocks = np.array_split(values, 4)
            total = float(values.sum())
            for segment, block in enumerate(blocks, 1):
                row = {
                    "query": qid,
                    "model": model_id,
                    "segment": segment,
                    "start_index": int(sum(len(x) for x in blocks[: segment - 1])),
                    "end_index_exclusive": int(sum(len(x) for x in blocks[:segment])),
                    "share": float(block.sum() / total) if total > 0 else float("nan"),
                    "mean_gap": float(block.mean()),
                }
                rows.append(row)
                query_rows.append(row)

        for segment in range(1, 5):
            selected = [r for r in query_rows if r["segment"] == segment]
            shares = np.asarray([r["share"] for r in selected])
            gaps = np.asarray([r["mean_gap"] for r in selected])
            per_query.append({
                "query": qid,
                "segment": segment,
                "models": len(selected),
                "share_mean": float(np.nanmean(shares)),
                "share_std": float(np.nanstd(shares)),
                "mean_gap_mean": float(np.mean(gaps)),
                "mean_gap_std": float(np.std(gaps)),
            })

    fields = list(rows[0])
    with (out_dir / "pairwise.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    fields = list(per_query[0])
    with (out_dir / "per_query.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_query)

    print("TRAJECTORY DEVIATION BY QUARTER")
    print("SEGMENT    MEAN SHARE    SHARE STD       RAW MEAN GAP      RAW GAP STD")
    print("-----------------------------------------------------------------------")
    overall = []
    for segment in range(1, 5):
        selected = [r for r in rows if r["segment"] == segment]
        shares = np.asarray([r["share"] for r in selected])
        gaps = np.asarray([r["mean_gap"] for r in selected])
        summary = {
            "segment": segment,
            "pairs": len(selected),
            "share_mean": float(np.nanmean(shares)),
            "share_std": float(np.nanstd(shares)),
            "mean_gap_mean": float(np.mean(gaps)),
            "mean_gap_std": float(np.std(gaps)),
        }
        overall.append(summary)
        print(
            f"{segment}/4       {100*summary['share_mean']:10.3f}%"
            f"   {100*summary['share_std']:9.3f}%"
            f"   {summary['mean_gap_mean']:16.9g}"
            f"   {summary['mean_gap_std']:14.9g}"
        )
    (out_dir / "summary.json").write_text(json.dumps({
        "queries": len(query_ids),
        "models_per_query": args.expected_models,
        "pairs": len(query_ids) * args.expected_models,
        "std_definition": "population std across query-model pairs; per-query std is across models",
        "segments": overall,
    }, indent=2))
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
