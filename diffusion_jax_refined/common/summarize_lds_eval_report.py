from __future__ import annotations

"""Create a compact Markdown/CSV report from LDS aggregate summaries."""

import argparse
import csv
import json
from pathlib import Path


def _rows(eval_root: Path) -> list[dict]:
    rows = []
    for summary_path in sorted(eval_root.glob("**/per_seed_summary.json")):
        payload = json.loads(summary_path.read_text())
        parts = summary_path.relative_to(eval_root).parts
        if len(parts) < 8:
            continue
        mode = parts[0]
        query = parts[1]
        initial_seed = parts[2].replace("initial_seed_", "")
        eval_kind = parts[3]
        algorithm = parts[4]
        target = parts[5]
        prediction = parts[6]
        values = [
            float(row["lds_spearman"])
            for row in payload.get("per_seed", [])
            if row.get("lds_spearman") is not None
        ]
        if not values:
            continue
        rows.append(
            {
                "mode": mode,
                "query": query,
                "initial_seed": initial_seed,
                "eval_kind": eval_kind,
                "algorithm": algorithm,
                "target": target,
                "prediction": prediction,
                "num_seeds": len(values),
                "mean_lds": sum(values) / len(values),
                "min_lds": min(values),
                "max_lds": max(values),
                "aggregate_dir": str(summary_path.parent),
                "scatter_grid": str(summary_path.parent / "per_seed_scatter_grid.svg"),
                "all_points_scatter": str(summary_path.parent / "all_points_scatter.svg"),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "mode",
        "query",
        "initial_seed",
        "algorithm",
        "target",
        "prediction",
        "num_seeds",
        "mean_lds",
        "min_lds",
        "max_lds",
        "aggregate_dir",
        "scatter_grid",
        "all_points_scatter",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_md(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# LDS eval report", ""]
    if not rows:
        lines.extend(["No aggregate summaries were found.", ""])
        path.write_text("\n".join(lines))
        return
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["algorithm"]), []).append(row)
    for algorithm in sorted(grouped):
        alg_rows = grouped[algorithm]
        mean = sum(float(row["mean_lds"]) for row in alg_rows) / len(alg_rows)
        lines.extend([f"## {algorithm}", "", f"Mean across query groups: {mean:.6f}", ""])
        lines.append("| mode | query | initial seed | seeds | mean LDS | min | max | scatter |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
        for row in sorted(alg_rows, key=lambda item: (item["mode"], item["query"], item["initial_seed"])):
            lines.append(
                "| {mode} | {query} | {initial_seed} | {num_seeds} | {mean_lds:.6f} | "
                "{min_lds:.6f} | {max_lds:.6f} | {scatter_grid} |".format(**row)
            )
        lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize LDS eval aggregate folders.")
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    rows = _rows(Path(args.eval_root))
    output_dir = Path(args.output_dir)
    _write_csv(output_dir / "lds_eval_report.csv", rows)
    _write_md(output_dir / "lds_eval_report.md", rows)
    print(f"Wrote {len(rows)} LDS aggregate rows to {output_dir}")


if __name__ == "__main__":
    main()
