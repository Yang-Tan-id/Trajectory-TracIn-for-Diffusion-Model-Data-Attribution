"""Compare existing raw-gradient and Full-AdamW PINT 10% removal runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon

from exp_config import ROOT


QUERY_IDS = tuple(range(60))
FRACTION_TAG = "remove_10pct"
OLD_ROOT = ROOT / "topk_removal_retrac_endtracin_tracindas_100q"
NEW_ROOT = (
    ROOT / "topk_removal_pint_endpoint_retrac_random_60q_10pct_correct_direction"
)
OLD_TAG = "tracin_das_aligned_timestamp_square"
NEW_TAG = "pint_raw"
METRICS = (
    "noise_trajectory_mse",
    "trajectory_mse",
    "endpoint_mse",
    "simple_loss_abs_change",
)


def job_dir(root: Path, tag: str, query_id: int) -> Path:
    return root / tag / FRACTION_TAG / f"q{query_id:02d}"


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    with open(path) as handle:
        return json.load(handle)


def saved_score_direction(job: dict) -> str:
    explicit = job.get("ranking_saved_score_direction")
    if explicit is not None:
        return str(explicit)
    ranking = str(job.get("ranking", ""))
    if ranking.startswith("descending_saved_score"):
        return "descending"
    if ranking.startswith("ascending_saved_score"):
        return "ascending"
    return f"unknown:{ranking}"


def main() -> None:
    raw = {
        q: load_json(job_dir(OLD_ROOT, OLD_TAG, q) / "evaluation.json")
        for q in QUERY_IDS
    }
    adamw = {
        q: load_json(job_dir(NEW_ROOT, NEW_TAG, q) / "evaluation.json")
        for q in QUERY_IDS
    }
    raw_jobs = {
        q: load_json(job_dir(OLD_ROOT, OLD_TAG, q) / "job.json") for q in QUERY_IDS
    }
    adamw_jobs = {
        q: load_json(job_dir(NEW_ROOT, NEW_TAG, q) / "job.json") for q in QUERY_IDS
    }

    raw_rankings = sorted({str(item["ranking"]) for item in raw_jobs.values()})
    adamw_rankings = sorted({str(item["ranking"]) for item in adamw_jobs.values()})
    raw_methods = sorted({str(item["method"]) for item in raw_jobs.values()})
    adamw_methods = sorted({str(item["method"]) for item in adamw_jobs.values()})
    raw_directions = sorted({saved_score_direction(item) for item in raw_jobs.values()})
    adamw_directions = sorted(
        {saved_score_direction(item) for item in adamw_jobs.values()}
    )

    overlaps = []
    rows = []
    for query_id in QUERY_IDS:
        raw_removed = np.load(
            job_dir(OLD_ROOT, OLD_TAG, query_id) / "removed_indices.npy"
        ).astype(np.int64)
        adamw_removed = np.load(
            job_dir(NEW_ROOT, NEW_TAG, query_id) / "removed_indices.npy"
        ).astype(np.int64)
        overlap = float(
            np.intersect1d(raw_removed, adamw_removed).size / raw_removed.size
        )
        overlaps.append(overlap)
        row = {
            "query_id": query_id,
            "family": raw[query_id]["family"],
            "topk_overlap_fraction": overlap,
        }
        for metric in METRICS:
            raw_value = float(raw[query_id][metric])
            adamw_value = float(adamw[query_id][metric])
            row[f"raw_gradient_{metric}"] = raw_value
            row[f"adamw_{metric}"] = adamw_value
            row[f"adamw_minus_raw_{metric}"] = adamw_value - raw_value
        rows.append(row)

    ranking_matched = raw_directions == adamw_directions
    summary = {
        "query_ids": list(QUERY_IDS),
        "difference_definition": "Full-AdamW PINT minus old raw-gradient PINT",
        "old_method": raw_methods,
        "new_method": adamw_methods,
        "old_ranking": raw_rankings,
        "new_ranking": adamw_rankings,
        "old_saved_score_direction": raw_directions,
        "new_saved_score_direction": adamw_directions,
        "ranking_contract_matched": ranking_matched,
        "controlled_transform_only_comparison": ranking_matched,
        "warning": (
            None
            if ranking_matched
            else (
                "Ranking contracts differ, so removal-effect differences cannot be "
                "attributed solely to raw-gradient versus AdamW."
            )
        ),
        "topk_overlap_fraction": {
            "mean": float(np.mean(overlaps)),
            "std": float(np.std(overlaps)),
            "min": float(np.min(overlaps)),
            "max": float(np.max(overlaps)),
        },
        "metrics": {},
    }
    for metric in METRICS:
        raw_values = np.asarray([raw[q][metric] for q in QUERY_IDS], dtype=np.float64)
        adamw_values = np.asarray(
            [adamw[q][metric] for q in QUERY_IDS], dtype=np.float64
        )
        differences = adamw_values - raw_values
        wins = int(np.sum(differences > 0))
        losses = int(np.sum(differences < 0))
        summary["metrics"][metric] = {
            "raw_gradient_mean": float(raw_values.mean()),
            "raw_gradient_std": float(raw_values.std()),
            "adamw_mean": float(adamw_values.mean()),
            "adamw_std": float(adamw_values.std()),
            "adamw_minus_raw_mean": float(differences.mean()),
            "adamw_minus_raw_std": float(differences.std()),
            "adamw_over_raw_mean_ratio": (
                float(adamw_values.mean() / raw_values.mean())
                if raw_values.mean() != 0
                else None
            ),
            "adamw_wins": wins,
            "ties": int(np.sum(differences == 0)),
            "raw_gradient_wins": losses,
            "p_wilcoxon_two_sided": float(
                wilcoxon(differences, alternative="two-sided").pvalue
            ),
            "p_sign_two_sided": (
                float(binomtest(wins, wins + losses, 0.5).pvalue)
                if wins + losses
                else 1.0
            ),
        }

    json_path = NEW_ROOT / "existing_raw_gradient_vs_adamw_pint_topk.json"
    csv_path = NEW_ROOT / "existing_raw_gradient_vs_adamw_pint_topk_per_query.csv"
    text_path = NEW_ROOT / "existing_raw_gradient_vs_adamw_pint_topk.txt"
    with open(json_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(text_path, "w") as handle:
        handle.write("PINT TRUE 10% REMOVAL: EXISTING RAW-GRADIENT VS FULL-ADAMW\n")
        handle.write("q00-q59; delta = Full-AdamW minus raw-gradient\n\n")
        handle.write(f"old method: {raw_methods}\n")
        handle.write(f"new method: {adamw_methods}\n")
        handle.write(f"old ranking: {raw_rankings}\n")
        handle.write(f"new ranking: {adamw_rankings}\n")
        handle.write(f"old saved-score direction: {raw_directions}\n")
        handle.write(f"new saved-score direction: {adamw_directions}\n")
        handle.write(f"ranking contract matched: {ranking_matched}\n")
        if summary["warning"]:
            handle.write(f"WARNING: {summary['warning']}\n")
        handle.write(
            f"Top-K overlap: {np.mean(overlaps):.4f} +/- {np.std(overlaps):.4f}\n\n"
        )
        handle.write(
            f"{'metric':28s} {'raw-gradient':24s} {'Full-AdamW':24s} "
            f"{'delta':>12s} {'ratio':>9s} {'wins':>8s} {'p-W':>12s}\n"
        )
        for metric in METRICS:
            item = summary["metrics"][metric]
            handle.write(
                f"{metric:28s} "
                f"{item['raw_gradient_mean']:.6e}+/-{item['raw_gradient_std']:.3e} "
                f"{item['adamw_mean']:.6e}+/-{item['adamw_std']:.3e} "
                f"{item['adamw_minus_raw_mean']:+.3e} "
                f"{item['adamw_over_raw_mean_ratio']:9.4f} "
                f"{item['adamw_wins']:02d}/60 "
                f"{item['p_wilcoxon_two_sided']:.6g}\n"
            )
    print(text_path.read_text(), flush=True)
    print(f"[saved] {json_path}", flush=True)
    print(f"[saved] {csv_path}", flush=True)
    print(f"[saved] {text_path}", flush=True)


if __name__ == "__main__":
    main()
