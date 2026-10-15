"""Cross-query validation of timestamp-spot selection for x3 Top-K removal.

Timestamp spots are selected only from training queries.  Their averaged
predicted-noise MSE is then evaluated out-of-fold on unseen queries.  The
strict ``other_methods`` mode also excludes the held-out attribution method
from timestamp selection.
"""

import argparse
import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr, wilcoxon


SCOPES = ("all", "prompted", "unprompted")
TARGETS = ("endpoint_mse", "trajectory_mse")
MODES = ("same_method", "other_methods")
SPOT_COUNTS = (1, 5, 10, 20)


def load_prepare_module():
    path = Path(__file__).with_name("117_prepare_retrac_tracindas_topk_100q.py")
    spec = spec_from_file_location("retrac_topk_prepare", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def setting_label(method, fraction_tag):
    return f"{method}/{fraction_tag}"


def spearman(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 3 or np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def spearman_curve(noise, target):
    """Vectorized Spearman correlation for every timestamp column."""
    noise = np.asarray(noise, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    noise_rank = rankdata(noise, axis=0)
    target_rank = rankdata(target)
    noise_rank -= noise_rank.mean(axis=0, keepdims=True)
    target_rank -= target_rank.mean()
    numerator = np.sum(noise_rank * target_rank[:, None], axis=0)
    denominator = np.sqrt(
        np.sum(noise_rank ** 2, axis=0) * np.sum(target_rank ** 2)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.full(noise.shape[1], np.nan, dtype=np.float64),
        where=denominator > 0.0,
    )


def make_folds(query_ids, families, fold_count, seed):
    """Family-stratified folds; within-family scopes naturally use one stratum."""
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(fold_count)]
    query_ids = np.asarray(query_ids, dtype=np.int64)
    families = np.asarray(families)
    for family in sorted(set(families.tolist())):
        values = query_ids[families == family].copy()
        rng.shuffle(values)
        for index, query_id in enumerate(values.tolist()):
            folds[index % fold_count].append(int(query_id))
    return [sorted(fold) for fold in folds]


def load_data(prepare):
    with open(prepare.OUT_ROOT / "jobs.json") as handle:
        jobs = json.load(handle)
    with open(prepare.QUERY_DIR / "manifest.json") as handle:
        manifest = json.load(handle)
    query_by_id = {int(row["query_id"]): row for row in manifest}
    data = {}
    reference_timesteps = None
    for job in jobs:
        job_dir = Path(job["job_dir"])
        noise_path = job_dir / "predicted_noise_mse_per_snapshot.npy"
        evaluation_path = job_dir / "evaluation.json"
        if not noise_path.is_file() or not evaluation_path.is_file():
            raise FileNotFoundError(
                f"missing evaluation arrays in {job_dir}; run eval-only first"
            )
        with open(evaluation_path) as handle:
            evaluation = json.load(handle)
        query_id = int(job["query_id"])
        query = query_by_id[query_id]
        timesteps = np.asarray(
            np.load(Path(query["dir"]) / "trajectory_t.npy"), dtype=np.int64
        ).reshape(-1)
        noise = np.asarray(np.load(noise_path), dtype=np.float64).reshape(-1)
        if noise.shape != timesteps.shape:
            raise ValueError(f"noise/timestep mismatch in {job_dir}")
        if reference_timesteps is None:
            reference_timesteps = timesteps
        elif not np.array_equal(reference_timesteps, timesteps):
            raise ValueError(f"timestamp grid mismatch at q{query_id:02d}")
        label = setting_label(job["method_tag"], job["removal_fraction_tag"])
        data.setdefault(label, {})[query_id] = {
            "family": evaluation["family"],
            "noise": noise,
            **{target: float(evaluation[target]) for target in TARGETS},
        }
    return data, reference_timesteps


def scope_query_ids(data, scope):
    first = next(iter(data.values()))
    ids = sorted(first)
    if scope == "all":
        return ids
    return [query_id for query_id in ids if first[query_id]["family"] == scope]


def arrays_for(data, label, query_ids, target):
    return (
        np.stack([data[label][query_id]["noise"] for query_id in query_ids]),
        np.asarray(
            [data[label][query_id][target] for query_id in query_ids],
            dtype=np.float64,
        ),
    )


def selector_curve(data, labels, train_ids, target):
    values = []
    for label in labels:
        noise, outcome = arrays_for(data, label, train_ids, target)
        values.append(spearman_curve(noise, outcome))
    return np.nanmean(np.stack(values, axis=0), axis=0)


def cross_validated_setting(
    data,
    labels,
    heldout,
    folds,
    target,
    mode,
    spot_counts,
):
    query_ids = sorted(query_id for fold in folds for query_id in fold)
    index_by_id = {query_id: index for index, query_id in enumerate(query_ids)}
    predictions = {
        count: np.full(len(query_ids), np.nan, dtype=np.float64)
        for count in spot_counts
    }
    selected_positions = {count: [] for count in spot_counts}
    outcomes = np.asarray(
        [data[heldout][query_id][target] for query_id in query_ids],
        dtype=np.float64,
    )
    full_scores = np.asarray(
        [data[heldout][query_id]["noise"].mean() for query_id in query_ids],
        dtype=np.float64,
    )
    heldout_method = heldout.split("/", 1)[0]

    for test_ids in folds:
        test_set = set(test_ids)
        train_ids = [query_id for query_id in query_ids if query_id not in test_set]
        if mode == "same_method":
            selector_labels = [heldout]
        else:
            selector_labels = [
                label
                for label in labels
                if label.split("/", 1)[0] != heldout_method
            ]
        curve = selector_curve(data, selector_labels, train_ids, target)
        for count in spot_counts:
            positions = np.argsort(-curve, kind="stable")[:count]
            selected_positions[count].append(positions.tolist())
            for query_id in test_ids:
                predictions[count][index_by_id[query_id]] = float(
                    data[heldout][query_id]["noise"][positions].mean()
                )

    return {
        "full_100_timestamp_spearman": spearman(full_scores, outcomes),
        "spot_counts": {
            str(count): {
                "spearman": spearman(predictions[count], outcomes),
                "fold_selected_positions": selected_positions[count],
            }
            for count in spot_counts
        },
    }


def summarize(values_by_seed, labels, spot_counts):
    summary = {}
    for count in spot_counts:
        per_seed_mean = []
        per_setting_values = {label: [] for label in labels}
        full_by_setting = {}
        for seed_result in values_by_seed:
            selected = []
            for label in labels:
                entry = seed_result[label]
                value = entry["spot_counts"][str(count)]["spearman"]
                selected.append(value)
                per_setting_values[label].append(value)
                full_by_setting[label] = entry["full_100_timestamp_spearman"]
            per_seed_mean.append(float(np.nanmean(selected)))
        setting_selected = np.asarray(
            [np.nanmean(per_setting_values[label]) for label in labels],
            dtype=np.float64,
        )
        setting_full = np.asarray(
            [full_by_setting[label] for label in labels], dtype=np.float64
        )
        differences = setting_selected - setting_full
        valid = np.isfinite(differences) & (differences != 0.0)
        p_value = (
            float(wilcoxon(differences[valid], alternative="greater").pvalue)
            if np.any(valid)
            else float("nan")
        )
        seed_deltas = np.asarray(per_seed_mean) - float(np.nanmean(setting_full))
        summary[str(count)] = {
            "selected_spearman_mean": float(np.nanmean(per_seed_mean)),
            "full_100_spearman_mean": float(np.nanmean(setting_full)),
            "delta_mean": float(np.nanmean(seed_deltas)),
            "delta_split_95pct_interval": [
                float(np.nanpercentile(seed_deltas, 2.5)),
                float(np.nanpercentile(seed_deltas, 97.5)),
            ],
            "setting_wins": int(np.sum(differences > 0.0)),
            "setting_count": int(len(labels)),
            "wilcoxon_across_settings_p_greater": p_value,
            "per_setting_selected_spearman": {
                label: float(value) for label, value in zip(labels, setting_selected)
            },
            "per_setting_full_spearman": {
                label: float(value) for label, value in zip(labels, setting_full)
            },
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261014)
    args = parser.parse_args()
    if args.folds < 2 or args.repeats < 1:
        parser.error("--folds must be >=2 and --repeats must be >=1")

    prepare = load_prepare_module()
    data, timesteps = load_data(prepare)
    labels = sorted(data)
    spot_counts = tuple(count for count in SPOT_COUNTS if count <= len(timesteps))
    result = {
        "protocol": (
            "Repeated family-stratified K-fold cross-validation. Timestamp spots "
            "are selected on train queries only; reported Spearman is computed "
            "from out-of-fold predictions on unseen queries."
        ),
        "folds": args.folds,
        "repeats": args.repeats,
        "seed": args.seed,
        "timesteps": timesteps.astype(int).tolist(),
        "results": {},
    }
    lines = [
        "X3 TIMESTAMP-SPOT CROSS-QUERY VALIDATION",
        f"{args.repeats} repeated {args.folds}-fold CV; out-of-fold query predictions",
        "same_method = cross-query only; other_methods = cross-query + cross-method",
        "p is one-sided Wilcoxon over 9 method/fraction settings after averaging splits",
        "",
    ]

    first_setting = next(iter(data.values()))
    for scope in SCOPES:
        query_ids = scope_query_ids(data, scope)
        families = [first_setting[query_id]["family"] for query_id in query_ids]
        if len(query_ids) < args.folds * 2:
            raise ValueError(f"scope {scope} has only {len(query_ids)} queries")
        result["results"][scope] = {}
        lines.append(f"SCOPE: {scope} (n={len(query_ids)})")
        for target in TARGETS:
            result["results"][scope][target] = {}
            for mode in MODES:
                seed_results = []
                for repeat in range(args.repeats):
                    folds = make_folds(
                        query_ids,
                        families,
                        args.folds,
                        args.seed + repeat,
                    )
                    seed_results.append(
                        {
                            label: cross_validated_setting(
                                data,
                                labels,
                                label,
                                folds,
                                target,
                                mode,
                                spot_counts,
                            )
                            for label in labels
                        }
                    )
                summary = summarize(seed_results, labels, spot_counts)
                result["results"][scope][target][mode] = {
                    "summary": summary,
                    "repeated_cv": seed_results,
                }
                lines.append(f"  {target} / {mode}")
                lines.append(
                    "    spots  selected-rho  full100-rho    delta       split95%CI       wins     p"
                )
                for count in spot_counts:
                    entry = summary[str(count)]
                    interval = entry["delta_split_95pct_interval"]
                    lines.append(
                        f"    {count:5d}  {entry['selected_spearman_mean']:+.4f}       "
                        f"{entry['full_100_spearman_mean']:+.4f}    "
                        f"{entry['delta_mean']:+.4f}  "
                        f"[{interval[0]:+.4f},{interval[1]:+.4f}]  "
                        f"{entry['setting_wins']}/{entry['setting_count']}  "
                        f"{entry['wilcoxon_across_settings_p_greater']:.4g}"
                    )
        lines.append("")

    output_root = prepare.OUT_ROOT / "timestamp_spot_query_transfer"
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "timestamp_spot_query_transfer.json"
    text_path = output_root / "timestamp_spot_query_transfer.txt"
    with open(json_path, "w") as handle:
        json.dump(result, handle, indent=2)
    with open(text_path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print(f"[saved] {json_path}", flush=True)
    print(f"[saved] {text_path}", flush=True)


if __name__ == "__main__":
    main()
