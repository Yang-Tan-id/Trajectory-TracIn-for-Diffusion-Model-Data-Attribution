"""Summarize the three raw-gradient PINT/Endpoint true-removal arms."""

import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


METRICS = (
    "noise_trajectory_mse",
    "trajectory_mse",
    "endpoint_mse",
    "simple_loss_abs_change",
)


def load_prepare():
    path = Path(__file__).with_name(
        "291_prepare_raw_gradient_pint_endpoint_60q_10pct.py"
    )
    spec = spec_from_file_location("raw_gradient_topk_prepare", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    module.configure_base()
    return module


def main():
    prepare = load_prepare()
    rows = {}
    for method in prepare.METHODS:
        tag = method["tag"]
        rows[tag] = []
        for query_id in prepare.QUERY_IDS:
            path = (
                prepare.OUT_ROOT
                / tag
                / prepare.BASE.REMOVAL_FRACTION_TAG
                / f"q{query_id:02d}"
                / "evaluation.json"
            )
            if not path.is_file():
                raise FileNotFoundError(path)
            with open(path) as handle:
                rows[tag].append(json.load(handle))

    labels = {method["tag"]: method["label"] for method in prepare.METHODS}
    tags = list(labels)
    output = {"query_ids": list(prepare.QUERY_IDS), "methods": {}, "pairwise": {}}
    lines = ["RAW-GRADIENT PINT/ENDPOINT TRUE 10% REMOVAL — Q00-Q59", ""]
    for metric in METRICS:
        lines.append(f"METRIC: {metric}")
        lines.append(f"{'method':42s} {'mean +/- std':28s}")
        for tag in tags:
            values = np.asarray([row[metric] for row in rows[tag]], dtype=np.float64)
            output["methods"].setdefault(tag, {})[metric] = {
                "mean": float(values.mean()), "std": float(values.std())
            }
            lines.append(f"{labels[tag]:42s} {values.mean():.6e} +/- {values.std():.3e}")
        lines.append("")

    for left_index, left in enumerate(tags):
        for right in tags[left_index + 1:]:
            key = f"{left}_minus_{right}"
            output["pairwise"][key] = {}
            for metric in METRICS:
                a = np.asarray([row[metric] for row in rows[left]], dtype=np.float64)
                b = np.asarray([row[metric] for row in rows[right]], dtype=np.float64)
                delta = a - b
                output["pairwise"][key][metric] = {
                    "mean_difference": float(delta.mean()),
                    "wins": int(np.sum(delta > 0)),
                    "p_wilcoxon_two_sided": float(wilcoxon(delta).pvalue),
                }

    json_path = prepare.OUT_ROOT / "summary.json"
    text_path = prepare.OUT_ROOT / "summary.txt"
    with open(json_path, "w") as handle:
        json.dump(output, handle, indent=2)
    text_path.write_text("\n".join(lines) + "\n")
    print(text_path.read_text(), flush=True)
    print(f"[saved] {json_path}", flush=True)
    print(f"[saved] {text_path}", flush=True)


if __name__ == "__main__":
    main()
