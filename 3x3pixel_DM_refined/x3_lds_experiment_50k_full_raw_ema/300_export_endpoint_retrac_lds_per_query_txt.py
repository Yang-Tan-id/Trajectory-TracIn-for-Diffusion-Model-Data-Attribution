"""Export per-query LDS values for PINT, Endpoint TracIn, and ReTrac to text.

The exporter consumes the existing q00-q99 LDS JSON summaries; it does not
recompute attribution scores. Endpoint TracIn is exported for RAW and Both-L2
using the linear contraction and both parameter transforms. ReTrac is exported
for its native Both-L2 normalization and both parameter transforms. There is
no ReTrac/RAW artifact in the existing 100-timestamp x MC10 experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path("x3_lds_exp_50k")
LDS_DIR = ROOT / "lds"

TARGETS = (
    "simple_loss_ema",
    "simple_loss_raw",
    "traj_ref_ema",
    "traj_ref_raw",
    "endpoint_deviation_ema",
    "endpoint_deviation_raw",
    "trajectory_state_mse_ema",
    "trajectory_state_mse_raw",
)

ENDPOINT_VARIANTS = (
    ("raw", "RAW"),
    ("query_train_l2", "BOTH-L2"),
)

MANIFEST = """\
3x3 LDS PER-QUERY EXPORT

All sections use 100 query timestamps x MC10, q00-q99, and shared
CountSketch4096 projections.

Included:
  PINT            | raw-gradient | RAW
  PINT            | raw-gradient | BOTH-L2
  PINT            | Full-AdamW   | RAW
  PINT            | Full-AdamW   | BOTH-L2
  Endpoint TracIn | raw-gradient | RAW
  Endpoint TracIn | raw-gradient | BOTH-L2
  Endpoint TracIn | Full-AdamW   | RAW
  Endpoint TracIn | Full-AdamW   | BOTH-L2
  ReTrac          | raw-gradient | BOTH-L2 (native ReTrac normalization)
  ReTrac          | Full-AdamW   | BOTH-L2 (native ReTrac normalization)

Not included:
  ReTrac | RAW is unavailable: the existing ReTrac experiment normalizes both
  query and training features. The unnormalized artifact is Diffusion-TracIn,
  not ReTrac, so it is intentionally not relabeled here.
"""


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"missing LDS summary: {path}\n"
            "Run the corresponding merge/evaluation script first."
        )
    with path.open() as handle:
        return json.load(handle)


def format_table(
    title: str,
    artifact: str,
    sign_name: str,
    query_ids: list[int],
    target_results: dict,
) -> str:
    sign_label = "-1" if sign_name == "negative" else "+1"
    lines = [
        "=" * 148,
        title,
        f"artifact: {artifact}",
        f"reported LDS sign: {sign_label}",
        f"queries: q{min(query_ids):02d}-q{max(query_ids):02d} (n={len(query_ids)})",
        "=" * 148,
        "query " + " ".join(f"{target:>27s}" for target in TARGETS),
    ]
    for position, query_id in enumerate(query_ids):
        values = []
        for target in TARGETS:
            values.append(
                float(target_results[target][sign_name]["per_query"][position])
            )
        lines.append(
            f"q{query_id:02d}  " + " ".join(f"{value:+27.8f}" for value in values)
        )
    lines.extend(
        [
            "-" * 148,
            "mean   "
            + " ".join(
                f"{float(target_results[target][sign_name]['mean']):+27.8f}"
                for target in TARGETS
            ),
            "std    "
            + " ".join(
                f"{float(target_results[target][sign_name]['std']):27.8f}"
                for target in TARGETS
            ),
            "",
        ]
    )
    return "\n".join(lines)


def endpoint_sections(path: Path, transform_label: str, sign_name: str) -> list[str]:
    data = load_json(path)
    query_ids = [int(value) for value in data["query_ids"]]
    sections = []
    for variant, variant_label in ENDPOINT_VARIANTS:
        result = data["variants"][variant]["linear"]
        sections.append(
            format_table(
                title=(
                    f"ENDPOINT TRACIN 100 timestamps x MC10 | {transform_label} | "
                    f"{variant_label} | LINEAR"
                ),
                artifact=result["method"],
                sign_name=sign_name,
                query_ids=query_ids,
                target_results=result["targets"],
            )
        )
    return sections


def pint_sections(path: Path, transform: str, transform_label: str) -> list[str]:
    data = load_json(path)
    query_ids = [int(value) for value in data["query_ids"]]
    sections = []
    for variant, variant_label in ENDPOINT_VARIANTS:
        result = data["variants"][transform][variant]["timestamp_sum_squared"]
        sections.append(
            format_table(
                title=(
                    f"PINT / TRACIN-DAS 100 timestamps | {transform_label} | "
                    f"{variant_label} | TIMESTAMP-SUM-SQUARED"
                ),
                artifact=result["method"],
                sign_name="negative",
                query_ids=query_ids,
                target_results=result["targets"],
            )
        )
    return sections


def retrac_section(data: dict, key: str, title: str, sign_name: str) -> str:
    query_ids = [int(value) for value in data["query_ids"]]
    result = data["methods"][key]
    return format_table(
        title=title,
        artifact=result["artifact"],
        sign_name=sign_name,
        query_ids=query_ids,
        target_results=result["targets"],
    )


def write_text(path: Path, sections: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sections).rstrip() + "\n")
    print(f"[saved] {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=LDS_DIR / "txt_exports",
    )
    args = parser.parse_args()

    endpoint_raw_path = (
        LDS_DIR
        / "endpoint_tracin_raw_gradient_100t_querymc10_trainmc10_norm4_q00_q99.json"
    )
    endpoint_adamw_path = (
        LDS_DIR / "endpoint_tracin_adamw_full_100t_mc10_norm4_q00_q99.json"
    )
    retrac_path = LDS_DIR / "diffusion_tracin_vs_retrac_adamw_full_replayed_100q.json"
    pint_path = LDS_DIR / "tracin_das_norm4_gradient_and_adamw_full_q00_q99.json"

    pint_raw = pint_sections(pint_path, "gradient", "RAW-GRADIENT")
    pint_adamw = pint_sections(pint_path, "adamw_full", "FULL-ADAMW")

    # Raw-gradient scores use the negative LDS sign; full-AdamW scores use the
    # positive LDS sign.  These are the same directions used by corrected Top-K.
    endpoint_raw = endpoint_sections(endpoint_raw_path, "RAW-GRADIENT", "negative")
    endpoint_adamw = endpoint_sections(endpoint_adamw_path, "FULL-ADAMW", "positive")

    retrac_data = load_json(retrac_path)
    retrac_raw = retrac_section(
        retrac_data,
        "diffusion_retrac",
        "RETRAC 100 timestamps x MC10 | RAW-GRADIENT | BOTH-L2 | LINEAR",
        "negative",
    )
    retrac_adamw = retrac_section(
        retrac_data,
        "diffusion_retrac_adamw_full",
        "RETRAC 100 timestamps x MC10 | FULL-ADAMW | BOTH-L2 | LINEAR",
        "positive",
    )

    files = {
        "pint_mc1_raw_gradient_per_query.txt": pint_raw,
        "pint_mc1_adamw_full_per_query.txt": pint_adamw,
        "endpoint_tracin_mc10_raw_gradient_per_query.txt": endpoint_raw,
        "endpoint_tracin_mc10_adamw_full_per_query.txt": endpoint_adamw,
        "retrac_mc10_raw_gradient_per_query.txt": [retrac_raw],
        "retrac_mc10_adamw_full_per_query.txt": [retrac_adamw],
    }
    for name, sections in files.items():
        write_text(args.output_dir / name, sections)

    combined = (
        [MANIFEST]
        + pint_raw
        + pint_adamw
        + endpoint_raw
        + endpoint_adamw
        + [retrac_raw, retrac_adamw]
    )
    write_text(args.output_dir / "pint_endpoint_retrac_all_per_query.txt", combined)


if __name__ == "__main__":
    main()
