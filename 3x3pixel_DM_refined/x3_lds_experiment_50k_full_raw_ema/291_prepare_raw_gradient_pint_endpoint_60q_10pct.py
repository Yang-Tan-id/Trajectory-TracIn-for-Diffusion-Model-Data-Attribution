"""Prepare three raw-gradient 10% Top-K removal arms for q00-q59."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from endpoint_tracin_adamw_mc10_config import eta_raw_gradient_method
from exp_config import ROOT
from tracin_das_norm_adam_config import tdna_method


def load_base():
    path = Path(__file__).with_name("287_prepare_topk5_random_60q_10pct.py")
    spec = spec_from_file_location("topk_prepare_base", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_base()
QUERY_IDS = tuple(range(60))
OUT_ROOT = ROOT / "topk_removal_raw_gradient_pint_both_endpoint_mc1_60q_10pct"
METHODS = (
    {
        "tag": "pint_both_l2_raw_gradient",
        "label": "RAW-GRADIENT PINT / BOTH-L2",
        "method": tdna_method("gradient", "query_train_l2", "timestamp_sum_squared"),
        "prediction_sign": -1,
        "description": (
            "Raw-gradient PINT, query/train L2, timestamp-sum-squared"
        ),
    },
    {
        "tag": "endpoint_mc1_raw_gradient_raw",
        "label": "RAW-GRADIENT ENDPOINT MC1 / RAW",
        "method": eta_raw_gradient_method("raw", "linear", query_mc=1, train_mc=1),
        "prediction_sign": -1,
        "description": (
            "Raw-gradient endpoint-loss TracIn, query100x1/train100x1, raw, linear"
        ),
    },
    {
        "tag": "endpoint_mc1_raw_gradient_both_l2",
        "label": "RAW-GRADIENT ENDPOINT MC1 / BOTH-L2",
        "method": eta_raw_gradient_method(
            "query_train_l2", "linear", query_mc=1, train_mc=1
        ),
        "prediction_sign": -1,
        "description": (
            "Raw-gradient endpoint-loss TracIn, query100x1/train100x1, "
            "query/train L2, linear"
        ),
    },
)
EXPECTED_JOB_COUNT = len(QUERY_IDS) * len(METHODS)


def configure_base():
    BASE.QUERY_IDS = QUERY_IDS
    BASE.OUT_ROOT = OUT_ROOT
    BASE.METHODS = METHODS
    BASE.EXPECTED_JOB_COUNT = EXPECTED_JOB_COUNT


def main():
    configure_base()
    BASE.main()


if __name__ == "__main__":
    main()
