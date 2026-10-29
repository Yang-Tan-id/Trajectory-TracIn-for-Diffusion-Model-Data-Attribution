"""Prepare raw-gradient Endpoint-TracIn MC10 Both-L2 10% removals."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from endpoint_tracin_adamw_mc10_config import eta_raw_gradient_method
from exp_config import ROOT


def load_base():
    path = Path(__file__).with_name("287_prepare_topk5_random_60q_10pct.py")
    spec = spec_from_file_location("topk_prepare_base", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_base()
QUERY_IDS = tuple(range(60))
OUT_ROOT = ROOT / "topk_removal_endpoint_raw_gradient_mc10_both_l2_60q_10pct"
METHODS = (
    {
        "tag": "endpoint_mc10_raw_gradient_both_l2",
        "label": "RAW-GRADIENT ENDPOINT MC10 / BOTH-L2",
        "method": eta_raw_gradient_method(
            "query_train_l2", "linear", query_mc=10, train_mc=10
        ),
        "prediction_sign": -1,
        "description": (
            "Raw-gradient endpoint-loss TracIn, query100x10/train100x10, "
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
