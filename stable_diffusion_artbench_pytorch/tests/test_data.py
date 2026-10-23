import pandas as pd

from sd_artbench.config import ArtBenchConfig, label_to_prompt
from sd_artbench.data import make_artbench2_splits, make_lds_subsets, official_lds_seed


def synthetic_metadata() -> pd.DataFrame:
    rows = []
    for label in ("post_impressionism", "ukiyo_e"):
        for split, count in (("train", 10), ("test", 4)):
            for index in range(count):
                rows.append(
                    {"name": f"{label}-{split}-{index}.jpg", "label": label, "split": split}
                )
    return pd.DataFrame(rows)


def test_prompt_normalization():
    assert label_to_prompt("post_impressionism") == "a post impressionism painting"
    assert label_to_prompt("ukiyo_e") == "a ukiyo e painting"


def test_stratified_target_and_lds_splits():
    config = ArtBenchConfig(
        train_size=10,
        validation_size=4,
        lds_subset_count=3,
        lds_subset_fraction=0.5,
    )
    train, validation = make_artbench2_splits(synthetic_metadata(), config)
    assert train["label"].value_counts().to_dict() == {
        "post_impressionism": 5,
        "ukiyo_e": 5,
    }
    assert validation["label"].value_counts().to_dict() == {
        "post_impressionism": 2,
        "ukiyo_e": 2,
    }
    subsets = list(make_lds_subsets(train, config))
    assert [seed for _, seed, _ in subsets] == [42, 44, 46]
    assert all(len(subset) == 5 for _, _, subset in subsets)


def test_official_notebook_seed_progression():
    assert official_lds_seed(42, 0) == 42
    assert official_lds_seed(42, 63) == 168

