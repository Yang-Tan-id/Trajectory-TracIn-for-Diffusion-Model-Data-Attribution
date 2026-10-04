"""Merge prompted/unprompted checkpoint shards into q00-q99 artifacts."""

import argparse
import json

import numpy as np

from diffusion_retrac_config import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-shard-count", type=int, default=2)
    args = parser.parse_args()
    totals = {
        "diffusion_tracin": np.zeros((len(RETRAC_QUERY_IDS), N_TRAIN), dtype=np.float64),
        "diffusion_retrac": np.zeros((len(RETRAC_QUERY_IDS), N_TRAIN), dtype=np.float64),
        "diffusion_retrac_adamw_full": np.zeros(
            (len(RETRAC_QUERY_IDS), N_TRAIN), dtype=np.float64
        ),
    }
    for family in RETRAC_FAMILIES:
        covered = []
        query_ids = list(retrac_query_ids(family))
        for shard_index in range(args.checkpoint_shard_count):
            root = retrac_shard_root(
                family, shard_index, args.checkpoint_shard_count
            )
            with open(root / "done.json") as handle:
                metadata = json.load(handle)
            if metadata["query_ids"] != query_ids:
                raise ValueError(f"query IDs mismatch in {root}")
            covered.extend(int(value) for value in metadata["checkpoint_indices"])
            partial = np.load(root / "partial_scores.npz")
            totals["diffusion_tracin"][query_ids] += partial["tracin"]
            totals["diffusion_retrac"][query_ids] += partial["retrac"]
            totals["diffusion_retrac_adamw_full"][query_ids] += partial[
                "retrac_adamw_full"
            ]
        if sorted(covered) != list(range(50)):
            raise ValueError(
                f"{family} checkpoint coverage mismatch: {sorted(covered)}"
            )
    for key, method in RETRAC_METHODS.items():
        definitions = {
            "diffusion_tracin": (
                "sum_checkpoint eta * dot(mean_t(query_loss_gradient), "
                "mean_4_replayed_training_event_gradients)"
            ),
            "diffusion_retrac": (
                "sum_checkpoint eta * "
                "dot(mean_t(normalize(mean_mc(query_loss_gradient))), "
                "mean_4(normalize(replayed_training_event_gradient)))"
            ),
            "diffusion_retrac_adamw_full": (
                "sum_checkpoint eta * "
                "dot(mean_t(normalize(mean_mc(query_loss_gradient))), "
                "mean_4(normalize(saved-state full-AdamW hypothetical update "
                "from replayed training event gradient)))"
            ),
        }
        for position, query_id in enumerate(RETRAC_QUERY_IDS):
            output = ATTR_DIR / method / f"q{query_id:02d}"
            output.mkdir(parents=True, exist_ok=True)
            np.save(output / "scores.npy", totals[key][position])
            with open(output / "info.json", "w") as handle:
                json.dump(
                    {
                        "method": key,
                        "query_id": query_id,
                        "definition": definitions[key],
                        "query_timesteps": list(RETRAC_TIMESTEPS),
                        "query_mc": RETRAC_QUERY_MC,
                        "train_events_per_checkpoint": RETRAC_EVENTS_PER_CHECKPOINT,
                        "train_event_source": "exact replayed t_train and epsilon_train",
                        "checkpoint_count": 50,
                        "parameter_source": RETRAC_PARAM_SOURCE,
                        "projection_dim": RETRAC_PROJ_DIM,
                        "query_adamw_transform": False,
                        "train_adamw_transform": key == "diffusion_retrac_adamw_full",
                    },
                    handle,
                    indent=2,
                )
        print(f"[saved] {method} q00-q99", flush=True)


if __name__ == "__main__":
    main()
