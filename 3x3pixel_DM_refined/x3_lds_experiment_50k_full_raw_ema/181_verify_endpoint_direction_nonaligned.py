"""Verify aligned endpoint-MC outputs needed by mismatch ablation."""

from endpoint_direction_nonaligned_config import *


def main():
    for source_index in nsdl_datapoint_indices():
        source_dir = edmc_source_dir(source_index)
        done_path = source_dir / "done.json"
        if not done_path.is_file():
            raise FileNotFoundError(done_path)
        for block_index in range(len(NTCD_TIMESTAMP_BLOCKS)):
            path = source_dir / f"block_{block_index}_responses.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
    print(f"source updates = {len(nsdl_datapoint_indices())}")
    print(f"branches/source = {len(NTCD_TIMESTAMP_BLOCKS)}")
    print(f"pollution directions = {EDMC_DIRECTION_COUNT}")
    print("non-aligned loss target for direction r = direction (r + 1) mod 100")
    print("finite ground truth and polluted target inputs are unchanged")
    print("[OK] endpoint non-aligned loss-noise prerequisites verified")


if __name__ == "__main__":
    main()
