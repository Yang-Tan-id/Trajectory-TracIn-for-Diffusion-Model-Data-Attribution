#!/usr/bin/env python3
"""Define the 70 CelebA full-10k random top-1000 removal jobs."""

from __future__ import annotations


METHOD = "random_full10k_seeded"
OUTPUT_GROUP = "endpoint_random_top1000_removal_q0_69"
QUERY_IDS = tuple(range(70))


def build_tasks() -> list[tuple[str, int]]:
    tasks = [(METHOD, query_id) for query_id in QUERY_IDS]
    if len(tasks) != 70 or len(set(tasks)) != 70:
        raise AssertionError("CelebA random baseline must contain 70 unique jobs")
    return tasks


def main() -> None:
    for method, query_id in build_tasks():
        print(f"{method}\t{query_id}")


if __name__ == "__main__":
    main()
