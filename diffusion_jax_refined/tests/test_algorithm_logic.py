from __future__ import annotations

import math
import random
import unittest

try:
    from .test_lds_eval_math import spearman
    from .test_score_indices import combine_scores
except ImportError:  # Allows running this file directly from the tests folder.
    from test_lds_eval_math import spearman
    from test_score_indices import combine_scores


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def add(a: list[float], b: list[float]) -> list[float]:
    return [x + y for x, y in zip(a, b)]


def scale(a: list[float], c: float) -> list[float]:
    return [x * c for x in a]


def linear_grad(theta: list[float], x: list[float], y: float) -> list[float]:
    """Gradient of 0.5 * (theta dot x - y)^2."""
    residual = dot(theta, x) - y
    return scale(x, residual)


def linear_loss(theta: list[float], x: list[float], y: float) -> float:
    residual = dot(theta, x) - y
    return 0.5 * residual * residual


def one_step_subset_theta(
    theta0: list[float],
    train_x: list[list[float]],
    train_y: list[float],
    kept: list[int],
    *,
    lr: float,
) -> list[float]:
    grad = [0.0 for _ in theta0]
    for idx in kept:
        grad = add(grad, linear_grad(theta0, train_x[idx], train_y[idx]))
    grad = scale(grad, 1.0 / len(kept))
    return add(theta0, scale(grad, -lr))


def tracin_scores(
    theta0: list[float],
    train_x: list[list[float]],
    train_y: list[float],
    query_x: list[float],
    query_y: float,
) -> list[float]:
    query_grad = linear_grad(theta0, query_x, query_y)
    return [dot(query_grad, linear_grad(theta0, x, y)) for x, y in zip(train_x, train_y)]


def das_raw_scores(query_phi: list[float], train_phi: list[list[float]]) -> list[float]:
    return [dot(query_phi, phi) for phi in train_phi]


def das_batched_squared_scores(query_phi: list[float], train_phi: list[list[float]], batch_size: int) -> list[float]:
    out = []
    for start in range(0, len(train_phi), batch_size):
        for raw in das_raw_scores(query_phi, train_phi[start : start + batch_size]):
            out.append(raw * raw)
    return out


def invert_2x2(matrix: list[list[float]]) -> list[list[float]]:
    [[a, b], [c, d]] = matrix
    det = a * d - b * c
    if abs(det) < 1e-12:
        raise ValueError("singular matrix")
    return [[d / det, -b / det], [-c / det, a / det]]


def mat_vec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [dot(row, vector) for row in matrix]


def projected_das_scores(
    query_phi: list[float],
    train_phi: list[list[float]],
    residuals: list[float],
    *,
    damping: float,
    use_denominator: bool,
) -> list[float]:
    h = [[damping, 0.0], [0.0, damping]]
    for phi in train_phi:
        h[0][0] += phi[0] * phi[0]
        h[0][1] += phi[0] * phi[1]
        h[1][0] += phi[1] * phi[0]
        h[1][1] += phi[1] * phi[1]
    h_inv = invert_2x2(h)
    u = mat_vec(h_inv, query_phi)
    scores = []
    for phi, residual in zip(train_phi, residuals):
        raw = dot(phi, u) * residual
        if use_denominator:
            h_inv_phi = mat_vec(h_inv, phi)
            raw /= 1.0 - dot(phi, h_inv_phi)
        scores.append(raw * raw)
    return scores


def projected_das_scores_batched(
    query_phi: list[float],
    train_phi: list[list[float]],
    residuals: list[float],
    *,
    damping: float,
    batch_size: int,
    use_denominator: bool,
) -> list[float]:
    cached_phi = []
    cached_residuals = []
    h = [[damping, 0.0], [0.0, damping]]
    for start in range(0, len(train_phi), batch_size):
        for phi, residual in zip(train_phi[start : start + batch_size], residuals[start : start + batch_size]):
            cached_phi.append(phi)
            cached_residuals.append(residual)
            h[0][0] += phi[0] * phi[0]
            h[0][1] += phi[0] * phi[1]
            h[1][0] += phi[1] * phi[0]
            h[1][1] += phi[1] * phi[1]
    h_inv = invert_2x2(h)
    u = mat_vec(h_inv, query_phi)
    scores = []
    for start in range(0, len(cached_phi), batch_size):
        for phi, residual in zip(cached_phi[start : start + batch_size], cached_residuals[start : start + batch_size]):
            raw = dot(phi, u) * residual
            if use_denominator:
                h_inv_phi = mat_vec(h_inv, phi)
                raw /= 1.0 - dot(phi, h_inv_phi)
            scores.append(raw * raw)
    return scores


def mc_endpoint_objective(theta: list[float], eps_predictions: list[list[float]], eps_targets: list[list[float]]) -> float:
    losses = []
    for pred, target in zip(eps_predictions, eps_targets):
        residual = [dot(theta, [p]) - t for p, t in zip(pred, target)]
        losses.append(sum(r * r for r in residual) / len(residual))
    return sum(losses) / len(losses)


def mc_endpoint_objective_batched(
    theta: list[float],
    eps_predictions: list[list[float]],
    eps_targets: list[list[float]],
    batch_size: int,
) -> float:
    weighted_sum = 0.0
    count = 0
    for start in range(0, len(eps_predictions), batch_size):
        batch_preds = eps_predictions[start : start + batch_size]
        batch_targets = eps_targets[start : start + batch_size]
        weighted_sum += mc_endpoint_objective(theta, batch_preds, batch_targets) * len(batch_preds)
        count += len(batch_preds)
    return weighted_sum / count


class TestAlgorithmLogic(unittest.TestCase):
    def test_toy_tracin_scores_predict_one_step_lds_direction(self):
        theta0 = [0.0, 0.0]
        train_x = [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [-1.0, 0.0],
            [-0.8, -0.2],
            [0.2, -1.0],
        ]
        train_y = [1.0, 1.0, 0.2, -1.0, -1.0, -0.2]
        query_x = [1.0, 0.0]
        query_y = 1.0
        scores = tracin_scores(theta0, train_x, train_y, query_x, query_y)

        rng = random.Random(7)
        preds = []
        true_improvements = []
        baseline = linear_loss(theta0, query_x, query_y)
        for _ in range(30):
            kept = sorted(rng.sample(range(len(train_x)), 3))
            pred = sum(scores[i] for i in kept)
            theta1 = one_step_subset_theta(theta0, train_x, train_y, kept, lr=0.15)
            true_improvement = baseline - linear_loss(theta1, query_x, query_y)
            preds.append(pred)
            true_improvements.append(true_improvement)

        self.assertGreater(spearman(preds, true_improvements), 0.95)
        self.assertLess(spearman([-x for x in preds], true_improvements), -0.95)

    def test_traj_shard_merge_preserves_full_prediction(self):
        scores = {idx: math.sin(idx) + 0.1 * idx for idx in range(20)}
        shards = [
            list(range(0, 4)),
            list(range(4, 8)),
            list(range(8, 12)),
            list(range(12, 16)),
            list(range(16, 20)),
        ]
        kept = [0, 3, 5, 8, 11, 13, 19]
        full_pred = sum(scores[idx] for idx in kept)
        shard_pred = 0.0
        for shard in shards:
            shard_pred += sum(scores[idx] for idx in kept if idx in set(shard))
        self.assertAlmostEqual(shard_pred, full_pred)

    def test_das_batched_and_unbatched_are_mathematically_identical(self):
        query_phi = [0.5, -1.0, 2.0]
        train_phi = [
            [1.0, 0.0, 0.5],
            [0.0, -2.0, 1.0],
            [1.5, 1.0, -0.5],
            [-1.0, 0.3, 0.2],
            [0.2, 0.4, 0.6],
        ]
        unbatched = [raw * raw for raw in das_raw_scores(query_phi, train_phi)]
        batched = das_batched_squared_scores(query_phi, train_phi, batch_size=2)
        self.assertEqual(len(batched), len(unbatched))
        for a, b in zip(batched, unbatched):
            self.assertAlmostEqual(a, b)

    def test_projected_das_batched_path_matches_unbatched_without_result_io(self):
        query_phi = [0.6, -0.4]
        train_phi = [
            [1.0, 0.2],
            [0.1, -0.7],
            [0.3, 0.8],
            [-0.5, 0.4],
            [0.9, -0.1],
        ]
        residuals = [0.8, -0.3, 1.1, -0.6, 0.2]

        for use_denominator in (False, True):
            with self.subTest(use_denominator=use_denominator):
                unbatched = projected_das_scores(
                    query_phi,
                    train_phi,
                    residuals,
                    damping=0.25,
                    use_denominator=use_denominator,
                )
                batched = projected_das_scores_batched(
                    query_phi,
                    train_phi,
                    residuals,
                    damping=0.25,
                    batch_size=2,
                    use_denominator=use_denominator,
                )
                self.assertEqual(len(batched), len(unbatched))
                for a, b in zip(batched, unbatched):
                    self.assertAlmostEqual(a, b)

    def test_projected_das_sherman_morrison_denominator_changes_scores(self):
        query_phi = [0.6, -0.4]
        train_phi = [[1.0, 0.2], [0.1, -0.7], [0.3, 0.8], [-0.5, 0.4]]
        residuals = [0.8, -0.3, 1.1, -0.6]
        without = projected_das_scores(
            query_phi,
            train_phi,
            residuals,
            damping=0.25,
            use_denominator=False,
        )
        with_denom = projected_das_scores(
            query_phi,
            train_phi,
            residuals,
            damping=0.25,
            use_denominator=True,
        )
        self.assertEqual(len(with_denom), len(without))
        self.assertTrue(any(abs(a - b) > 1e-8 for a, b in zip(with_denom, without)))

    def test_endpoint_mc_average_matches_batched_average(self):
        theta = [1.25]
        eps_predictions = [[0.1, 0.2, 0.3], [0.4, 0.0, -0.2], [1.0, 0.5, -0.5], [-0.1, 0.7, 0.2]]
        eps_targets = [[0.0, 0.1, 0.0], [0.2, -0.1, -0.3], [0.8, 0.4, -0.4], [0.0, 0.5, 0.1]]
        explicit = mc_endpoint_objective(theta, eps_predictions, eps_targets)
        batched = mc_endpoint_objective_batched(theta, eps_predictions, eps_targets, batch_size=3)
        self.assertAlmostEqual(explicit, batched)

    def test_duplicate_policy_changes_prediction_in_expected_way(self):
        part_a = ([0, 1, 2], [1.0, 2.0, 3.0])
        part_b = ([2, 3], [30.0, 4.0])
        kept = {1, 2, 3}
        expected = {
            "max": 2.0 + 30.0 + 4.0,
            "sum": 2.0 + 33.0 + 4.0,
            "mean": 2.0 + 16.5 + 4.0,
        }
        for policy, expected_pred in expected.items():
            with self.subTest(policy=policy):
                indices, scores = combine_scores([part_a, part_b], policy)
                score_map = dict(zip(indices, scores))
                pred = sum(score_map[idx] for idx in kept)
                self.assertAlmostEqual(pred, expected_pred)


if __name__ == "__main__":
    unittest.main()
