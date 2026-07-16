from .algorithm import compute_g_end, make_score_train_batch_fn

compute_train_gradients = make_score_train_batch_fn
compute_query_gradient = compute_g_end


def compute_scores(score_fn, params, x0_batch, cond_batch, rng):
    scores, _losses = score_fn(params, x0_batch, cond_batch, rng)
    return scores


__all__ = [
    "compute_train_gradients",
    "compute_query_gradient",
    "compute_scores",
]
