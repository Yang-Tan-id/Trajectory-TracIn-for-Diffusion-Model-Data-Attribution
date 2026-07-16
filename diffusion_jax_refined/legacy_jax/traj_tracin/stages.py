from .algorithm import make_query_grad_chunk_fn, make_score_snapshot_chunk_batch_fn

compute_train_gradients = make_score_snapshot_chunk_batch_fn
compute_query_gradients = make_query_grad_chunk_fn


def compute_scores(score_fn, params, query_grads, x0_batch, cond_batch, rngs, t_scalars):
    return score_fn(params, query_grads, x0_batch, cond_batch, rngs, t_scalars)


__all__ = [
    "compute_train_gradients",
    "compute_query_gradients",
    "compute_scores",
]
