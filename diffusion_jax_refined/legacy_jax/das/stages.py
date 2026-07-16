from .algorithm import compute_batched_das_term, make_projected_eps_grad_fn

compute_train_gradient_features = make_projected_eps_grad_fn
compute_query_gradient_features = make_projected_eps_grad_fn
compute_scores = compute_batched_das_term

__all__ = [
    "compute_train_gradient_features",
    "compute_query_gradient_features",
    "compute_scores",
]
