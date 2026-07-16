from .algorithm import make_query_phi_fn, make_train_phi_fn

compute_train_gradient_features = make_train_phi_fn
compute_query_gradient_features = make_query_phi_fn


def compute_scores(train_features, query_direction):
    return train_features @ query_direction


__all__ = [
    "compute_train_gradient_features",
    "compute_query_gradient_features",
    "compute_scores",
]
