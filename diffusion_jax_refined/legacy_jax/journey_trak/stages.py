from .algorithm import (
    diffusion_train_loss_expected_at_t_jax,
    grad_feature_phi_jax,
    trajectory_query_loss_expected_at_step_jax,
)

compute_train_gradient_features = diffusion_train_loss_expected_at_t_jax
compute_query_gradient_features = trajectory_query_loss_expected_at_step_jax
project_gradient_feature = grad_feature_phi_jax


def compute_scores(train_features, query_direction):
    return train_features @ query_direction


__all__ = [
    "compute_train_gradient_features",
    "compute_query_gradient_features",
    "project_gradient_feature",
    "compute_scores",
]
