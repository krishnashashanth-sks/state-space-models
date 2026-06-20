import torch

# --- Define NLL Loss Function ---
def calculate_nll_loss(innovations, innovation_covariances, obs_dim):
    loss = 0.0
    for innovation, innovation_covariance in zip(innovations, innovation_covariances):
        # NLL for multivariate normal distribution:
        # L = 0.5 * (log(det(S)) + innovation^T @ S_inv @ innovation + obs_dim * log(2 * pi))
        # We can ignore the constant obs_dim * log(2 * pi) for optimization

        S_inv = torch.linalg.inv(innovation_covariance)
        log_det_S = torch.logdet(innovation_covariance)

        loss += 0.5 * (log_det_S + innovation.T @ S_inv @ innovation)
    return loss