import torch

def simulate_pytorch_ssm(A, C, Q, R, x0, num_steps):
    states = torch.zeros(num_steps, A.shape[0])
    observations = torch.zeros(num_steps, C.shape[0])

    x_t = x0.clone()

    for t in range(num_steps):
        # State evolution
        process_noise = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(A.shape[0]), Q).sample().reshape(-1, 1)
        x_t = A @ x_t + process_noise
        states[t, :] = x_t.flatten()

        # Observation
        observation_noise = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(C.shape[0]), R).sample().reshape(-1, 1)
        y_t = C @ x_t + observation_noise
        observations[t, :] = y_t.flatten()

    return states, observations
