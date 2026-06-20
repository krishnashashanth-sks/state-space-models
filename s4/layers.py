import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class S4Layer(nn.Module):
    def __init__(self, d_model, state_dim, L, dt_min=0.001, dt_max=0.1):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.L = L

        # 1. Stable Initialization (HiPPO-like)
        # We use Log space for Delta to prevent small values from vanishing
        log_dt = torch.rand(d_model) * (np.log(dt_max) - np.log(dt_min)) + np.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

        # A matrix: Standard S4 uses a diagonal + low-rank (DPLR) structure
        # For simplicity and stability, we use a complex diagonal here
        self.A_real = nn.Parameter(torch.exp(torch.randn(d_model, state_dim) * 0.1))
        self.A_imag = nn.Parameter(torch.randn(d_model, state_dim) * 0.1)

        self.C = nn.Parameter(torch.randn(d_model, state_dim, dtype=torch.complex64))
        self.D = nn.Parameter(torch.randn(d_model))

    def forward(self, u):
        # u shape: (batch, d_model, seq_len)
        device = u.device
        L = u.size(-1)

        # 2. Stable Discretization
        # delta = exp(log_delta) ensures delta is always positive
        dt = torch.exp(self.log_dt) # (d_model)

        # Construct complex A: A = -A_real + i*A_imag (Negative real part for stability)
        A = torch.complex(-torch.abs(self.A_real), self.A_imag) # (d_model, state_dim)

        # Adding stability to the spectral filter:
        freqs = torch.exp(torch.arange(L, device=device) * (2j * np.pi / L))

        # 3. Bilinear Transform for frequency domain kernel (Spectral Filter)
        # The continuous frequency 's' is related to discrete frequency 'z' by s = (2/dt) * (z-1)/(z+1)
        # We need to calculate 1 / (s - A_eigenvalue)
        s_bilinear_transform_freqs = (2.0 / dt.unsqueeze(-1)) * ((freqs - 1.0) / (freqs + 1.0)) # Shape (d_model, L)

        # Expand s_bilinear_transform_freqs to (d_model, L, 1) and A to (d_model, 1, state_dim)
        # This allows broadcasting for element-wise subtraction, resulting in (d_model, L, state_dim)
        kernel_f_denominator = s_bilinear_transform_freqs.unsqueeze(-1) - A.unsqueeze(1) + 1e-7
        kernel_f = 1.0 / kernel_f_denominator # Shape (d_model, L, state_dim)

        # Apply C matrix for output. einsum('hsn,hn->hs', kernel_f, self.C) performs: (d_model, L, state_dim) * (d_model, state_dim) -> (d_model, L)
        kernel_f = torch.einsum('hsn,hn->hs', kernel_f, self.C)

        # Convert frequency-domain kernel to time-domain kernel via iFFT
        k = torch.fft.irfft(kernel_f, n=L, dim=-1) # The S4 Kernel, shape (d_model, L)

        # 4. Convolution (in frequency domain using FFT)
        u_f = torch.fft.rfft(u, n=L, dim=-1)
        k_f = torch.fft.rfft(k, n=L, dim=-1)
        y = torch.fft.irfft(u_f * k_f, n=L, dim=-1)

        # 5. Skip connection (D term)
        return y + self.D.unsqueeze(-1) * u