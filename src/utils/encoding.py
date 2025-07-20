import torch
import torch.nn as nn

class PositionalEncoder(nn.Module):
    """
    Positional Encoding for NeRF. Maps a low-dimensional input to a
    higher-dimensional space using a combination of sine and cosine functions.
    """
    def __init__(self, d_input: int, n_freqs: int, log_space: bool = True):
        """
        Args:
            d_input (int): Dimension of the input tensor.
            n_freqs (int): Number of frequency bands for encoding.
            log_space (bool): If True, frequencies are sampled in log space.
        """
        super().__init__()
        self.d_input = d_input
        self.n_freqs = n_freqs
        self.log_space = log_space
        self.d_output = d_input * (1 + 2 * self.n_freqs)
        self.embed_fns = [lambda x: x]

        # Define frequency bands
        if self.log_space:
            freq_bands = 2.**torch.linspace(0., self.n_freqs - 1, self.n_freqs)
        else:
            freq_bands = torch.linspace(1., 2.**(self.n_freqs - 1), self.n_freqs)

        for freq in freq_bands:
            self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))
            self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies positional encoding to the input tensor.

        Args:
            x (torch.Tensor): The input tensor of shape (..., d_input).

        Returns:
            torch.Tensor: The encoded tensor of shape (..., d_output).
        """
        return torch.cat([fn(x) for fn in self.embed_fns], dim=-1) 