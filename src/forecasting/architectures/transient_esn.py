import numpy as np
from .esn import ESN


def make_gain_schedule(T: int, g_start: float, g_end: float) -> np.ndarray:
    """Exponentially decaying gain from g_start to g_end over T steps."""
    if T <= 1:
        return np.asarray([g_end], dtype=np.float32)
    t_arr = np.arange(T, dtype=np.float32)
    gains = g_start * (g_end / g_start) ** (t_arr / max(1, T - 1))
    return gains.astype(np.float32)


class TransientChaoticESN(ESN):
    """ESN with transient chaotic regime controlled by per-step gains."""

    def __init__(
        self,
        n_inputs: int = 1,
        n_reservoir: int = 500,
        spectral_radius: float = 1.5,
        input_scale: float = 0.1,
        leak_rate: float = 1.0,
        ridge_alpha: float = 1e-4,
        g_start: float = 1.5,
        g_end: float = 0.9,
        seed: int = 44,
    ) -> None:
        super().__init__(
            n_inputs=n_inputs,
            n_reservoir=n_reservoir,
            spectral_radius=spectral_radius,
            input_scale=input_scale,
            leak_rate=leak_rate,
            ridge_alpha=ridge_alpha,
            seed=seed,
        )
        self.g_start = g_start
        self.g_end = g_end

    def _compute_states(self, X: np.ndarray) -> np.ndarray:
        N, T = X.shape
        states = np.zeros((N, self.n_reservoir), dtype=np.float32)
        gains = make_gain_schedule(T, g_start=self.g_start, g_end=self.g_end)
        for n in range(N):
            x = np.zeros(self.n_reservoir, dtype=np.float32)
            u_seq = X[n]
            for t in range(T):
                u_t = np.array([u_seq[t]], dtype=np.float32)
                g_t = gains[t]
                preact = self.W_in @ u_t + g_t * (self.W @ x)
                x_new = np.tanh(preact)
                x = (1.0 - self.leak_rate) * x + self.leak_rate * x_new
            states[n] = x
        return states
