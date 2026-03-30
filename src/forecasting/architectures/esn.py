import numpy as np


class ESN:
    """Classic Echo State Network with fixed reservoir and ridge readout."""

    def __init__(
        self,
        n_inputs: int = 1,
        n_reservoir: int = 500,
        spectral_radius: float = 0.9,
        input_scale: float = 0.1,
        leak_rate: float = 1.0,
        ridge_alpha: float = 1e-4,
        seed: int = 42,
    ) -> None:
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.input_scale = input_scale
        self.leak_rate = leak_rate
        self.ridge_alpha = ridge_alpha

        rng = np.random.RandomState(seed)
        # work in float64 for linear algebra stability, cast to float32 at the end
        self.W_in = (rng.rand(n_reservoir, n_inputs).astype(np.float64) - 0.5) * 2.0 * input_scale

        W_raw = rng.rand(n_reservoir, n_reservoir).astype(np.float64) - 0.5
        eigvals = np.linalg.eigvals(W_raw)
        sr = np.max(np.abs(eigvals))
        if not np.isfinite(sr) or sr == 0.0:
            raise ValueError("Failed to initialize ESN reservoir: invalid spectral radius norm")
        self.W = (W_raw * (spectral_radius / sr)).astype(np.float64)
        self.W_out = None

    def _compute_states(self, X: np.ndarray) -> np.ndarray:
        """Compute reservoir states for each window in X (N, T)."""
        N, T = X.shape
        states = np.zeros((N, self.n_reservoir), dtype=np.float64)
        for n in range(N):
            x = np.zeros(self.n_reservoir, dtype=np.float64)
            u_seq = X[n]
            for t in range(T):
                u_t = np.array([u_seq[t]], dtype=np.float64)
                preact = self.W_in @ u_t + self.W @ x
                x_new = np.tanh(preact)
                x = (1.0 - self.leak_rate) * x + self.leak_rate * x_new
            states[n] = x
        return states.astype(np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit readout via ridge regression solved as a regularized least squares
        to avoid potential MKL/BLAS instability from direct solve.
        """
        states = self._compute_states(X).astype(np.float64)
        N = states.shape[0]
        S = np.hstack([states, np.ones((N, 1), dtype=np.float64)])
        alpha = float(self.ridge_alpha)
        I = np.eye(S.shape[1], dtype=np.float64)
        A = S.T @ S + alpha * I
        B = S.T @ y.astype(np.float64)
        self.W_out = np.linalg.solve(A, B).astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.W_out is not None, 'Call fit() first.'
        states = self._compute_states(X)
        N = states.shape[0]
        S = np.hstack([states, np.ones((N, 1), dtype=np.float32)])
        y_pred = S @ self.W_out
        return y_pred.astype(np.float32)


class ChaoticESN(ESN):
    """ESN with optional chaotic spectral radius override for pairwise comparisons."""

    def __init__(
        self,
        n_inputs: int = 1,
        n_reservoir: int = 500,
        spectral_radius: float = 0.9,
        chaotic_spectral_radius: float | None = 1.5,
        input_scale: float = 0.1,
        leak_rate: float = 1.0,
        ridge_alpha: float = 1e-4,
        seed: int = 43,
    ) -> None:
        self.base_spectral_radius = float(spectral_radius)
        self.chaotic_spectral_radius = (
            float(chaotic_spectral_radius) if chaotic_spectral_radius is not None else float(spectral_radius)
        )
        super().__init__(
            n_inputs=n_inputs,
            n_reservoir=n_reservoir,
            spectral_radius=self.chaotic_spectral_radius,
            input_scale=input_scale,
            leak_rate=leak_rate,
            ridge_alpha=ridge_alpha,
            seed=seed,
        )
