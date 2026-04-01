from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.multioutput import MultiOutputRegressor


class MetaModelError(RuntimeError):
    """Raised when a meta-model cannot be built or trained."""


@dataclass
class MetaModelSpec:
    name: str
    family: str
    enabled_default: bool
    notes: str


class CatBoostPerTargetRegressor:
    """Train one CatBoost regressor per output dimension for reproducibility."""

    def __init__(self, random_seed: int = 42, iterations: int = 300, depth: int = 6, learning_rate: float = 0.05) -> None:
        self.random_seed = int(random_seed)
        self.iterations = int(iterations)
        self.depth = int(depth)
        self.learning_rate = float(learning_rate)
        self._models: list[Any] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CatBoostPerTargetRegressor":
        try:
            from catboost import CatBoostRegressor
        except Exception as exc:  # pragma: no cover - environment-dependent
            raise MetaModelError("catboost is not installed in the current environment") from exc

        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim != 2:
            raise MetaModelError("CatBoostPerTargetRegressor expects 2D target array")

        self._models = []
        for col_idx in range(y_arr.shape[1]):
            model = CatBoostRegressor(
                loss_function="RMSE",
                random_seed=self.random_seed,
                iterations=self.iterations,
                depth=self.depth,
                learning_rate=self.learning_rate,
                verbose=False,
            )
            model.fit(X, y_arr[:, col_idx])
            self._models.append(model)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._models:
            raise MetaModelError("CatBoostPerTargetRegressor is not fitted")
        cols = [m.predict(X) for m in self._models]
        return np.column_stack(cols).astype(np.float64)


class CatBoostClassifierWrapper:
    """Thin wrapper around CatBoostClassifier with reproducible defaults."""

    def __init__(
        self,
        random_seed: int = 42,
        iterations: int = 300,
        depth: int = 6,
        learning_rate: float = 0.05,
        auto_class_weights: str | None = None,
        train_dir: str | None = None,
    ) -> None:
        self.random_seed = int(random_seed)
        self.iterations = int(iterations)
        self.depth = int(depth)
        self.learning_rate = float(learning_rate)
        self.auto_class_weights = str(auto_class_weights) if auto_class_weights else None
        self.train_dir = str(train_dir) if train_dir else None
        self._model: Any | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CatBoostClassifierWrapper":
        try:
            from catboost import CatBoostClassifier
        except Exception as exc:  # pragma: no cover - environment-dependent
            raise MetaModelError("catboost is not installed in the current environment") from exc

        model_kwargs: dict[str, Any] = {
            "random_seed": self.random_seed,
            "iterations": self.iterations,
            "depth": self.depth,
            "learning_rate": self.learning_rate,
            "verbose": False,
        }
        if self.auto_class_weights:
            model_kwargs["auto_class_weights"] = self.auto_class_weights
        if self.train_dir:
            model_kwargs["train_dir"] = self.train_dir

        model = CatBoostClassifier(
            **model_kwargs,
        )
        model.fit(X, y)
        self._model = model
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise MetaModelError("CatBoostClassifierWrapper is not fitted")
        return np.asarray(self._model.predict(X)).reshape(-1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise MetaModelError("CatBoostClassifierWrapper is not fitted")
        return np.asarray(self._model.predict_proba(X), dtype=np.float64)

    @property
    def feature_importances_(self) -> np.ndarray:
        if self._model is None:
            raise MetaModelError("CatBoostClassifierWrapper is not fitted")
        return np.asarray(self._model.get_feature_importance(), dtype=np.float64)


class CatBoostRegressorWrapper:
    """Thin wrapper around CatBoostRegressor for single-target scoring tasks."""

    def __init__(
        self,
        random_seed: int = 42,
        iterations: int = 300,
        depth: int = 6,
        learning_rate: float = 0.05,
    ) -> None:
        self.random_seed = int(random_seed)
        self.iterations = int(iterations)
        self.depth = int(depth)
        self.learning_rate = float(learning_rate)
        self._model: Any | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CatBoostRegressorWrapper":
        try:
            from catboost import CatBoostRegressor
        except Exception as exc:  # pragma: no cover - environment-dependent
            raise MetaModelError("catboost is not installed in the current environment") from exc

        model = CatBoostRegressor(
            loss_function="RMSE",
            random_seed=self.random_seed,
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            verbose=False,
        )
        model.fit(X, y)
        self._model = model
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise MetaModelError("CatBoostRegressorWrapper is not fitted")
        return np.asarray(self._model.predict(X), dtype=np.float64).reshape(-1)

    @property
    def feature_importances_(self) -> np.ndarray:
        if self._model is None:
            raise MetaModelError("CatBoostRegressorWrapper is not fitted")
        return np.asarray(self._model.get_feature_importance(), dtype=np.float64)


def get_meta_model_specs() -> dict[str, MetaModelSpec]:
    return {
        "ridge": MetaModelSpec(
            name="ridge",
            family="linear",
            enabled_default=True,
            notes="MultiOutputRegressor over sklearn Ridge",
        ),
        "random_forest": MetaModelSpec(
            name="random_forest",
            family="tree",
            enabled_default=True,
            notes="native multi-output sklearn RandomForestRegressor",
        ),
        "catboost": MetaModelSpec(
            name="catboost",
            family="boosting",
            enabled_default=True,
            notes="one CatBoostRegressor per output dimension",
        ),
    }


def build_meta_model(model_name: str, cfg: dict[str, Any]) -> Any:
    specs = get_meta_model_specs()
    if model_name not in specs:
        raise MetaModelError(f"Meta-model is not registered: {model_name}")

    model_overrides = dict(cfg.get("model_overrides", {}).get(model_name, {}))
    random_seed = int(cfg.get("split", {}).get("random_seed", 42))

    if model_name == "ridge":
        alpha = float(model_overrides.get("alpha", 1.0))
        return MultiOutputRegressor(Ridge(alpha=alpha, random_state=random_seed))

    if model_name == "random_forest":
        n_estimators = int(model_overrides.get("n_estimators", 300))
        max_depth = model_overrides.get("max_depth", None)
        if max_depth is not None:
            max_depth = int(max_depth)
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_seed,
            n_jobs=int(model_overrides.get("n_jobs", -1)),
        )

    if model_name == "catboost":
        return CatBoostPerTargetRegressor(
            random_seed=random_seed,
            iterations=int(model_overrides.get("iterations", 300)),
            depth=int(model_overrides.get("depth", 6)),
            learning_rate=float(model_overrides.get("learning_rate", 0.05)),
        )

    raise MetaModelError(f"Unsupported meta-model: {model_name}")


def build_meta_classifier(
    model_name: str,
    cfg: dict[str, Any],
    *,
    balancing_mode: str = "default",
    catboost_train_dir: str | None = None,
) -> Any:
    """Factory for multiclass routing models."""
    model_overrides = dict(cfg.get("model_overrides", {}).get(model_name, {}))
    random_seed = int(cfg.get("split", {}).get("random_seed", 42))

    if model_name == "logistic_regression":
        c_value = float(model_overrides.get("C", 1.0))
        max_iter = int(model_overrides.get("max_iter", 1000))
        class_weight = "balanced" if str(balancing_mode).lower() == "balanced" else None
        return LogisticRegression(
            C=c_value,
            max_iter=max_iter,
            random_state=random_seed,
            class_weight=class_weight,
        )

    if model_name == "random_forest_classifier":
        n_estimators = int(model_overrides.get("n_estimators", 300))
        max_depth = model_overrides.get("max_depth", None)
        if max_depth is not None:
            max_depth = int(max_depth)
        class_weight = "balanced" if str(balancing_mode).lower() == "balanced" else None
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_seed,
            n_jobs=int(model_overrides.get("n_jobs", -1)),
            class_weight=class_weight,
        )

    if model_name == "catboost_classifier":
        auto_class_weights = "Balanced" if str(balancing_mode).lower() == "balanced" else None
        return CatBoostClassifierWrapper(
            random_seed=random_seed,
            iterations=int(model_overrides.get("iterations", 300)),
            depth=int(model_overrides.get("depth", 6)),
            learning_rate=float(model_overrides.get("learning_rate", 0.05)),
            auto_class_weights=auto_class_weights,
            train_dir=catboost_train_dir,
        )

    raise MetaModelError(f"Unsupported meta-classifier: {model_name}")


def build_scoring_model(model_name: str, cfg: dict[str, Any]) -> Any:
    """Factory for long-format model scoring regressors."""
    model_overrides = dict(cfg.get("model_overrides", {}).get(model_name, {}))
    random_seed = int(cfg.get("split", {}).get("random_seed", 42))

    if model_name == "ridge":
        alpha = float(model_overrides.get("alpha", 1.0))
        return Ridge(alpha=alpha, random_state=random_seed)

    if model_name == "random_forest":
        n_estimators = int(model_overrides.get("n_estimators", 300))
        max_depth = model_overrides.get("max_depth", None)
        if max_depth is not None:
            max_depth = int(max_depth)
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_seed,
            n_jobs=int(model_overrides.get("n_jobs", -1)),
        )

    if model_name == "catboost":
        return CatBoostRegressorWrapper(
            random_seed=random_seed,
            iterations=int(model_overrides.get("iterations", 300)),
            depth=int(model_overrides.get("depth", 6)),
            learning_rate=float(model_overrides.get("learning_rate", 0.05)),
        )

    raise MetaModelError(f"Unsupported scoring model: {model_name}")
