"""Evaluation layer: metrics, portfolio simulation, baselines and the backtest.

Houses the performance/calibration metrics (Sharpe, Sortino, max drawdown,
Deflated Sharpe per Bailey & Lopez de Prado 2014, Expected Calibration Error
per Naeini 2015), the shared portfolio simulator, the four comparison
baselines of ADR-0002, the :class:`WalkForwardBacktest` harness and the
experiment runner (``python -m advisor.evaluation.run_evaluation``).
"""

from __future__ import annotations

__all__: list[str] = []

try:  # Re-export the metric functions when importable.
    from .metrics import (  # noqa: F401
        deflated_sharpe_ratio,
        expected_calibration_error,
        max_drawdown,
        sharpe_ratio,
        sortino_ratio,
    )

    __all__ += [
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "deflated_sharpe_ratio",
        "expected_calibration_error",
    ]
except Exception:  # pragma: no cover - keep package import resilient.
    pass

try:
    from .portfolio import (  # noqa: F401
        decision_dates,
        pivot_prices,
        simulate_weights,
        summarise,
    )

    __all__ += ["pivot_prices", "simulate_weights", "summarise", "decision_dates"]
except Exception:  # pragma: no cover
    pass

try:
    from .baselines import (  # noqa: F401
        buy_and_hold,
        markowitz_mean_variance,
        momentum_12_1,
        random_allocation,
        random_allocation_ensemble,
    )

    __all__ += [
        "buy_and_hold",
        "random_allocation",
        "random_allocation_ensemble",
        "momentum_12_1",
        "markowitz_mean_variance",
    ]
except Exception:  # pragma: no cover
    pass

try:
    from .backtest import WalkForwardBacktest, ablation_table, evaluate_calibration  # noqa: F401

    __all__ += ["WalkForwardBacktest", "evaluate_calibration", "ablation_table"]
except Exception:  # pragma: no cover
    pass
