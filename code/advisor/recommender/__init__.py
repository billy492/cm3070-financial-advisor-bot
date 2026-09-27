"""Recommender layer: prompt construction, schema, and the advisors.

Bundles the :class:`Recommendation` schema, the prompt templates, the
LLM-backed :class:`OllamaAdvisor`, the rule-based :class:`HeuristicAdvisor`
(the "no AI" baseline / ablation arm) and the :class:`CachedAdvisor` wrapper
that memoises answers so backtests are resumable and reproducible offline
(ADR-0001/0002).
"""

from __future__ import annotations

__all__: list[str] = []

try:  # Re-export the recommendation schema when importable.
    from .schema import Action, Recommendation  # noqa: F401

    __all__ += ["Action", "Recommendation"]
except Exception:  # pragma: no cover - keep package import resilient.
    pass

try:
    from .prompts import (  # noqa: F401
        PROMPT_VERSION,
        SYSTEM_PROMPT,
        build_advisor_prompt,
        select_features,
    )

    __all__ += ["PROMPT_VERSION", "SYSTEM_PROMPT", "build_advisor_prompt", "select_features"]
except Exception:  # pragma: no cover
    pass

try:
    from .llm import OllamaAdvisor  # noqa: F401

    __all__ += ["OllamaAdvisor"]
except Exception:  # pragma: no cover
    pass

try:
    from .heuristic import HeuristicAdvisor  # noqa: F401

    __all__ += ["HeuristicAdvisor"]
except Exception:  # pragma: no cover
    pass

try:
    from .cache import CachedAdvisor  # noqa: F401

    __all__ += ["CachedAdvisor"]
except Exception:  # pragma: no cover
    pass
