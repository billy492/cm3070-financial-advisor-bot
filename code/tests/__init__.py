"""Test suite for the CM3070 Financial Advisor Bot.

This package contains the offline, deterministic ``pytest`` test suite for the
``advisor`` package. Every test in this suite is designed to run with only
``pandas``, ``numpy``, ``pyarrow`` and ``pytest`` installed -- with **no
network access**, **no Ollama server** and **no ``dice-ml``** dependency.

The tests exercise the *working* components of the system (schema, technical
indicators, evaluation metrics, and the cache-first data loader). Stubbed
components (calibration, counterfactuals, baselines, backtest) are intentionally
not covered here because they raise :class:`NotImplementedError` by design.

See ``ADR-0001`` / ``ADR-0002`` for the locked architecture decisions that these
tests guard against regression.
"""

from __future__ import annotations
