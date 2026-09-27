# Financial Advisor Bot — CM3070 Final Project

> BSc Computer Science final project, University of London (Goldsmiths). Template: CM3020 Artificial Intelligence, *Project Idea 2: Financial Advisor Bot*. Author: Aly Thabet.

A calibrated, explainable stock-recommendation system for non-expert investors. For one US large-cap stock on one date it returns four things: a **BUY / HOLD / SELL** call, a **confidence** shown next to the advisor's measured **track record** on held-out data, a plain-English **reason**, and a **verified counterfactual** ("I would change my mind from BUY to HOLD if the 14-day RSI rose above 75"), found by re-querying the model rather than trusting its own sentence.

**Research and demonstration system only. It never places trades, holds no broker keys, and handles no money.**

## How it works (six stages)

1. **Data** — daily OHLCV for 51 US large caps (six sectors) plus SPY from Yahoo Finance via `yfinance`, cached as one parquet file per ticker (`advisor/data`).
2. **Features** — 1-day return, 10- and 50-day moving averages, 10-day momentum, 20-day volatility, Wilder RSI-14; plain pandas, strictly point-in-time (`advisor/features`).
3. **Recommender** — a pretrained open-weight LLM served locally by [Ollama](https://ollama.com) reads the indicators and answers in strict JSON. Two models of the same size class are compared under an identical prompt: **Llama 3.1 8B-Instruct** and **Qwen3 8B**. A transparent momentum/moving-average rule is the no-LLM ablation (`advisor/recommender`).
4. **Calibration** — the model's verbal confidence is temperature-scaled (Guo et al. 2017) on past, already-resolved recommendations; actions never change, only how sure the bot sounds (`advisor/calibration`).
5. **Counterfactual** — a DiCE-style black-box search (Mothilal et al. 2020; Wachter et al. 2018) finds the smallest feasible indicator change that flips the call and renders it in plain English (`advisor/counterfactual`).
6. **Web UI** — a Streamlit page for non-technical users (`streamlit_app.py`).

Evaluation: expanding-window **walk-forward backtest** on the held-out window 2025-01-01 → 2026-06-01 with weekly decisions and 10 bps costs, against four baselines (S&P 500 buy-and-hold, seeded random allocation with a 100-seed null distribution, 12-1 momentum, long-only Markowitz mean-variance), reporting the Deflated Sharpe ratio (Bailey & López de Prado 2014), Sortino, maximum drawdown, Calmar, a permutation p-value, and calibration metrics (ECE, adaptive ECE, Brier, sharpness). Hypotheses were pre-registered in `../docs/preregistration.md` before the first LLM run.

## Results at a glance (held-out 2 Jan 2025 – 29 May 2026, weekly, 10 bps costs)

| Strategy | Total return | Sharpe | Deflated Sharpe (7 trials) | Max drawdown |
|---|---:|---:|---:|---:|
| Qwen3 8B advisor | +25.5% | 1.06 | 0.80 | −18.5% |
| Llama 3.1 8B advisor | +20.5% | 0.98 | 0.77 | −16.9% |
| Rule-based advisor (no LLM) | +16.6% | 0.81 | 0.70 | −17.7% |
| S&P 500 buy-and-hold | +31.1% | 1.17 | 0.84 | −18.8% |
| Markowitz mean-variance | +34.2% | 1.34 | 0.88 | −16.6% |

Pre-registered verdicts: neither LLM beats buy-and-hold after deflation (H1 not supported); temperature scaling lowers ECE only by collapsing sharpness (H2 not supported); removing the LLM costs 23% of Sharpe against the better model (H3 supported). Post-hoc: within BUY/SELL calls the models' verbal confidence carries no information, Llama follows its own prompt's decision policy on only 47.6% of answers, and when the models' written reasons say the RSI is "above 70" or "below 30" they are wrong more than half the time (Llama 676/1,093, Qwen 521/909). Details: report Chapter 5.

## Quick start (three commands)

```bash
git clone <repo-url> && cd cm3070-financial-advisor-bot/code
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m advisor.reproduce          # offline smoke run: features -> rule-based recommendation (+ results table if present)
```

### Full pipeline

```bash
# 1. Local models (one-off; ~5 GB each)
ollama pull llama3.1:8b && ollama pull qwen3:8b

# 2. Warm the price cache (network, ~1 minute)
.venv/bin/python scripts/fetch_data.py

# 3. Run the evaluation (rule-based advisor + baselines take seconds; each LLM advisor ≈ 3,825 model calls, several hours, resumable)
.venv/bin/python -m advisor.evaluation.run_evaluation --advisors heuristic --tickers ALL
.venv/bin/python -m advisor.evaluation.run_evaluation --advisors llama qwen --tickers ALL --resume

# 3b. Post-hoc calibration analysis (no model calls; replays temperature vs Platt scaling from the logs)
.venv/bin/python -m advisor.evaluation.calibration_analysis --tags llama qwen heuristic

# 3c. Post-hoc policy adherence (no model calls) and counterfactual experiments (local Ollama)
.venv/bin/python -m advisor.evaluation.policy_adherence --tags llama qwen
.venv/bin/python -m advisor.evaluation.sharpe_difference        # bootstrap CIs for Sharpe gaps
# cost sensitivity (copy the two llm_cache_*.jsonl files into the folder first; no model calls)
.venv/bin/python -m advisor.evaluation.run_evaluation --advisors llama qwen heuristic --tickers ALL \
    --results-dir results/sensitivity/cost_0bps --cost-bps 0 --no-figures
.venv/bin/python scripts/cf_llm_experiment.py --tag llama --parts faith --n-faith 40
.venv/bin/python scripts/cf_llm_experiment.py --tag llama --parts dice --n-dice 12 --max-calls 40
.venv/bin/python scripts/cf_llm_experiment.py --compare     # Fisher's test: Llama vs Qwen own-sentence faithfulness

# 4. Web interface
.venv/bin/python scripts/run_app.py             # opens http://localhost:8501
```

Outputs land in `results/`: `summary_table.csv`, `<tag>_recommendations.csv`, `<tag>_equity.csv`, `calibration_<tag>.json`, `calibration_analysis_<tag>.csv` (post-hoc), `baselines_equity.csv`, `random_null.csv`, `ablation.csv`, `figures/*.png|svg`, and `run_manifest.json` (config, seed, package versions, git hash). LLM answers are memoised in `results/llm_cache_<tag>.jsonl`, so every number in the report can be recomputed without the models.

## Tests

```bash
.venv/bin/python -m pytest          # 354 tests, all offline: HTTP mocked, fake advisors, synthetic prices
.venv/bin/ruff check .
```

Every component has its own test module: data loader, features, schema, metrics, calibration (temperature and Platt), post-hoc calibration replay, policy adherence, counterfactual search, portfolio simulator, baselines, walk-forward backtest, evaluation runner, LLM client, cache, heuristic advisor, and the Streamlit page (rendered with `streamlit.testing`).

## Where each part of the report lives

| Report section | Code |
|---|---|
| 4.2 Data and features | `advisor/data/loader.py`, `advisor/features/indicators.py` |
| 4.3 The LLM advisor (prompt, client, cache, rule baseline) | `advisor/recommender/prompts.py`, `llm.py`, `cache.py`, `heuristic.py` |
| 4.4 Calibration | `advisor/calibration/temperature.py`, `platt.py`; metrics in `advisor/evaluation/metrics.py` |
| 4.5 Counterfactual search | `advisor/counterfactual/dice.py` |
| 4.6 Backtest, simulator, baselines, Deflated Sharpe | `advisor/evaluation/backtest.py`, `portfolio.py`, `baselines.py`, `metrics.py` |
| 4.7 Experiment runner | `advisor/evaluation/run_evaluation.py` |
| 4.8 Web interface | `streamlit_app.py` |
| 4.9 Tests | `tests/` (one file per module) |
| 5.2 Bootstrap intervals, per-year Sharpe | `advisor/evaluation/sharpe_difference.py` |
| 5.3 Policy adherence and RSI claims | `advisor/evaluation/policy_adherence.py` |
| 5.4 Post-hoc calibration replay | `advisor/evaluation/calibration_analysis.py` |
| 5.5 Counterfactual experiments | `scripts/cf_llm_experiment.py` |
| Figures 1–3 | `scripts/make_design_figures.py`, `scripts/make_cf_figure.py` |

## Repository layout

```
code/
├── advisor/
│   ├── config.py             # paths, date windows, model tags, seed
│   ├── data/                 # loader (yfinance + parquet cache), universe
│   ├── features/             # technical indicators
│   ├── recommender/          # schema, prompts, OllamaAdvisor, HeuristicAdvisor, CachedAdvisor
│   ├── calibration/          # TemperatureScaler, PlattScaler, reliability diagrams
│   ├── counterfactual/       # DiCE-style search
│   ├── evaluation/           # metrics, portfolio simulator, baselines, backtest, run_evaluation,
│   │                         # calibration_analysis, policy_adherence, sharpe_difference (post-hoc)
│   └── reproduce.py          # offline reproduction entry point
├── scripts/                  # fetch_data.py, run_app.py, cf_llm_experiment.py, make_cf_figure.py
├── streamlit_app.py          # non-technical web UI
├── tests/                    # pytest suite
├── results/                  # generated tables, figures, caches (LLM caches are committed)
└── requirements.txt, pyproject.toml, LICENSE (MIT)
```

## Requirements

Python 3.11, macOS/Linux, ~6 GB free disk for the two models, 16 GB RAM recommended for 8B models. Set `OLLAMA_BASE_URL` / `OLLAMA_MODEL` to point at another Ollama host or model.

## Reproducibility notes

Fixed seed (42), temperature 0 decoding with a fixed seed, pinned model tags, cached market data (dated snapshot) and cached model responses, a locked evaluation protocol (`../docs/preregistration.md`), and a run manifest. Exact LLM outputs can differ across hardware and Ollama versions; the cached responses make the reported numbers reproducible regardless.

## Citation

Thabet, A. (2026). *Financial Advisor Bot: a calibrated, explainable stock-recommendation system for non-expert investors.* BSc Computer Science final project report, University of London.
