# Pre-registration of evaluation hypotheses — Financial Advisor Bot (CM3070)

**Author:** Aly Thabet · **Written:** 2026-09-14 19:10 EEST, *before* any LLM-backed backtest was run on the held-out window.
**Locked protocol.** Universe: the 51 US large-cap tickers in `advisor/data/universe.py`. Held-out test window: 2025-01-01 → 2026-06-01 (never used during development; the calibration prototype in the preliminary report used 2018–2024 data and an RSI rule only). Decision dates: last trading day of each week (`W-FRI`). Portfolio rule: equal-weight long every ticker whose action is BUY (HOLD keeps a name already held, SELL removes it), cash otherwise; 10 bps transaction cost on turnover. Correctness label: 5-trading-day forward return sign (HOLD counted correct if |return| < 1%). Calibration: temperature scaling refitted each calendar-month fold on all past, already-resolved recommendations (minimum 50), never on future data. Baselines: S&P 500 buy-and-hold (SPY), seeded random allocation (100 seeds → null distribution), 12-1 momentum, long-only Markowitz mean-variance (Ledoit-Wolf shrinkage). Models: Llama 3.1 8B-Instruct (`llama3.1:8b`, primary, as promised in the proposal) and Qwen3 8B (`qwen3:8b`, the comparison requested by the preliminary-report examiner), both via Ollama at temperature 0 with a fixed seed; prompt version fixed in `advisor/recommender/prompts.py`. Number of strategies compared for the Deflated Sharpe ratio: 3 advisors (Llama, Qwen, rule-based ablation) + 4 baselines = 7 trials. No other configurations will be tried; if any change becomes necessary it will be recorded here with its date and reason.

## Hypotheses

- **H1 (profitability):** each LLM advisor's Deflated Sharpe ratio on the held-out window exceeds that of S&P 500 buy-and-hold, and its Sharpe ratio lies above the 95th percentile of the 100-seed random-allocation null distribution.
- **H2 (calibration):** for each LLM advisor, walk-forward temperature scaling lowers Expected Calibration Error relative to the raw verbal confidence, *without* collapsing sharpness (mean |p − 0.5|) below half of its raw value; the Brier score does not worsen.
- **H3 (LLM ablation):** removing the LLM (the transparent momentum/SMA rule advisor) lowers risk-adjusted return by at least 20% (Sharpe) relative to the better LLM advisor.
- **H4 (user study, qualitative):** with 3–5 non-technical participants, the counterfactual-on interface receives higher "I understand why" and "I could sanity-check this" Likert ratings than the counterfactual-off interface; reported descriptively, no significance claim.
- **H5 (model comparison, exploratory):** Llama 3.1 8B and Qwen3 8B differ in calibration (ECE) more than in profitability; the direction is not predicted.

## What counts as a negative result
A hypothesis is reported as *not supported* if the pre-specified comparison fails; results will not be re-run with different windows, tickers, thresholds or prompts to rescue them. The Deflated Sharpe ratio will be reported with n_trials = 7 regardless of outcome.

## Post-hoc addendum — 2026-09-15 14:35 EEST (written *after* the Llama result was known)

**Status: exploratory, not a replacement for H2.** H2 is reported as *not supported* exactly as pre-registered (ECE fell, sharpness collapsed, T pinned at the 20 ceiling in every fold). The addendum records one secondary analysis added to *explain* that outcome, not to rescue it.

**Reason.** Temperature scaling is a one-parameter map that fixes p = 0.5, so it cannot represent a correctness rate below one half. Under the locked HOLD rule (|5-day return| < 1%, true ≈21% of the time) a HOLD-heavy advisor is correct ≈28% of the time, which is below the reach of any temperature. The failure is structural, not a tuning issue.

**What was added.** `advisor/calibration/platt.py` (Platt 1999: logistic regression with intercept on the confidence logit) and `advisor/evaluation/calibration_analysis.py`, which replays the *identical* monthly walk-forward protocol (expanding window, refit on labels realised strictly before each fold, minimum 50) for raw / expanding base rate / temperature / Platt on (a) all recommendations and (b) BUY and SELL only, where the HOLD rule plays no part. No model calls; the log is re-read. The temperature replay reproduces the primary numbers exactly (ECE 0.245, Brier 0.266, n = 3,723). Outputs: `results/calibration_analysis_<tag>.csv`, `..._folds.csv`, `figures/reliability_analysis_<tag>_<subset>.*`. Every number from this script is to be labelled post-hoc in the report.

## Post-hoc addendum 2 — 2026-09-23 (written *after* all pre-registered results were known)

**Status: exploratory.** None of the verdicts on H1–H5 changes. These analyses were added to explain the results and to test the robustness of the verdicts, and every number from them is labelled post-hoc in the report. No window, ticker, threshold, prompt or model setting was changed, and no model was re-run to change a result.

**What was added, and why.**
- *Policy adherence* (`advisor/evaluation/policy_adherence.py`). The prompt states an exact decision policy, so the action it implies can be recomputed for every answer and compared with what each model did. This checks whether H1 and H3 describe the stated strategy or something else. The same module checks every written reason that says the RSI "is above 70" or "is below 30" against the real value (false only if more than 2 points out).
- *Counterfactual experiments* (`scripts/cf_llm_experiment.py`). Each model's own "what would change my mind" sentence was tested on a seeded sample of 40 BUY/SELL answers, by moving the named indicator just past its stated threshold and re-asking the model. The verified search was run on the first 12 of those answers with the interface's 40-call budget, and every counterfactual it returned was re-checked with a fresh call. This supports the counterfactual-validity row of the evaluation design, which was planned but not given a sample size in advance.
- *Robustness of the Sharpe comparisons* (`advisor/evaluation/sharpe_difference.py`). Paired circular block-bootstrap confidence intervals for Sharpe differences (10,000 resamples, 10-day blocks, seed 42), and Sharpe ratios per calendar year.
- *Cost sensitivity*: the simulator re-run from the cached answers at 0 and 20 bps (`results/sensitivity/`), to see whether H1 and H3 depend on the pre-registered 10 bps.
