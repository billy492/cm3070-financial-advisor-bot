"""Tradable universe definition for the financial advisor bot.

Defines a fixed, reproducible universe of large-cap US equities grouped by
broad GICS-style sectors. Keeping the universe static (rather than fetching
index constituents at runtime) makes every backtest and recommendation run
deterministic and offline-reproducible, which is a prerequisite for the
walk-forward evaluation described in ADR-0002.

The universe is intentionally large-cap-only: such names have deep, liquid
price histories on yfinance going back well before ``TRAIN_START`` (2020-01-01),
which avoids survivorship/listing-gap artefacts that would bias the backtest.

Public symbols
--------------
SECTORS : dict[str, list[str]]
    Mapping of sector name -> list of ticker symbols (~50 tickers total
    across six sectors).
UNIVERSE : list[str]
    Flattened, de-duplicated, sorted list of all tickers in ``SECTORS``.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Sector -> ticker mapping.
#
# Six broad sectors covering ~50 large-cap US tickers. Symbols use the exact
# yfinance/Yahoo Finance spelling (e.g. "BRK-B" with a hyphen for the Berkshire
# Hathaway class-B share). All names trade on NYSE/NASDAQ with continuous
# history predating TRAIN_START (2020-01-01).
# ---------------------------------------------------------------------------
SECTORS: dict[str, list[str]] = {
    "technology": [
        "AAPL",   # Apple
        "MSFT",   # Microsoft
        "NVDA",   # NVIDIA
        "AVGO",   # Broadcom
        "ORCL",   # Oracle
        "CRM",    # Salesforce
        "ADBE",   # Adobe
        "CSCO",   # Cisco
        "INTC",   # Intel
        "QCOM",   # Qualcomm
    ],
    "healthcare": [
        "JNJ",    # Johnson & Johnson
        "UNH",    # UnitedHealth Group
        "LLY",    # Eli Lilly
        "PFE",    # Pfizer
        "MRK",    # Merck
        "ABBV",   # AbbVie
        "TMO",    # Thermo Fisher Scientific
        "ABT",    # Abbott Laboratories
        "AMGN",   # Amgen
    ],
    "financials": [
        "JPM",    # JPMorgan Chase
        "BAC",    # Bank of America
        "WFC",    # Wells Fargo
        "GS",     # Goldman Sachs
        "MS",     # Morgan Stanley
        "V",      # Visa
        "MA",     # Mastercard
        "BRK-B",  # Berkshire Hathaway (class B)
        "BLK",    # BlackRock
    ],
    "industrials": [
        "CAT",    # Caterpillar
        "BA",     # Boeing
        "HON",    # Honeywell
        "GE",     # GE Aerospace
        "UPS",    # United Parcel Service
        "LMT",    # Lockheed Martin
        "DE",     # Deere & Co.
        "RTX",    # RTX (Raytheon)
    ],
    "consumer": [
        "AMZN",   # Amazon
        "TSLA",   # Tesla
        "HD",     # Home Depot
        "MCD",    # McDonald's
        "NKE",    # Nike
        "KO",     # Coca-Cola
        "PG",     # Procter & Gamble
        "WMT",    # Walmart
        "COST",   # Costco
        "PEP",    # PepsiCo
    ],
    "energy": [
        "XOM",    # ExxonMobil
        "CVX",    # Chevron
        "COP",    # ConocoPhillips
        "SLB",    # Schlumberger
        "EOG",    # EOG Resources
    ],
}


def _build_universe(sectors: dict[str, list[str]]) -> list[str]:
    """Flatten a sector mapping into a sorted, de-duplicated ticker list.

    Args:
        sectors: Mapping of sector name to a list of ticker symbols.

    Returns:
        Every ticker appearing in ``sectors``, de-duplicated and sorted in
        ascending lexicographic order for deterministic iteration.
    """
    seen: set[str] = set()
    for tickers in sectors.values():
        seen.update(tickers)
    return sorted(seen)


#: Flattened, de-duplicated, sorted list of all tickers in :data:`SECTORS`.
UNIVERSE: list[str] = _build_universe(SECTORS)
