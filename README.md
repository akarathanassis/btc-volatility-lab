# btc-volatility-lab

A Bitcoin volatility modelling lab aimed at:

- Modelling and forecasting BTC volatility
- Serving a visualisation dashboard of key volatility metrics

## Structure

- `data/` – raw and processed BTC price / order book / implied vol data
- `notebooks/` – exploratory analysis and research
- `src/`
  - `data/loader.py` – data loading utilities
  - `features/volatility.py` – volatility feature engineering (RV, Garman-Klass, etc.)
  - `models/garch.py` – classical GARCH-style volatility models
- `app/dashboard.py` – main visualisation dashboard (e.g. Streamlit)
- `tests/` – unit tests for loaders, features, and models
