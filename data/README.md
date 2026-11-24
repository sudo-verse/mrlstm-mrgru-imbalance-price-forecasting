# Data access and schema

The processed CSV files used for the experiments are intentionally not included in this repository. This prevents accidental redistribution of data supplied to the project and protects API credentials.

## Austria

The model expects `data/austria_imbalance_dataset.csv`, sampled at 15 minutes, with `Time [UTC] start` and `imbalance_price` plus the market, reserve-energy, intraday, day-ahead and forecast columns used by the scripts.

The original project dataset combines APG/ENTSO-E source material and IGCC netting data. Obtain the underlying data from the relevant provider under its terms, reproduce the feature engineering, then pass the resulting CSV path with `--data`.

## Belgium

The reusable public-data pipeline is included:

```bash
export ENTSOE_TOKEN='your-entsoe-api-token'
python tools/build_dataset_be.py
python tools/clean_be.py
```

This writes `tools/imbalance_data_BE_clean.csv`. Copy or rename it to `data/belgium_imbalance_dataset.csv` before running a Belgium model.

The builder downloads Elia Open Data and ENTSO-E day-ahead prices. Observe both providers’ terms, attribution requirements and request limits.

## Data availability rule

All experiments must respect publication time. In particular, day-ahead prices for day D+1 are masked until their publication time on day D. The scripts enforce the masks used in the reported evaluation.
