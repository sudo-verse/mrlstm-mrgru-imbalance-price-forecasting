# Reproducibility protocol

## Forecasting task

At every 15-minute origin, predict the next 96 quarter-hours of imbalance price. Each model emits five quantiles: 0.10, 0.25, 0.50, 0.75 and 0.90.

## Splits

Austria uses the 2025 test year. Belgium uses October 2025 to September 2026. For each three-month test fold, the previous 3 months form validation data and the preceding 12 months form training data.

## Seeds and metrics

Use seeds `42 43 44`. Each script writes prediction archives and per-seed metrics. Report the mean across seeds.

* **AQL**: average quantile (pinball) loss across all five quantiles and horizons.
* **MAE/RMSE**: computed from the median forecast.
* **MASE**: scaled against the seasonal-naive same-time-yesterday error.
* **AQCR**: quantile-crossing rate; the hierarchical output head is designed to keep it at zero.

## Leakage safeguards

Scalers and rule constants are fitted only to training windows. Future day-ahead and forecast features are masked until their actual publication times. Do not replace these availability masks when comparing results.
