# Reported results

These are the project’s completed three-seed results. Metrics are in EUR/MWh except MASE and AQCR. Lower is better.

## Austria (test year 2025)

| Model | AQL | MAE | RMSE | MASE | AQCR % | Parameters |
|---|---:|---:|---:|---:|---:|---:|
| MR-GRU baseline | 25.89 | 66.27 | 389.61 | 0.687 | 0.00 | 5,055 |
| MR-GRU modified | 25.58 | 64.66 | 389.24 | 0.670 | 0.00 | 10,343 |
| MR-LSTM baseline | 25.85 | 66.49 | 389.71 | 0.689 | 0.00 | 5,895 |
| MR-LSTM modified | 25.57 | 64.57 | 389.24 | 0.669 | 0.00 | 11,687 |

## Belgium (test window October 2025 to September 2026)

| Model | AQL | MAE | RMSE | MASE | AQCR % | Parameters |
|---|---:|---:|---:|---:|---:|---:|
| MR-GRU baseline | 17.845 | 45.357 | 195.653 | 0.697 | 0.00 | 3,793 |
| MR-GRU modified | 17.678 | 44.592 | 194.945 | 0.686 | 0.00 | 6,089 |
| MR-LSTM baseline | 17.863 | 45.969 | 195.774 | 0.707 | 0.00 | 4,409 |
| MR-LSTM modified | 17.655 | 44.921 | 195.176 | 0.691 | 0.00 | 7,097 |

Screening comparisons against LQR, MLP, AttnBiLSTM, PatchTST, iTransformer and MRINN use the same leak-free evaluation pipeline, but a 20-epoch cap. Do not claim their ranking is a fully tuned-model comparison.
