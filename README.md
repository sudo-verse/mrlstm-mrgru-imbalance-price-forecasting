# Market-Rule-Informed GRU/LSTM for Imbalance-Price Forecasting

Reproducible code for our Final Year Project on **day-ahead probabilistic imbalance-price forecasting**. The models forecast the next 96 quarter-hours (24 hours) as five non-crossing quantiles: q10, q25, q50, q75 and q90.

The project evaluates Austria (APG) and Belgium (Elia), and compares baseline market-rule-informed recurrent models with modified versions that add market-informed features.

> This is a student research implementation, built on the model concept introduced by Yu et al. (2026). It is not an official implementation of the original MRINN authors.

## What is included

```text
.
├── src/
│   ├── austria_baseline_mrgru_mrlstm.py
│   ├── austria_modified_mrgru_mrlstm.py
│   ├── belgium_baseline_mrgru_mrlstm.py
│   └── belgium_modified_mrgru_mrlstm.py
├── tools/
│   ├── build_dataset_be.py          # downloads/builds the Belgium source dataset
│   └── clean_be.py                  # deterministic cleaning step for Belgium
├── data/
│   └── README.md                    # schema, provenance, and access instructions
├── docs/
│   ├── DATA.md
│   └── REPRODUCIBILITY.md
├── results/
│   └── README.md                    # reported benchmark metrics
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

Processed datasets and training outputs are intentionally excluded from Git. See [data/README.md](data/README.md) before running an experiment.

## Installation

Python 3.10 or 3.11 is recommended.

```bash
git clone https://github.com/sudo-verse/mrlstm-mrgru-imbalance-price-forecasting.git
cd mrlstm-mrgru-imbalance-price-forecasting
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick start

Place the requested CSV in `data/`, then run one small smoke experiment:

```bash
python src/austria_modified_mrgru_mrlstm.py \
  --data data/austria_imbalance_dataset.csv \
  --cell gru --seeds 42 --epochs 2 --out results/smoke_at_gru
```

For the full reported protocol, use three seeds and the default 120-epoch cap:

```bash
python src/austria_modified_mrgru_mrlstm.py \
  --data data/austria_imbalance_dataset.csv \
  --cell gru --seeds 42 43 44 --out results/austria_modified_gru
```

Each seed evaluates four rolling folds: 12 months training, 3 months validation and 3 months test. A full CPU run can take several hours.

## Models

| Market | Baseline | Modified model |
|---|---|---|
| Austria | MR-GRU / MR-LSTM with differentiable reserve-energy, exchange-reference and scarcity blocks | Fusion width 32, IGCC netting-aware reserve-energy block, and leak-free day-ahead AT/DE load, wind and solar forecast features |
| Belgium | MR-GRU / MR-LSTM with marginal MIP/MDP and incentive-alpha settlement block | Free Elia reserve-bid-ladder and intraday-net-position signals |

The Austrian modified model includes German load, wind and solar forecasts because Austrian balancing conditions are coupled to the wider Central European electricity system. Germany is an explanatory input; the prediction target remains the Austrian imbalance price.

## Results

The reported metrics are the mean of seeds 42, 43 and 44. Lower is better for AQL, MAE, RMSE, MASE and AQCR.

| Model | Austria AQL | Austria MAE | Belgium AQL | Belgium MAE |
|---|---:|---:|---:|---:|
| MR-GRU baseline | 25.89 | 66.27 | 17.85 | 45.36 |
| MR-GRU modified | **25.58** | **64.66** | **17.68** | **44.59** |
| MR-LSTM baseline | 25.85 | 66.49 | 17.86 | 45.97 |
| MR-LSTM modified | **25.57** | **64.57** | **17.66** | **44.92** |

See [results/README.md](results/README.md) for the complete benchmark tables and [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for evaluation details.

## Data and responsible use

The repository does **not** redistribute the project’s processed CSV files. They contain material compiled from APG, Elia and ENTSO-E sources, and use of those sources is governed by their own terms. The Belgium build utility requires an ENTSO-E API token supplied through `ENTSOE_TOKEN`; never commit it.

## Citation

If this implementation helps your work, cite the original MRINN paper and this repository. Citation metadata is in [CITATION.cff](CITATION.cff).

```bibtex
@article{yu2026mrinn,
  title={A Market-Rule-Informed Neural Network for Efficient Imbalance Electricity Price Forecasting},
  author={Yu, Runyao and others},
  journal={Advanced Engineering Informatics},
  year={2026},
  doi={10.1016/j.aei.2026.105083}
}
```

## License

The original code in this repository is released under the MIT License. External datasets, APIs and third-party software retain their own licenses and terms.
