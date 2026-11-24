# Data provenance

| Market | Target | Main sources | Frequency |
|---|---|---|---|
| Austria | APG imbalance price | APG, ENTSO-E Transparency, IGCC/netting information | 15 minutes |
| Belgium | Elia imbalance price | Elia Open Data, ENTSO-E Transparency | 15 minutes |

For Belgium, `tools/build_dataset_be.py` documents the Elia dataset identifiers used before and after the May 2024 MARI redesign. `tools/clean_be.py` makes the missing-value and activation-price conventions explicit.

Do not commit provider downloads, API tokens or personally restricted datasets. Store them outside version control and supply their paths through `--data`.
