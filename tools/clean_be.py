"""imbalance_data_BE.csv -> imbalance_data_BE_clean.csv (gap handling; see build_dataset_be.py for sources)."""
import os, pandas as pd
D = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(os.path.join(D, "imbalance_data_BE.csv"), index_col=0, parse_dates=True)
vol = ["E_aFRR_pos", "E_aFRR_neg", "E_mFRR_pos", "E_mFRR_neg", "IGCC_up", "IGCC_down"]
d[vol] = d[vol].fillna(0.0)                                    # nothing published = nothing activated
for p, e in (("P_aFRR_pos", "E_aFRR_pos"), ("P_aFRR_neg", "E_aFRR_neg"), ("P_mFRR_pos", "E_mFRR_pos"), ("P_mFRR_neg", "E_mFRR_neg")):
    d[p + "_active"] = (d[e] > 0).astype(int)
    d[p] = d[p].fillna(0.0)                                    # weight 0 in the rule when nothing activated
d["solar_act"] = d["solar_act"].interpolate(limit=8)
post = d.index >= pd.Timestamp("2024-05-21 22:00", tz="UTC")   # bid-ladder data exists only after MARI go-live
for c in ["avail_aFRR_up_P", "avail_mFRR_up_P", "avail_aFRR_dn_P", "avail_mFRR_dn_P"]:
    d.loc[post, c] = d.loc[post, c].ffill(limit=8).fillna(d.loc[post, c].median())
for c in ["avail_up_vol", "avail_dn_vol", "ID_netpos"]:
    d[c] = d[c].fillna(0.0)
d.to_csv(os.path.join(D, "imbalance_data_BE_clean.csv"))
print(d.shape)
print((100 * d.loc[post].notna().mean()).round(2).to_string())
