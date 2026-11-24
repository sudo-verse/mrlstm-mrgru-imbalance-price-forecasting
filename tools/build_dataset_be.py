"""Build the Belgian (Elia) imbalance dataset, 15-min, UTC, 2022-01-01 -> 2026-10-01.

    ENTSOE_TOKEN=... python build_dataset_be.py      -> imbalance_data_BE.csv  (+ raw/ cache)

Sources (Elia Open Data, opendata.elia.be; ENTSO-E Transparency for the day-ahead price).
Elia changed its balancing design on 22/05/2024 (MARI go-live), so most series come from two datasets:
                               up to 22/05/2024        from 22/05/2024
  imbalance price, SI, MIP/MDP, alpha   ods047                  ods134
  activated volumes (aFRR/mFRR/IGCC)    ods063                  ods127
  activation prices (aFRR/mFRR)         ods064                  ods166
  load (actual, day-ahead forecast)     ods001
  wind (measured, day-ahead forecast)   ods031   (sum of all regions / connection types)
  solar (measured, day-ahead forecast)  ods032   (region = Belgium)
  day-ahead price BE                    ENTSO-E A44, 10YBE----------2

Conventions (chosen to match the Austrian dataset):
  system_imbalance   = -SI_Elia   (MW, + = system short / deficit; Elia's SI is + when long)
  E_* volumes        = MW averaged over the quarter-hour, always >= 0 (direction in the name)
  imbalance_price    = Elia single imbalance price (EUR/MWh)
"""
import io, os, sys, time, zipfile, urllib.request, urllib.error, urllib.parse
import numpy as np, pandas as pd

D = os.path.dirname(os.path.abspath(__file__)); RAWD = os.path.join(D, "raw"); os.makedirs(RAWD, exist_ok=True)
START, END = "2022-01-01T00:00:00Z", "2026-10-01T00:00:00Z"
ELIA = "https://opendata.elia.be/api/explore/v2.1/catalog/datasets/{ds}/exports/csv?"


def http(url, tries=8):
    for a in range(tries):
        try:
            return urllib.request.urlopen(url, timeout=300).read()
        except (urllib.error.URLError, TimeoutError) as e:
            print("   retry", a + 1, type(e).__name__, flush=True); time.sleep(10 * (a + 1))
    raise RuntimeError(url)


def elia(ds, fields, where_extra=None, chunks="YS"):
    """Download dataset `ds` (selected fields) for START..END in yearly chunks; cached in raw/."""
    path = os.path.join(RAWD, f"{ds}.csv")
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["datetime"])
    parts = []
    edges = list(pd.date_range(START, END, freq=chunks, tz="UTC")) + [pd.Timestamp(END)]
    edges = sorted(set([pd.Timestamp(START)] + edges))
    for s, e in zip(edges[:-1], edges[1:]):
        where = f'datetime >= "{s:%Y-%m-%dT%H:%M:%S}Z" and datetime < "{e:%Y-%m-%dT%H:%M:%S}Z"'
        if where_extra:
            where += " and " + where_extra
        q = urllib.parse.urlencode({"select": ",".join(["datetime"] + fields), "where": where,
                                    "delimiter": ",", "timezone": "UTC"})
        raw = http(ELIA.format(ds=ds) + q)
        df = pd.read_csv(io.BytesIO(raw))
        print(f"  {ds} {s:%Y-%m} -> {e:%Y-%m}: {len(df)} rows", flush=True)
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df.to_csv(path, index=False)
    return df


def qh(df, how="mean"):
    """Aggregate to one row per quarter-hour (UTC)."""
    df = df.copy()
    for c in df.columns:                                   # empty yearly chunks make columns 'object'; force numeric
        if c != "datetime":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    g = df.groupby("datetime")
    return (g.sum(min_count=1) if how == "sum" else g.mean(numeric_only=True)).sort_index()


def entsoe_da_be():
    path = os.path.join(RAWD, "da_price_BE.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0, parse_dates=True)["P_DA"]
    from lxml import etree
    tok = os.environ["ENTSOE_TOKEN"]; rows = []
    months = pd.date_range(START, END, freq="MS", tz="UTC")
    for s, e in zip(months[:-1], months[1:]):
        url = (f"https://web-api.tp.entsoe.eu/api?securityToken={tok}&documentType=A44&in_Domain=10YBE----------2"
               f"&out_Domain=10YBE----------2&periodStart={s:%Y%m%d%H%M}&periodEnd={e:%Y%m%d%H%M}")
        raw = http(url)
        if raw[:2] == b"PK":
            docs = [zipfile.ZipFile(io.BytesIO(raw)).read(n) for n in zipfile.ZipFile(io.BytesIO(raw)).namelist()]
        else:
            docs = [raw]
        for doc in docs:
            t = etree.fromstring(doc)
            for ts in t.iter("{*}TimeSeries"):
                # A44 may contain several classification sequences; keep the default one (position 1 or absent)
                seq = ts.findtext("{*}classificationSequence_AttributeInstanceComponent.position")
                if seq not in (None, "1"):
                    continue
                for per in ts.iter("{*}Period"):
                    st = pd.Timestamp(per.findtext("{*}timeInterval/{*}start"))
                    en = pd.Timestamp(per.findtext("{*}timeInterval/{*}end"))
                    res = pd.Timedelta(per.findtext("{*}resolution").replace("PT", "").replace("M", "min"))
                    pts = {int(p.findtext("{*}position")): float(p.findtext("{*}price.amount")) for p in per.iter("{*}Point")}
                    n = int((en - st) / res); last = None
                    for i in range(1, n + 1):          # A44 uses curve type A03: missing positions repeat the previous value
                        last = pts.get(i, last)
                        rows.append((st + (i - 1) * res, last, res))
        print(f"  DA BE {s:%Y-%m}: total points {len(rows)}", flush=True)
        time.sleep(0.5)
    df = pd.DataFrame(rows, columns=["t", "P", "res"]).drop_duplicates("t", keep="last").set_index("t").sort_index()
    s = df["P"].resample("15min").ffill()                # hourly products before 1 Oct 2025 -> each quarter-hour
    s.name = "P_DA"; s.to_frame().to_csv(path)
    return s


if __name__ == "__main__":
    idx = pd.date_range(START, END, freq="15min", tz="UTC", inclusive="left")
    out = pd.DataFrame(index=idx)
    cut = pd.Timestamp("2024-05-21T22:00:00Z")           # first quarter-hour of the MARI-era datasets

    # ---- imbalance price, SI, MIP/MDP, alpha
    a = qh(elia("ods047", ["systemimbalance", "alpha", "marginalincrementalprice", "marginaldecrementalprice",
                           "positiveimbalanceprice", "negativeimbalanceprice"]))
    b = qh(elia("ods134", ["systemimbalance", "ace", "alpha", "alpha_prime", "marginalincrementalprice",
                           "marginaldecrementalprice", "imbalanceprice"]))
    a["imbalanceprice"] = a["positiveimbalanceprice"]
    p = pd.concat([a[a.index < cut], b[b.index >= cut]])
    out["imbalance_price"] = p["imbalanceprice"]
    out["system_imbalance"] = -p["systemimbalance"]       # + = short
    out["MIP"] = p["marginalincrementalprice"]; out["MDP"] = p["marginaldecrementalprice"]
    out["alpha"] = p["alpha"]

    # ---- activated volumes (MW, >= 0)
    v1 = qh(elia("ods063", ["igccvolumeup", "igccvolumedown", "r2volumeup", "r2volumedown", "bidsvolumeup",
                            "bidsvolumedown", "r3stdvolume", "r3flexvolume", "intertsovolumeimport", "intertsovolumeexport"]))
    v2 = qh(elia("ods127", ["igccvolumeup", "igccvolumedown", "afrrvolumeup", "afrrvolumedown", "mfrrsaup", "mfrrdaup",
                            "mfrrsadown", "mfrrdadown", "reserve_sharing_import", "reserve_sharing_export"]))
    old = pd.DataFrame({"E_aFRR_pos": v1.r2volumeup, "E_aFRR_neg": v1.r2volumedown,
                        "E_mFRR_pos": v1[["bidsvolumeup", "r3stdvolume", "r3flexvolume", "intertsovolumeimport"]].sum(1, min_count=1),
                        "E_mFRR_neg": v1[["bidsvolumedown", "intertsovolumeexport"]].sum(1, min_count=1),
                        "IGCC_up": v1.igccvolumeup, "IGCC_down": v1.igccvolumedown})
    new = pd.DataFrame({"E_aFRR_pos": v2.afrrvolumeup, "E_aFRR_neg": v2.afrrvolumedown,
                        "E_mFRR_pos": v2[["mfrrsaup", "mfrrdaup", "reserve_sharing_import"]].sum(1, min_count=1),
                        "E_mFRR_neg": v2[["mfrrsadown", "mfrrdadown", "reserve_sharing_export"]].sum(1, min_count=1),
                        "IGCC_up": v2.igccvolumeup, "IGCC_down": v2.igccvolumedown})
    vol = pd.concat([old[old.index < cut], new[new.index >= cut]]).abs()
    for c in vol:
        out[c] = vol[c]

    # ---- activation prices (EUR/MWh)
    q1 = qh(elia("ods064", ["r2priceup", "r2pricedown", "bidspriceup", "bidspricedown", "igccpriceup", "igccpricedown"]))
    q2 = qh(elia("ods166", ["afrrpriceup", "afrrpricedown", "mfrrsaup", "mfrrdaup", "mfrrsadown", "mfrrdadown",
                            "floorprice", "cap"]))
    oldp = pd.DataFrame({"P_aFRR_pos": q1.r2priceup, "P_aFRR_neg": q1.r2pricedown,
                         "P_mFRR_pos": q1.bidspriceup, "P_mFRR_neg": q1.bidspricedown})
    newp = pd.DataFrame({"P_aFRR_pos": q2.afrrpriceup, "P_aFRR_neg": q2.afrrpricedown,
                         "P_mFRR_pos": q2[["mfrrsaup", "mfrrdaup"]].max(1), "P_mFRR_neg": q2[["mfrrsadown", "mfrrdadown"]].min(1),
                         "P_floor": q2.floorprice, "P_cap": q2.cap})
    pr = pd.concat([oldp[oldp.index < cut], newp[newp.index >= cut]])
    for c in pr:
        out[c] = pr[c]

    # ---- load, wind, solar (MW): actual + day-ahead forecast
    ld = qh(elia("ods001", ["totalload", "dayaheadforecast"]))
    out["load_act"], out["load_fc"] = ld.totalload, ld.dayaheadforecast
    w = qh(elia("ods031", ["measured", "dayaheadforecast"]), how="sum")
    out["wind_act"], out["wind_fc"] = w.measured, w.dayaheadforecast
    pv = qh(elia("ods032", ["measured", "dayaheadforecast"], where_extra='region = "Belgium"'))
    out["solar_act"], out["solar_fc"] = pv.measured, pv.dayaheadforecast

    # ---- free substitutes for the (licensed) intraday price index
    # reserve bid ladder: price of the next available aFRR / mFRR bid and remaining volumes (from 22/05/2024)
    ap = qh(elia("ods153", ["upwardavailableafrrprice", "upwardavailablemfrrprice",
                            "downwardavailableafrrprice", "downwardavailablemfrrprice"]))
    av = qh(elia("ods152", ["upwardavailabletotalvol", "downwardavailabletotalvol"]))
    out["avail_aFRR_up_P"], out["avail_mFRR_up_P"] = ap.upwardavailableafrrprice, ap.upwardavailablemfrrprice
    out["avail_aFRR_dn_P"], out["avail_mFRR_dn_P"] = ap.downwardavailableafrrprice, ap.downwardavailablemfrrprice
    out["avail_up_vol"], out["avail_dn_vol"] = av.upwardavailabletotalvol, av.downwardavailabletotalvol
    # intraday market coupling: Belgium's implicit net position (MW) - intraday trading activity / liquidity
    out["ID_netpos"] = qh(elia("ods022", ["implicitnetposition"])).implicitnetposition

    # ---- day-ahead price
    out["P_DA"] = entsoe_da_be().reindex(out.index)

    out.index.name = "Time [UTC] start"
    out.to_csv(os.path.join(D, "imbalance_data_BE.csv"))
    print("saved", out.shape)
    print((100 * out.notna().mean()).round(2).to_string())
    print(out.describe().T[["mean", "min", "max"]].round(1).to_string())
