"""Austria - MODIFIED MR-GRU / MR-LSTM (corrected pipeline).

Day-ahead probabilistic forecasting of the Austrian (APG) imbalance price: every 15 minutes the model forecasts
the next 96 quarter-hours (24 h) as five quantiles (q10, q25, q50, q75, q90).

Model: MR-GRU / MR-LSTM - one GRU (or LSTM) encoder per input signal, the Austrian imbalance-settlement rules
(MRINN, Yu et al. 2026) applied as differentiable operators in the latent space, a known-future path
(day-ahead price + calendar) and a hierarchical quantile head (quantiles cannot cross).

Modified, three changes over the baseline:
  1. fusion layer widened from 8 to 32 units;
  2. day-ahead load / wind / solar forecasts of Austria and Germany (+ residual load) as known-future inputs,
     and their past forecast errors as history inputs (ENTSO-E);
  3. netting-aware reserve-energy block: IGCC netting volumes (ENTSO-E) -> learned netting share ->
     residual activation -> merit-order depth term and branch gate (Objective 1).
Parameters: 10,343 (GRU) / 11,687 (LSTM).

Evaluation: 4 rolling folds (12 months train / 3 validation / 3 test) covering the 2025 test year; leak-free
(day-ahead prices for D+1 are hidden until they are published at ~12:45 on day D).

Usage
    python austria_modified_mrgru_mrlstm.py --data austria_imbalance_dataset.csv --cell gru --seeds 42 43 44
    python austria_modified_mrgru_mrlstm.py --data austria_imbalance_dataset.csv --cell lstm --seeds 42 43 44
Outputs (in --out, default ./results): per-seed predictions (.npz) and a metrics table (.csv)
Requirements: Python >= 3.10, tensorflow >= 2.16 (Keras 3), numpy, pandas, scikit-learn
"""
import argparse, os, random, time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, initializers
from keras import ops as K
from sklearn.preprocessing import RobustScaler

# ================================================================== settings
TIME_COL, LABEL, DA_COL = "Time [UTC] start", "imbalance_price", "P_DA_nemo"
HORIZONS, LOOKBACK, HIDDEN = 96, 32, 8                  # 96 quarter-hours ahead, 8 h of history, 8 latent units
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
TEST_START, TEST_END = "2025-01-01 00:00:00+00:00", "2026-01-01 00:00:00+00:00"
FUSION_WIDTH = 32                                # width of the layer combining rule output and known future
DA_DROPOUT = 0.2                                         # blank the whole day-ahead input on 20% of training samples
EPOCHS, PATIENCE, BATCH, LR = 120, 12, 1024, 1e-3

FEATS_PRICES = ["P_aFRR_pos", "P_mFRR_pos", "P_aFRR_neg", "P_mFRR_neg", "P_aFRR_pos_MOL", "P_aFRR_neg_MOL",
                "P_ID15_nemo", "P_ID60_nemo", "P_DA_nemo"]
FEATS_CAPACITIES = ["L_ID15", "L_ID60", "L_DA"]
FEATS_VOLUME = ["system_imbalance", "E_aFRR_pos", "E_mFRR_pos", "E_aFRR_neg", "E_mFRR_neg"]
# the 17 history signals, in the order the rule blocks use them (column, model input name)
BASE_SPEC = [("system_imbalance", "system_imbalance"),
             ("E_aFRR_pos", "E_aFRR_pos_in"), ("E_mFRR_pos", "E_mFRR_pos_in"),
             ("P_aFRR_pos", "P_aFRR_pos_in"), ("P_mFRR_pos", "P_mFRR_pos_in"),
             ("E_aFRR_neg", "E_aFRR_neg_in"), ("E_mFRR_neg", "E_mFRR_neg_in"),
             ("P_aFRR_neg", "P_aFRR_neg_in"), ("P_mFRR_neg", "P_mFRR_neg_in"),
             ("P_aFRR_pos_MOL", "P_VoAA_pos_in"), ("P_aFRR_neg_MOL", "P_VoAA_neg_in"),
             ("P_ID15_nemo", "P_ID15_nemo_in"), ("P_ID60_nemo", "P_ID60_nemo_in"), ("P_DA_nemo", "P_DA_nemo_in"),
             ("L_ID15", "L_ID15_in"), ("L_ID60", "L_ID60_in"), ("L_DA", "L_DA_in")]
NET_SPEC = [("IN_import_MWh", "IN_import_in"), ("IN_export_MWh", "IN_export_in")]   # IGCC netting (ENTSO-E)


def set_random_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed); random.seed(seed); np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


# ================================================================== data
def load_data(path):
    """Load the dataset and prepare every column the model uses. Returns (frame, feature list, input spec, forecast columns)."""
    df = pd.read_csv(path, parse_dates=[TIME_COL]).sort_values(TIME_COL).reset_index(drop=True)
    df["L_DA"] = 0                                       # day-ahead volume is not published: constant 0, as in MRINN
    feats = FEATS_PRICES + FEATS_CAPACITIES + FEATS_VOLUME
    spec = list(BASE_SPEC)
    fc_cols = []
    df[["IN_import_MWh", "IN_export_MWh"]] = df[["IN_import_MWh", "IN_export_MWh"]].fillna(0.0)
    # day-ahead load / wind / solar forecasts (Austria + Germany): known future; their past errors: history inputs
    f = df.set_index(TIME_COL)[[c for c in df.columns if c[:3] in ("AT_", "DE_")]].interpolate(limit=8)
    fc = [c for c in f.columns if "_fc_" in c]
    for a in ("AT", "DE"):                               # residual load = load - wind - solar
        wind = [c for c in fc if c.startswith(f"{a}_ws_fc_wind")]
        f[f"{a}_fc_resload"] = f[f"{a}_load_fc_load"] - f[wind].sum(1) - f[f"{a}_ws_fc_solar"]
        fc.append(f"{a}_fc_resload")
    ref = f.loc[:"2023-12-31"]                           # standardise with 2022-2023 statistics (before every test fold)
    for c in fc:
        df["fcN_" + c] = ((f[c] - ref[c].mean()) / (ref[c].std() + 1e-9)).fillna(0.0).to_numpy()
        fc_cols.append("fcN_" + c)
    err_spec = []
    for c in fc:                                         # actual - forecast
        act = c.replace("_fc_", "_act_")
        if act in f:
            df["err_" + c] = (f[act] - f[c]).fillna(0.0).to_numpy()
            err_spec.append(("err_" + c, "err_" + c + "_in"))
    feats = feats + [b for b, _ in NET_SPEC] + [b for b, _ in err_spec]
    spec = spec + NET_SPEC + err_spec
    return df, feats, spec, fc_cols


def ts(x):
    t = pd.Timestamp(x); return t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")


def rolling_folds(test_months=3, train_months=12, val_months=3):
    folds, start, stop = [], ts(TEST_START), ts(TEST_END)
    while start < stop:
        te_end = min(start + pd.DateOffset(months=test_months), stop)
        va = start - pd.DateOffset(months=val_months); tr = va - pd.DateOffset(months=train_months)
        folds.append({"train": (tr, va), "val": (va, start), "test": (start, te_end)}); start = te_end
    return folds


def slice_range(df, rng):
    return df.loc[(df[TIME_COL] >= ts(rng[0])) & (df[TIME_COL] < ts(rng[1]))].copy()


def shift_frame(d, feature_cols, lags):
    """Lagged copies of every feature (lag 1 = the latest quarter-hour already observed)."""
    lag_cols = [f"{c}_lag{L}" for c in feature_cols for L in lags]
    new = {f"{c}_lag{L}": d[c].shift(L) for c in feature_cols for L in lags}
    s = pd.concat([d, pd.DataFrame(new, index=d.index)], axis=1)
    s = s.loc[:, ~s.columns.duplicated(keep="first")]
    return s[lag_cols + [LABEL, TIME_COL]].dropna().copy(), lag_cols


def scale_param(df_train, cols, params):
    sc = RobustScaler().fit(pd.concat([df_train[c] for c in cols], ignore_index=True).to_numpy().reshape(-1, 1))
    return {k: float(sc.transform(np.array([[v]], float))[0, 0]) for k, v in params.items()}


def rule_constants(df_train):
    """Austrian settlement constants (MRINN Sec. 3), expressed in the scale of the inputs they are compared with.
    Price constants use the price scaler; liquidity thresholds the intraday-volume scaler; imbalance thresholds
    (MW) are converted to MWh per quarter-hour (the unit of the system imbalance) and use its own scaler."""
    sv = RobustScaler().fit(df_train[["system_imbalance"]].to_numpy(float))
    v = lambda mw: float(sv.transform([[mw / 4.0]])[0, 0])
    C = {"C0": 0.1, **scale_param(df_train, FEATS_PRICES, {"C1": 5, "C2": 10, "C3": 15, "C10": 1000}),
         **scale_param(df_train, ["L_ID15", "L_ID60"], {"C5": 200, "C6": 200}),
         "C4": v(50), "C7": v(200), "C8": v(800), "C9": v(1000)}
    return [C[f"C{i}"] for i in range(11)]


def published_mask(st, horizons, hour_local):
    """True where the day-ahead value of the target quarter-hour is public at the forecast origin
    (values for day D+1 become public on day D at hour_local, Vienna time)."""
    o = pd.DatetimeIndex(st); o = (o.tz_localize("UTC") if o.tz is None else o.tz_convert("UTC")).tz_convert("Europe/Vienna")
    tgt = o.values[:, None] + (np.arange(1, horizons + 1) * np.timedelta64(15, "m"))[None, :]
    tl = pd.DatetimeIndex(tgt.ravel()).tz_localize("UTC").tz_convert("Europe/Vienna")
    oday = np.repeat(o.tz_localize(None).normalize().values, horizons); tday = tl.tz_localize(None).normalize().values
    otime = np.repeat((o.hour * 60 + o.minute).values, horizons)
    ok = (tday <= oday) | ((tday == oday + np.timedelta64(1, "D")) & (otime >= hour_local * 60))
    return ok.reshape(len(o), horizons)


def build_dataset(df, raw, shifted, scaled, lags, spec, fc_cols, y_scaler, stride=1):
    """Arrays for one split: history inputs (N, T, 1) per signal, day-ahead (N, 96, 2), calendar (N, 96, 4)
    [, forecasts (N, 96, k)] and the scaled 96-step target."""
    tgt = np.column_stack([shifted[LABEL].shift(-h).to_numpy("float32") for h in range(1, HORIZONS + 1)])
    lead = np.column_stack([raw[DA_COL].shift(-h).to_numpy("float32") for h in range(1, HORIZONS + 1)])
    pos = pd.Series(np.arange(len(raw)), index=pd.DatetimeIndex(raw[TIME_COL]))
    lead = lead[pos.reindex(pd.DatetimeIndex(shifted[TIME_COL])).to_numpy().astype(int)]
    present = np.where(np.isnan(lead), 0.0, (lead != 0.0)).astype("float32")
    da = np.stack([np.nan_to_num(lead, nan=0.0), present], axis=-1)
    idx = pd.DatetimeIndex(shifted[TIME_COL]); cal = np.empty((len(shifted), HORIZONS, 4), dtype="float32")
    for h in range(1, HORIZONS + 1):
        t = idx + pd.Timedelta(minutes=15 * h)
        tod = (t.hour * 60 + t.minute).to_numpy() / 1440.0; dow = t.dayofweek.to_numpy() / 7.0
        cal[:, h - 1] = np.stack([np.sin(2 * np.pi * tod), np.cos(2 * np.pi * tod), np.sin(2 * np.pi * dow), np.cos(2 * np.pi * dow)], -1)
    hist = [np.ascontiguousarray(scaled[[f"{b}_lag{L}" for L in lags]].to_numpy("float32")[:, ::-1])[:, :, None] for b, _ in spec]
    stamps = idx.to_numpy()
    cut = slice(None, -HORIZONS)
    hist = [h[cut] for h in hist]; da, cal, tgt, stamps = da[cut], cal[cut], tgt[cut], stamps[cut]
    good = ~np.isnan(tgt).any(axis=1)
    if stride > 1:
        sel = np.zeros(len(tgt), bool); sel[::stride] = True; good &= sel
    hist = [h[good] for h in hist]; da, cal, tgt, stamps = da[good], cal[good], tgt[good], stamps[good]
    tgt = y_scaler.transform(tgt.reshape(-1, 1)).reshape(tgt.shape).astype("float32")
    da = da * published_mask(stamps, HORIZONS, 13).astype("float32")[..., None]      # leak-free day-ahead price
    X = hist + [da, cal]
    if fc_cols:                                          # forecasts of each target quarter-hour, public from 18:00 on D-1
        p = pd.Index(pd.DatetimeIndex(df[TIME_COL])).get_indexer(pd.DatetimeIndex(stamps, tz="UTC"))
        A = df[fc_cols].to_numpy("float32"); j = np.clip(p[:, None] + np.arange(1, HORIZONS + 1)[None, :], 0, len(A) - 1)
        X = X + [A[j] * published_mask(stamps, HORIZONS, 18).astype("float32")[..., None]]
    return X, tgt, stamps


def prepare_fold(df, fold, feats, spec, fc_cols):
    lags = list(range(1, LOOKBACK + 1))
    raws = {k: slice_range(df, fold[k]) for k in ("train", "val", "test")}
    shifted, lag_cols = {}, None
    for k, d in raws.items():
        shifted[k], lag_cols = shift_frame(d, feats, lags)
    x_scaler = RobustScaler().fit(shifted["train"][lag_cols].to_numpy(float))          # fit on training data only
    y_scaler = RobustScaler().fit(shifted["train"][[LABEL]].to_numpy(float))
    scaled = {k: pd.DataFrame(x_scaler.transform(shifted[k][lag_cols].to_numpy(float)), columns=lag_cols) for k in raws}
    C = rule_constants(raws["train"])
    sets = {k: build_dataset(df, raws[k], shifted[k], scaled[k], lags, spec, fc_cols, y_scaler, stride=2 if k == "train" else 1)
            for k in ("train", "val", "test")}
    return sets, y_scaler, C


# ================================================================== differentiable market-rule operators (MRINN Sec. 4)
def smooth_abs(x, eps=1e-9): return K.sqrt(K.square(x) + eps)
def smooth_sign(x): return K.tanh(x)
def smooth_max(x, y): return y + K.softplus(x - y)
def smooth_min(x, y): return -smooth_max(-x, -y)
def safe_div_pair(t, eps=1e-7):
    P, L = t; return P / smooth_max(K.abs(L), eps)
def weighted_frac(numer, denom, H): return layers.Lambda(safe_div_pair, output_shape=(H,))([numer, denom])


def get_P_RE_netting(E_ap, E_mp, E_an, E_mn, V, N_imp, N_exp, P_ap, P_mp, P_an, P_mn, P_vp, P_vn, H):
    """Netting-aware reserve-energy component (Objective 1). The original activation-weighted price and VoAA branches
    are kept; a learned netting share s reduces the activation to a residual activation (1 - s) * E, which adds a
    merit-order-depth term to the activated price and drives the branch selection. s starts near 0.12 and the depth
    term at 0, so training starts from the original rule."""
    tanh = lambda t, n=None: layers.Activation("tanh", name=n)(t)
    E_sum_pos, E_sum_neg = E_ap + E_mp, E_an + E_mn
    P_act_pos = weighted_frac(E_ap * P_ap + E_mp * P_mp, E_sum_pos, H)
    P_act_neg = weighted_frac(E_an * P_an + E_mn * P_mn, E_sum_neg, H)
    N_net = layers.Subtract(name="netting_net")([N_imp, N_exp])
    s_in = layers.Concatenate(name="netting_share_in")([tanh(N_net), tanh(N_imp), tanh(N_exp), V, tanh(E_sum_pos), tanh(E_sum_neg)])
    s = layers.Dense(H, activation="sigmoid", bias_initializer=initializers.Constant(-2.0), name="netting_share")(s_in)
    keep = layers.Lambda(lambda t: 1.0 - t, name="not_netted")(s)
    E_res_pos = layers.Multiply(name="residual_activation_pos")([keep, E_sum_pos])
    E_res_neg = layers.Multiply(name="residual_activation_neg")([keep, E_sum_neg])
    depth = lambda t, n: layers.Dense(H, use_bias=False, kernel_initializer="zeros", name=n)(tanh(t))
    P_act_pos = layers.Add(name="P_act_pos_depth")([P_act_pos, depth(E_res_pos, "merit_depth_pos")])
    P_act_neg = layers.Add(name="P_act_neg_depth")([P_act_neg, depth(E_res_neg, "merit_depth_neg")])
    gate_in = layers.Concatenate(name="gate_RE_in")([tanh(E_res_pos), tanh(E_res_neg), V])
    W = layers.Dense(4, activation="softmax", name="gate_RE")(gate_in)
    onesH = K.ones_like(P_act_pos)
    return layers.Add(name="P_RE_rep")([layers.Multiply()([P_act_pos, W[:, 0:1] * onesH]), layers.Multiply()([P_act_neg, W[:, 1:2] * onesH]),
                                        layers.Multiply()([P_vp, W[:, 2:3] * onesH]), layers.Multiply()([P_vn, W[:, 3:4] * onesH])])


def ramp_function(P_rep, V_rep, C4, label):
    C4_t = K.ones_like(V_rep) * K.cast(C4, V_rep.dtype)
    W = layers.Dense(3, activation="softmax", name=f"gate_ramp_{label}")(layers.Concatenate(name=f"gate_ramp_in_{label}")([V_rep, C4_t]))
    onesH = K.ones_like(P_rep)
    return -onesH * (W[:, 0:1] * onesH) + (V_rep / (C4_t + 1e-7)) * onesH * (W[:, 1:2] * onesH) + onesH * (W[:, 2:3] * onesH)


def get_P_EX(P15, L15, P60, L60, PDA, V, C0, C1, C2, C3, C4, C5, C6):
    """Exchange-reference component: liquidity-weighted, ramp-adjusted intraday / day-ahead reference."""
    r15, r60, rDA = (ramp_function(p, V, C4, n) for p, n in ((P15, "ID15"), (P60, "ID60"), (PDA, "DA")))
    C0_t = K.cast(C0, P15.dtype)
    m1 = smooth_max(K.ones_like(P15) * K.cast(C1, P15.dtype), C0_t * smooth_abs(P15))
    m2 = smooth_max(K.ones_like(P60) * K.cast(C2, P60.dtype), C0_t * smooth_abs(P60))
    m3 = smooth_max(K.ones_like(PDA) * K.cast(C3, PDA.dtype), C0_t * smooth_abs(PDA))
    P15m, P60m, PDAm = P15 + r15 * m1, P60 + r60 * m2, PDA + rDA * m3
    one = K.ones_like(L15)
    w15 = smooth_min(one, L15 / K.cast(C5, L15.dtype)); w60 = smooth_min(one - w15, L60 / K.cast(C6, L60.dtype)); wDA = one - w15 - w60
    P_EX = P15m * (w15 * K.ones_like(P15m)) + P60m * (w60 * K.ones_like(P60m)) + PDAm * (wDA * K.ones_like(PDAm))
    P_base = layers.Add(name="P_EX_basis")([layers.Multiply()([P15, w15]), layers.Multiply()([P60, w60]), layers.Multiply()([PDA, wDA])])
    return P_EX, P_base


def get_P_SC(P_base, V, C7, C8, C9, C10):
    """Scarcity component: signed cubic adjustment above a deadband of |imbalance|."""
    aV, sV = smooth_abs(V), smooth_sign(V)
    t = lambda c: K.ones_like(aV) * K.cast(c, aV.dtype)
    C7_t, C8_t, C9_t, C10_t = t(C7), t(C8), t(C9), t(C10)
    denom = (C9_t - C7_t) + 1e-7
    term1 = P_base + sV * C10_t * K.power((aV - C7_t) / denom, 3.0)
    term2 = P_base + sV * C10_t * K.power((C8_t - C7_t) / denom, 3.0)
    W = layers.Dense(3, activation="softmax", name="Psc_gate")(layers.Concatenate(name="Psc_gate_in")([aV, C7_t, C8_t]))
    return layers.Add(name="Psc_mix")([layers.Multiply()([P_base, W[:, 0:1]]), layers.Multiply()([term1, W[:, 1:2]]),
                                       layers.Multiply()([term2, W[:, 2:3]])])


def hierarchical_quantile_head(z, quantiles, prefix="hq"):
    """Median from a dense layer; other quantiles by adding / subtracting softplus steps -> they cannot cross."""
    qs = sorted(quantiles); mid = qs.index(0.5)
    q50 = layers.Dense(1, name=f"{prefix}_q50")(z); out = {0.5: q50}; cur = q50
    for q in qs[mid + 1:]:
        cur = layers.Add()([cur, layers.Dense(1, activation="softplus", name=f"{prefix}_up_{int(q * 100)}")(z)]); out[q] = cur
    cur = q50
    for q in reversed(qs[:mid]):
        cur = layers.Subtract()([cur, layers.Dense(1, activation="softplus", name=f"{prefix}_dn_{int(q * 100)}")(z)]); out[q] = cur
    return layers.Concatenate(name=f"{prefix}_out")([out[q] for q in qs])


@tf.keras.utils.register_keras_serializable(package="mrrnn")
class HorizonEmbedding(layers.Layer):
    """One learned vector per forecast horizon."""
    def __init__(self, horizons, dim=4, **kw):
        super().__init__(**kw); self.horizons, self.dim = int(horizons), int(dim)
    def build(self, input_shape):
        self.table = self.add_weight(shape=(self.horizons, self.dim), initializer="glorot_uniform", trainable=True, name="table")
        super().build(input_shape)
    def call(self, ref):
        return tf.tile(tf.expand_dims(self.table, 0), [tf.shape(ref)[0], 1, 1])
    def compute_output_shape(self, input_shape): return (input_shape[0], self.horizons, self.dim)
    def get_config(self): return {**super().get_config(), "horizons": self.horizons, "dim": self.dim}


@tf.keras.utils.register_keras_serializable(package="mrrnn")
class DAOutageDropout(layers.Layer):
    """Randomly blank the whole day-ahead input on a fraction of training samples (teaches the model to handle a
    missing / unpublished day-ahead price)."""
    def __init__(self, rate=0.2, **kw):
        super().__init__(**kw); self.rate = float(rate)
    def call(self, x, training=None):
        if not training or self.rate <= 0.0: return x
        return x * tf.cast(tf.random.uniform((tf.shape(x)[0], 1, 1)) >= self.rate, x.dtype)
    def compute_output_shape(self, input_shape): return input_shape
    def get_config(self): return {**super().get_config(), "rate": self.rate}


def pinball_loss(quantiles):
    qs = tf.reshape(tf.constant(list(quantiles), tf.float32), (1, 1, -1))
    def loss(y_true, y_pred):
        e = tf.expand_dims(tf.cast(y_true, tf.float32), -1) - y_pred
        return tf.reduce_mean(tf.maximum(qs * e, (qs - 1.0) * e))
    return loss


# ================================================================== model
def build_model(cell, C, spec, n_fc):
    H, F, Hz, Q, T = HIDDEN, FUSION_WIDTH, HORIZONS, len(QUANTILES), LOOKBACK
    bases, names = [b for b, _ in spec], [n for _, n in spec]
    hist_in = [layers.Input(shape=(T, 1), name=n) for n in names]
    da_in = layers.Input(shape=(Hz, 2), name="da_future_in")
    cal_in = layers.Input(shape=(Hz, 4), name="calendar_in")
    fc_in = layers.Input(shape=(Hz, n_fc), name="forecast_in") if n_fc else None
    Cell = layers.LSTM if cell == "lstm" else layers.GRU
    reps = [Cell(H, return_sequences=False, name=f"{b}_{cell}_0")(x) for x, b in zip(hist_in, bases)]      # one encoder per signal
    reps = [layers.RepeatVector(Hz, name=f"rep_{b}")(r) for r, b in zip(reps, bases)]
    emb = HorizonEmbedding(Hz, 4, name="horizon_embedding")(reps[0])
    mix = layers.Dense(H, activation="swish", name="horizon_mix")
    reps = [mix(layers.Concatenate(name=f"hcat_{b}")([r, emb])) for r, b in zip(reps, bases)]
    (V, E_ap, E_mp, P_ap, P_mp, E_an, E_mn, P_an, P_mn, P_vp, P_vn, P15, P60, PDA, L15, L60, LDA) = reps[:17]
    N_imp = layers.Dropout(0.2, name="IN_import_drop")(reps[17]); N_exp = layers.Dropout(0.2, name="IN_export_drop")(reps[18])
    extra = reps[19:]                                     # forecast-error encoders go to the fusion layer
    da_x = DAOutageDropout(DA_DROPOUT, name="da_outage")(da_in)
    PDA = layers.Dense(H, activation="swish", name="da_future_rep")(da_x)        # day-ahead price enters the rules as known future
    kn = [da_x, cal_in] + ([fc_in] if fc_in is not None else [])
    known = layers.Dense(H, activation="swish", name="known_future_rep")(layers.Concatenate(name="known_future_cat")(kn))
    fold = layers.Lambda(lambda t: tf.reshape(t, (-1, H)), name="fold")          # (batch, 96, H) -> (batch*96, H)
    foldK = layers.Lambda(lambda t: tf.reshape(t, (-1, H)), name="fold_known")
    (Vf, E_apf, E_mpf, P_apf, P_mpf, E_anf, E_mnf, P_anf, P_mnf, P_vpf, P_vnf, P15f, P60f, PDAf, L15f, L60f, LDAf) = [
        fold(t) for t in (V, E_ap, E_mp, P_ap, P_mp, E_an, E_mn, P_an, P_mn, P_vp, P_vn, P15, P60, PDA, L15, L60, LDA)]
    P_RE = get_P_RE_netting(E_apf, E_mpf, E_anf, E_mnf, Vf, fold(N_imp), fold(N_exp), P_apf, P_mpf, P_anf, P_mnf, P_vpf, P_vnf, H)
    P_EX, P_base = get_P_EX(P15f, L15f, P60f, L60f, PDAf, Vf, *C[:7])
    P_SC = get_P_SC(P_base, Vf, *C[7:11])
    lo = smooth_min(smooth_min(P_RE, P_EX), P_SC); hi = smooth_max(smooth_max(P_RE, P_EX), P_SC)
    W = layers.Dense(2, activation="softmax", name="final_gate")(Vf)                # soft "max if short, min if long"
    z = layers.Add(name="imbalance_price_rep")([layers.Multiply()([lo, W[:, 0:1]]), layers.Multiply()([hi, W[:, 1:2]])])
    fuse = [z, foldK(known)] + [fold(e) for e in extra]
    z = layers.Dense(F, activation="swish", name="rep_with_known")(layers.Concatenate(name="rep_concat")(fuse))
    out = layers.Lambda(lambda t: tf.reshape(t, (-1, Hz, Q)), name="unfold")(hierarchical_quantile_head(z, QUANTILES))
    m = models.Model(hist_in + [da_in, cal_in] + ([fc_in] if fc_in is not None else []), out, name=f"MR_modified_AT_{cell}")
    m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR), loss=pinball_loss(QUANTILES))
    return m


# ================================================================== evaluation
def inv(a, sc): a = np.asarray(a); return sc.inverse_transform(a.reshape(-1, 1)).ravel().reshape(a.shape)


def metrics(yt, yp, mase_scale):
    """AQL, MAE, RMSE, MASE, AQCR, 80% coverage (per horizon, then averaged over the 96 horizons; EUR/MWh)."""
    Q = np.array(QUANTILES); e = yt[..., None] - yp; err = yt - yp[..., 2]
    return {"AQL": float(np.maximum(Q * e, (Q - 1) * e).mean(axis=(0, 2)).mean()), "MAE": float(np.abs(err).mean(0).mean()),
            "RMSE": float(np.sqrt((err ** 2).mean(0)).mean()), "MASE": float(np.abs(err).mean(0).mean() / mase_scale),
            "AQCR": float(100 * (yp[..., :-1] > yp[..., 1:]).mean(axis=(0, 2)).max()),
            "Cov80": float(100 * ((yt >= yp[..., 0]) & (yt <= yp[..., 4])).mean())}


def mase_scale(df):
    """MAE of the seasonal-naive forecast (same quarter-hour yesterday) over the test year."""
    s = df.set_index(TIME_COL)[LABEL].sort_index(); s = s[~s.index.duplicated()].asfreq("15min")
    return float((s - s.shift(96)).loc[ts(TEST_START):ts(TEST_END) - pd.Timedelta(minutes=15)].abs().mean())


def run(args):
    df, feats, spec, fc_cols = load_data(args.data)
    scale = mase_scale(df); os.makedirs(args.out, exist_ok=True); rows = []
    tag = f"austria_modified_mr{args.cell}"
    for seed in args.seeds:
        yts, yps, sts = [], [], []; t0 = time.time()
        for fi, fold in enumerate(rolling_folds()):
            sets, ysc, C = prepare_fold(df, fold, feats, spec, fc_cols)
            (Xtr, ytr, _), (Xva, yva, _), (Xte, yte, ste) = sets["train"], sets["val"], sets["test"]
            set_random_seed(seed)
            m = build_model(args.cell, C, spec, len(fc_cols))
            es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True, mode="min")
            h = m.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=args.epochs, batch_size=BATCH, callbacks=[es], verbose=0)
            yp = m.predict(Xte, verbose=0, batch_size=1024)
            yts.append(inv(yte, ysc)); yps.append(np.stack([inv(yp[:, :, j], ysc) for j in range(len(QUANTILES))], -1)); sts.append(ste)
            print(f"seed {seed} fold {fi}: {len(h.history['loss'])} epochs, {m.count_params():,} parameters", flush=True)
            tf.keras.backend.clear_session()
        yt, yp, st = np.concatenate(yts), np.concatenate(yps), np.concatenate(sts)
        np.savez_compressed(os.path.join(args.out, f"{tag}_s{seed}_predictions.npz"), y_true=yt, y_pred=yp, origin=st.astype("datetime64[ns]"))
        r = {"model": tag, "seed": seed, **metrics(yt, yp, scale), "minutes": round((time.time() - t0) / 60, 1)}
        rows.append(r); print(r, flush=True)
    T_ = pd.DataFrame(rows); T_.to_csv(os.path.join(args.out, f"{tag}_metrics.csv"), index=False)
    print(T_.drop(columns=["model", "seed"]).mean().round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="austria_imbalance_dataset.csv")
    ap.add_argument("--cell", choices=["gru", "lstm"], default="gru")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--out", default="results")
    run(ap.parse_args())
