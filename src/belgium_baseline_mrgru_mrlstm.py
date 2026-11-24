"""Belgium - BASELINE MR-GRU / MR-LSTM (MR-GRU-EU with the Belgian rule configuration).

Day-ahead probabilistic forecasting of the Belgian (Elia) imbalance price: every 15 minutes the model forecasts the
next 96 quarter-hours (24 h) as five quantiles (q10, q25, q50, q75, q90).

Model: MR-GRU-EU / MR-LSTM-EU with the Belgian market-rule configuration - the same architecture as the Austrian
MR-GRU (one recurrent encoder per signal, known-future path, hierarchical quantile head), but the rule block follows
Elia's imbalance price: marginal activation prices plus an incentive alpha,
    price = MIP + alpha  (system short)        price = MDP - alpha  (system long)
    alpha = b * sigmoid((|SI| - c) / d),  b = 200 EUR/MWh, c = 450 MW, d = 65 MW
with the floor / cap as the value-of-avoided-activation branch. Belgium has no exchange-reference or scarcity
component, and Belgian intraday price indices are licensed, so they are not used.

Baseline: the 11 Elia balancing signals (imbalance, activated aFRR/mFRR volumes and prices, floor/cap).
Parameters: 3,793 (GRU).

Evaluation: data after Elia's balancing redesign only (MARI go-live 22 May 2024); 4 rolling folds (12 months train /
3 validation / 3 test) covering Oct 2025 - Sep 2026; leak-free day-ahead price (hidden until ~12:45 on D-1).

Usage
    python belgium_baseline_mrgru_mrlstm.py --data belgium_imbalance_dataset.csv --cell gru --seeds 42 43 44
    python belgium_baseline_mrgru_mrlstm.py --data belgium_imbalance_dataset.csv --cell lstm --seeds 42 43 44
Outputs (in --out, default ./results): per-seed predictions (.npz) and a metrics table (.csv)
Requirements: Python >= 3.10, tensorflow >= 2.16 (Keras 3), numpy, pandas, scikit-learn
"""
import argparse, os, random, time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from keras import ops as K
from sklearn.preprocessing import RobustScaler

# ================================================================== settings
TIME_COL, LABEL, DA_COL = "Time [UTC] start", "imbalance_price", "P_DA"
HORIZONS, LOOKBACK, HIDDEN = 96, 32, 8
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
TEST_START, TEST_END = "2025-10-01 00:00:00+00:00", "2026-10-01 00:00:00+00:00"
ALPHA_B, ALPHA_C, ALPHA_D = 200.0, 450.0, 65.0          # Elia incentive alpha: EUR/MWh, MW, MW
DA_DROPOUT = 0.2
EPOCHS, PATIENCE, BATCH, LR = 120, 12, 1024, 1e-3

# role -> column. The rule block reads its inputs by role.
ROLES = dict(V="system_imbalance", E_ap="E_aFRR_pos", E_mp="E_mFRR_pos", P_ap="P_aFRR_pos", P_mp="P_mFRR_pos",
             E_an="E_aFRR_neg", E_mn="E_mFRR_neg", P_an="P_aFRR_neg", P_mn="P_mFRR_neg", P_vp="P_floor", P_vn="P_cap")
BASE_SPEC = [(ROLES[r], ROLES[r] + "_in") for r in ("V", "E_ap", "E_mp", "P_ap", "P_mp", "E_an", "E_mn", "P_an", "P_mn", "P_vp", "P_vn")]


def set_random_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed); random.seed(seed); np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


# ================================================================== data
def load_data(path):
    df = pd.read_csv(path, parse_dates=[TIME_COL]).sort_values(TIME_COL).reset_index(drop=True)
    spec = list(BASE_SPEC)
    return df, [b for b, _ in spec], spec


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
    lag_cols = [f"{c}_lag{L}" for c in feature_cols for L in lags]
    new = {f"{c}_lag{L}": d[c].shift(L) for c in feature_cols for L in lags}
    s = pd.concat([d, pd.DataFrame(new, index=d.index)], axis=1)
    s = s.loc[:, ~s.columns.duplicated(keep="first")]
    return s[lag_cols + [LABEL, TIME_COL]].dropna().copy(), lag_cols


def rule_constants(df_train):
    """Alpha constants in the scale of their inputs: centre c and width d (MW, the unit of the imbalance) divided by the
    imbalance's robust scale; amplitude b (EUR/MWh) divided by the activation-price scale."""
    sv = RobustScaler().fit(df_train[["system_imbalance"]].to_numpy(float))
    pcols = [ROLES[r] for r in ("P_ap", "P_mp", "P_an", "P_mn")]
    sp = RobustScaler().fit(pd.concat([df_train[c] for c in pcols]).to_numpy().reshape(-1, 1))
    return dict(alpha_b=ALPHA_B / sp.scale_[0], alpha_c=ALPHA_C / sv.scale_[0], alpha_d=ALPHA_D / sv.scale_[0])


def published_mask(st, horizons, hour_local):
    o = pd.DatetimeIndex(st); o = (o.tz_localize("UTC") if o.tz is None else o.tz_convert("UTC")).tz_convert("Europe/Vienna")
    tgt = o.values[:, None] + (np.arange(1, horizons + 1) * np.timedelta64(15, "m"))[None, :]
    tl = pd.DatetimeIndex(tgt.ravel()).tz_localize("UTC").tz_convert("Europe/Vienna")
    oday = np.repeat(o.tz_localize(None).normalize().values, horizons); tday = tl.tz_localize(None).normalize().values
    otime = np.repeat((o.hour * 60 + o.minute).values, horizons)
    ok = (tday <= oday) | ((tday == oday + np.timedelta64(1, "D")) & (otime >= hour_local * 60))
    return ok.reshape(len(o), horizons)


def build_dataset(raw, shifted, scaled, lags, spec, y_scaler, stride=1):
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
    stamps = idx.to_numpy(); cut = slice(None, -HORIZONS)
    hist = [h[cut] for h in hist]; da, cal, tgt, stamps = da[cut], cal[cut], tgt[cut], stamps[cut]
    good = ~np.isnan(tgt).any(axis=1)
    if stride > 1:
        sel = np.zeros(len(tgt), bool); sel[::stride] = True; good &= sel
    hist = [h[good] for h in hist]; da, cal, tgt, stamps = da[good], cal[good], tgt[good], stamps[good]
    tgt = y_scaler.transform(tgt.reshape(-1, 1)).reshape(tgt.shape).astype("float32")
    da = da * published_mask(stamps, HORIZONS, 13).astype("float32")[..., None]      # leak-free day-ahead price
    return hist + [da, cal], tgt, stamps


def prepare_fold(df, fold, feats, spec):
    lags = list(range(1, LOOKBACK + 1))
    raws = {k: slice_range(df, fold[k]) for k in ("train", "val", "test")}
    shifted, lag_cols = {}, None
    for k, d in raws.items():
        shifted[k], lag_cols = shift_frame(d, feats, lags)
    x_scaler = RobustScaler().fit(shifted["train"][lag_cols].to_numpy(float))
    y_scaler = RobustScaler().fit(shifted["train"][[LABEL]].to_numpy(float))
    scaled = {k: pd.DataFrame(x_scaler.transform(shifted[k][lag_cols].to_numpy(float)), columns=lag_cols) for k in raws}
    C = rule_constants(raws["train"])
    sets = {k: build_dataset(raws[k], shifted[k], scaled[k], lags, spec, y_scaler, stride=2 if k == "train" else 1)
            for k in ("train", "val", "test")}
    return sets, y_scaler, C


# ================================================================== differentiable rule operators and layers
def smooth_abs(x, eps=1e-9): return K.sqrt(K.square(x) + eps)
def smooth_sign(x): return K.tanh(x)
def smooth_max(x, y): return y + K.softplus(x - y)
def smooth_min(x, y): return -smooth_max(-x, -y)


def balancing_component(r, H):
    """Belgian balancing price: marginal activation prices (highest upward / lowest downward activated bid) or the
    floor / cap when nothing was activated, selected by a learned soft if-else."""
    E_sum_pos, E_sum_neg = r["E_ap"] + r["E_mp"], r["E_an"] + r["E_mn"]
    P_pos = smooth_max(r["P_ap"], r["P_mp"])            # MIP
    P_neg = smooth_min(r["P_an"], r["P_mn"])            # MDP
    gate_in = layers.Concatenate(name="gate_RE_in")([layers.Activation("tanh")(E_sum_pos), layers.Activation("tanh")(E_sum_neg), r["V"]])
    W = layers.Dense(4, activation="softmax", name="gate_RE")(gate_in)
    onesH = K.ones_like(P_pos)
    return layers.Add(name="P_RE_rep")([layers.Multiply()([P_pos, W[:, 0:1] * onesH]), layers.Multiply()([P_neg, W[:, 1:2] * onesH]),
                                        layers.Multiply()([r["P_vp"], W[:, 2:3] * onesH]), layers.Multiply()([r["P_vn"], W[:, 3:4] * onesH])])


def alpha_component(V, C, H):
    """alpha = b * sigmoid((|V| - c) / d), added when the system is short and subtracted when it is long."""
    aV = smooth_abs(V)
    t = lambda c: K.ones_like(aV) * K.cast(c, aV.dtype)
    s = K.sigmoid((aV - t(C["alpha_c"])) / (t(C["alpha_d"]) + 1e-7))
    amp = layers.Dense(H, use_bias=False, name="alpha_amp", kernel_initializer=tf.keras.initializers.Identity(gain=float(C["alpha_b"])))(s)
    return smooth_sign(V) * amp


def hierarchical_quantile_head(z, quantiles, prefix="hq"):
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
    def __init__(self, horizons, dim=4, **kw):
        super().__init__(**kw); self.horizons, self.dim = int(horizons), int(dim)
    def build(self, input_shape):
        self.table = self.add_weight(shape=(self.horizons, self.dim), initializer="glorot_uniform", trainable=True, name="table")
        super().build(input_shape)
    def call(self, ref): return tf.tile(tf.expand_dims(self.table, 0), [tf.shape(ref)[0], 1, 1])
    def compute_output_shape(self, input_shape): return (input_shape[0], self.horizons, self.dim)
    def get_config(self): return {**super().get_config(), "horizons": self.horizons, "dim": self.dim}


@tf.keras.utils.register_keras_serializable(package="mrrnn")
class DAOutageDropout(layers.Layer):
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
def build_model(cell, C, spec):
    H, Hz, Q, T = HIDDEN, HORIZONS, len(QUANTILES), LOOKBACK
    bases, names = [b for b, _ in spec], [n for _, n in spec]
    hist_in = [layers.Input(shape=(T, 1), name=n) for n in names]
    da_in = layers.Input(shape=(Hz, 2), name="da_future_in")
    cal_in = layers.Input(shape=(Hz, 4), name="calendar_in")
    Cell = layers.LSTM if cell == "lstm" else layers.GRU
    reps = [Cell(H, return_sequences=False, name=f"{b}_{cell}_0")(x) for x, b in zip(hist_in, bases)]
    reps = [layers.RepeatVector(Hz, name=f"rep_{b}")(r) for r, b in zip(reps, bases)]
    emb = HorizonEmbedding(Hz, 4, name="horizon_embedding")(reps[0])
    mix = layers.Dense(H, activation="swish", name="horizon_mix")
    reps = [mix(layers.Concatenate(name=f"hcat_{b}")([r, emb])) for r, b in zip(reps, bases)]
    by_base = dict(zip(bases, reps))
    da_x = DAOutageDropout(DA_DROPOUT, name="da_outage")(da_in)
    known = layers.Dense(H, activation="swish", name="known_future_rep")(layers.Concatenate(name="known_future_cat")([da_x, cal_in]))
    fold = layers.Lambda(lambda t: tf.reshape(t, (-1, H)), name="fold")
    foldK = layers.Lambda(lambda t: tf.reshape(t, (-1, H)), name="fold_known")
    r = {k: fold(by_base[col]) for k, col in ROLES.items()}
    extra = [rep for b, rep in zip(bases, reps) if b not in set(ROLES.values())]      # market-signal encoders -> fusion
    P_bal = balancing_component(r, H)
    P_bal = layers.Add(name="P_bal_alpha")([P_bal, alpha_component(r["V"], C, H)])
    z = layers.Activation("linear", name="imbalance_price_rep")(P_bal)
    fuse = [z, foldK(known)] + [fold(e) for e in extra]
    z = layers.Dense(H, activation="swish", name="rep_with_known")(layers.Concatenate(name="rep_concat")(fuse))
    out = layers.Lambda(lambda t: tf.reshape(t, (-1, Hz, Q)), name="unfold")(hierarchical_quantile_head(z, QUANTILES))
    m = models.Model(hist_in + [da_in, cal_in], out, name=f"MR_baseline_BE_{cell}")
    m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR), loss=pinball_loss(QUANTILES))
    return m


# ================================================================== evaluation
def inv(a, sc): a = np.asarray(a); return sc.inverse_transform(a.reshape(-1, 1)).ravel().reshape(a.shape)


def metrics(yt, yp, mase_scale):
    Q = np.array(QUANTILES); e = yt[..., None] - yp; err = yt - yp[..., 2]
    return {"AQL": float(np.maximum(Q * e, (Q - 1) * e).mean(axis=(0, 2)).mean()), "MAE": float(np.abs(err).mean(0).mean()),
            "RMSE": float(np.sqrt((err ** 2).mean(0)).mean()), "MASE": float(np.abs(err).mean(0).mean() / mase_scale),
            "AQCR": float(100 * (yp[..., :-1] > yp[..., 1:]).mean(axis=(0, 2)).max()),
            "Cov80": float(100 * ((yt >= yp[..., 0]) & (yt <= yp[..., 4])).mean())}


def mase_scale(df):
    s = df.set_index(TIME_COL)[LABEL].sort_index(); s = s[~s.index.duplicated()].asfreq("15min")
    return float((s - s.shift(96)).loc[ts(TEST_START):ts(TEST_END) - pd.Timedelta(minutes=15)].abs().mean())


def run(args):
    df, feats, spec = load_data(args.data)
    scale = mase_scale(df); os.makedirs(args.out, exist_ok=True); rows = []
    tag = f"belgium_baseline_mr{args.cell}"
    for seed in args.seeds:
        yts, yps, sts = [], [], []; t0 = time.time()
        for fi, fold in enumerate(rolling_folds()):
            sets, ysc, C = prepare_fold(df, fold, feats, spec)
            (Xtr, ytr, _), (Xva, yva, _), (Xte, yte, ste) = sets["train"], sets["val"], sets["test"]
            set_random_seed(seed)
            m = build_model(args.cell, C, spec)
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
    ap.add_argument("--data", default="belgium_imbalance_dataset.csv")
    ap.add_argument("--cell", choices=["gru", "lstm"], default="gru")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--out", default="results")
    run(ap.parse_args())
