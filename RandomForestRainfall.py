# =============================================================================
# RF vs RF-AMM — Flood-Oriented Rainfall Prediction (Jakarta)
# Full experiment pipeline
# =============================================================================
# Requirements:
#   pip install pandas numpy scikit-learn matplotlib seaborn scipy openpyxl
#
# HOW TO USE:
#   1. Put BMKG .csv files in a folder called "data/"
#   2. Run: python RandomForestRainfall.py
#   3. Results saved in "output/" folder
# =============================================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import confusion_matrix as cm_func
from sklearn.metrics import (
    recall_score, f1_score, precision_score,
    roc_auc_score, confusion_matrix, precision_recall_curve
)

warnings.filterwarnings('ignore')
np.random.seed(42)

# =============================================================================
# 0. CONFIGURATION
# =============================================================================
DATA_DIR        = "data"
OUTPUT_DIR      = "output"
FLOOD_THRESHOLD = 20.0      # mm/day — BMKG "hujan lebat" threshold
K_DECAY         = 0.90      # API decay constant
N_FOLDS         = 5
TRAIN_RATIO     = 0.80
RANDOM_STATE    = 42
MIN_STATION_ROWS = 600      # stations with fewer rows are skipped

# --- Internal column names used throughout the script ---
COL_DATE   = "TANGGAL"
COL_RR     = "RR"
COL_TAVG   = "Tavg"
COL_TX     = "Tx"
COL_TN     = "Tn"
COL_RH     = "RH_avg"
COL_SS     = "ss"
COL_FF_AVG = "ff_avg"
COL_FF_X   = "ff_x"
COL_DDDX   = "ddd_x"
# NOTE: ddd_car excluded — BMKG encodes it as compass letters (N/S/E/W),
#       not degrees, so it cannot be used as a numeric feature.

ALL_METEO_COLS = [
    COL_RR, COL_TAVG, COL_TX, COL_TN,
    COL_RH, COL_SS, COL_FF_AVG, COL_FF_X, COL_DDDX
]

# Map from whatever capitalisation BMKG uses → internal names above
COL_MAP = {
    'TANGGAL' : COL_DATE,
    'TN'      : COL_TN,
    'TX'      : COL_TX,
    'TAVG'    : COL_TAVG,
    'RH_AVG'  : COL_RH,
    'RR'      : COL_RR,
    'SS'      : COL_SS,
    'FF_X'    : COL_FF_X,
    'DDD_X'   : COL_DDDX,
    'FF_AVG'  : COL_FF_AVG,
    'DDD_CAR' : 'ddd_car',   # kept in map but excluded from ALL_METEO_COLS
}

RF_PARAMS = dict(
    n_estimators=200,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',
    random_state=RANDOM_STATE,
    n_jobs=-1
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# HELPER — robust date parser
# =============================================================================
def parse_dates(series):
    formats = [
        '%d-%m-%Y', '%d/%m/%Y', '%d-%m-%y', '%d/%m/%y',
        '%Y-%m-%d', '%m/%d/%Y', '%d %m %Y', '%m-%d-%Y',
    ]
    parsed = pd.to_datetime(series, dayfirst=True, errors='coerce')
    for fmt in formats:
        still_na = parsed.isna()
        if not still_na.any():
            break
        try:
            parsed[still_na] = pd.to_datetime(
                series[still_na], format=fmt, errors='coerce'
            )
        except Exception:
            continue
    return parsed


# =============================================================================
# 1. DATA LOADING
# =============================================================================
def load_bmkg_files(data_dir):
    frames = []

    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith('.csv'):
            continue
        path = os.path.join(data_dir, fname)
        try:
            df = pd.read_csv(path, sep=',', encoding='latin-1', low_memory=False)
            df.columns = df.columns.str.strip()

            # Remap column names to internal standard
            df.columns = [COL_MAP.get(c.upper().strip(), c) for c in df.columns]

            # Replace BMKG missing-value codes and blank strings
            df.replace(
                {8888: np.nan, 9999: np.nan,
                 '8888': np.nan, '9999': np.nan,
                 '-': np.nan, '': np.nan, ' ': np.nan},
                inplace=True
            )

            # Empty RR cell means no rain (0), not missing
            if COL_RR in df.columns:
                df[COL_RR] = pd.to_numeric(df[COL_RR], errors='coerce').fillna(0.0)

            # Count valid date rows
            tmp_dates = parse_dates(df[COL_DATE]) if COL_DATE in df.columns else pd.Series([])
            valid_rows = tmp_dates.notna().sum()

            if valid_rows < MIN_STATION_ROWS:
                print(f"  SKIPPED (incomplete): {fname}  "
                      f"({valid_rows} valid dates — below {MIN_STATION_ROWS}-day threshold)")
                continue

            frames.append(df)
            print(f"  Loaded: {fname}  ({len(df)} rows, {valid_rows} valid dates)")

        except Exception as e:
            print(f"  WARNING: Could not load {fname}: {e}")

    if not frames:
        raise FileNotFoundError(
            f"No usable .csv files found in '{data_dir}/'. "
            "Check that files exist and have enough rows."
        )

    # Per-station: parse dates, force numeric, average duplicates on same date
    station_frames = []
    for df in frames:
        df[COL_DATE] = parse_dates(df[COL_DATE])
        df.dropna(subset=[COL_DATE], inplace=True)

        num_cols = [c for c in ALL_METEO_COLS if c in df.columns]
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df[num_cols] = df[num_cols].replace({8888: np.nan, 9999: np.nan})

        # Average across any duplicate dates within the same station file
        df = df.groupby(COL_DATE)[num_cols].mean(numeric_only=True).reset_index()
        station_frames.append(df)

    # Outer merge across stations — keeps every date present in ANY station
    combined = station_frames[0]
    for i, sf in enumerate(station_frames[1:], 1):
        combined = pd.merge(
            combined, sf,
            on=COL_DATE, how='outer',
            suffixes=('', f'_s{i}')
        )

    # Average the duplicated columns that came from different stations
    for col in ALL_METEO_COLS:
        matching = [c for c in combined.columns
                    if c == col or c.startswith(col + '_s')]
        if len(matching) > 1:
            combined[col] = combined[matching].mean(axis=1)
            combined.drop(
                columns=[c for c in matching if c != col],
                inplace=True
            )
        elif len(matching) == 1 and matching[0] != col:
            combined.rename(columns={matching[0]: col}, inplace=True)

    combined.sort_values(COL_DATE, inplace=True)
    combined.reset_index(drop=True, inplace=True)

    present = [c for c in ALL_METEO_COLS if c in combined.columns]
    missing_cols = [c for c in ALL_METEO_COLS if c not in combined.columns]
    if missing_cols:
        print(f"  WARNING: These columns not found in any station: {missing_cols}")

    print(f"\n  Combined dataset : {len(combined)} daily records")
    print(f"  Date range       : {combined[COL_DATE].min().date()} → "
          f"{combined[COL_DATE].max().date()}")
    print(f"  Features present : {present}")
    return combined


# =============================================================================
# 2. PREPROCESSING
# =============================================================================
def impute_missing(df):
    """7-day rolling median + linear interpolation for remaining gaps."""
    num_cols = [c for c in ALL_METEO_COLS if c in df.columns]
    for col in num_cols:
        if col == COL_RR:
            # Rainfall: missing = 0 (no rain recorded)
            df[col] = df[col].fillna(0.0)
        else:
            rolling = df[col].fillna(
                df[col].rolling(7, min_periods=1, center=True).median()
            )
            df[col] = rolling.interpolate(method='linear', limit_direction='both')

    remaining = df[num_cols].isna().sum().sum()
    print(f"  Missing values after imputation: {remaining}")
    if remaining > 0:
        for col in num_cols:
            n = df[col].isna().sum()
            if n > 0:
                print(f"    {col}: {n} remaining NaN → filled with column median")
                df[col] = df[col].fillna(df[col].median())
    return df


def encode_circular_wind(df):
    """Decompose ddd_x (degrees) into sin/cos. ddd_car is skipped (text)."""
    if COL_DDDX in df.columns:
        rad = pd.to_numeric(df[COL_DDDX], errors='coerce') * (np.pi / 180.0)
        df[COL_DDDX + '_sin'] = np.sin(rad).fillna(0.0)
        df[COL_DDDX + '_cos'] = np.cos(rad).fillna(0.0)
        df.drop(columns=[COL_DDDX], inplace=True)
        print(f"  Wind direction encoded: {COL_DDDX} → sin/cos")
    return df


# =============================================================================
# 3. FEATURE ENGINEERING
# =============================================================================
def add_lag_features(df, cols, lags=(1, 2, 3)):
    new_cols = []
    for col in cols:
        if col in df.columns:
            for lag in lags:
                name = f"{col}_lag{lag}"
                df[name] = df[col].shift(lag)
                new_cols.append(name)
    print(f"  Lag features created: {len(new_cols)} columns (lags {lags})")
    return df, new_cols


def compute_api(df, k=K_DECAY):
    """
    Antecedent Precipitation Index: API_t = RR_t + k * API_{t-1}
    Uses only past RR — zero data leakage.
    """
    rr = df[COL_RR].values.copy()
    api = np.zeros(len(rr))
    api[0] = rr[0]
    for i in range(1, len(rr)):
        api[i] = rr[i] + k * api[i - 1]
    df['API'] = api
    print(f"  API computed (k={k}):  "
          f"min={api.min():.2f}  max={api.max():.2f}  mean={api.mean():.2f}")
    return df


def create_target(df):
    """y_t = 1 if RR_{t+1} >= FLOOD_THRESHOLD, else 0."""
    df['target'] = (df[COL_RR].shift(-1) >= FLOOD_THRESHOLD).astype(int)
    n_pos   = int(df['target'].sum())
    n_total = int(df['target'].notna().sum())
    pct     = 100 * n_pos / n_total if n_total > 0 else 0
    print(f"  Target created: {n_pos} flood-risk days / {n_total} total ({pct:.1f}%)")
    return df


# =============================================================================
# 4. DATASET PREPARATION
# =============================================================================
def prepare_datasets(df):
    # Drop last row (target is NaN — no next-day RR available)
    df = df.dropna(subset=['target']).copy()

    # Feature lists
    exclude = {'target', 'API', COL_DATE}
    features_A = [c for c in df.columns if c not in exclude]
    features_B = features_A + ['API']

    # Fill any residual NaNs in features with 0
    df[features_B] = df[features_B].fillna(0.0)

    y   = df['target'].values.astype(int)
    X_A = df[features_A].values.astype(float)
    X_B = df[features_B].values.astype(float)

    n     = len(df)
    split = int(n * TRAIN_RATIO)

    X_A_tr, X_A_te = X_A[:split], X_A[split:]
    X_B_tr, X_B_te = X_B[:split], X_B[split:]
    y_tr,   y_te   = y[:split],   y[split:]

    # Min-Max scaling — fit on train only
    sc_A = MinMaxScaler()
    X_A_tr = sc_A.fit_transform(X_A_tr)
    X_A_te = sc_A.transform(X_A_te)

    sc_B = MinMaxScaler()
    X_B_tr = sc_B.fit_transform(X_B_tr)
    X_B_te = sc_B.transform(X_B_te)

    print(f"\n  Total rows      : {n}")
    print(f"  Train           : {len(y_tr)}  ({y_tr.sum()} flood days)")
    print(f"  Test            : {len(y_te)}  ({y_te.sum()} flood days, "
          f"{100*y_te.mean():.1f}%)")
    print(f"  Model A features: {len(features_A)}")
    print(f"  Model B features: {len(features_B)} (+API)")

    return X_A_tr, X_A_te, X_B_tr, X_B_te, y_tr, y_te, features_A, features_B


# =============================================================================
# 5. EVALUATION
# =============================================================================
def evaluate(y_true, y_pred, y_prob, label="Model"):
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    accuracy  = float((y_true == y_pred).mean())
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float('nan')
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    print(f"\n  [{label}]")
    print(f"    Recall    : {recall:.4f}")
    print(f"    F1-Score  : {f1:.4f}")
    print(f"    Precision : {precision:.4f}")
    print(f"    Accuracy  : {accuracy:.4f}")
    print(f"    AUC-ROC   : {auc:.4f}")
    print(f"    CSI       : {csi:.4f}")
    print(f"    TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    return dict(recall=recall, f1=f1, precision=precision,
                accuracy=accuracy, auc=auc, csi=csi,
                tp=tp, fp=fp, fn=fn, tn=tn)


def evaluate_optimal_threshold(y_true, y_prob, label="Model"):
    """Find the probability threshold that maximises F1, then re-evaluate."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx    = int(np.argmax(f1s))
    best_thresh = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    best_pred   = (y_prob >= best_thresh).astype(int)

    r   = recall_score(y_true, best_pred, zero_division=0)
    p   = precision_score(y_true, best_pred, zero_division=0)
    f1  = f1_score(y_true, best_pred, zero_division=0)
    tp  = int(((y_true == 1) & (best_pred == 1)).sum())
    fp  = int(((y_true == 0) & (best_pred == 1)).sum())
    fn  = int(((y_true == 1) & (best_pred == 0)).sum())

    print(f"\n  [{label}] Optimal threshold = {best_thresh:.3f}")
    print(f"    Recall={r:.4f}  Precision={p:.4f}  F1={f1:.4f}")
    print(f"    TP={tp}  FP={fp}  FN={fn}")
    return dict(threshold=best_thresh, recall=r, precision=p, f1=f1,
                tp=tp, fp=fp, fn=fn)


# =============================================================================
# 6. CROSS-VALIDATION
# =============================================================================
def run_cv(X_train, y_train, params, label="Model"):
    tscv         = TimeSeriesSplit(n_splits=N_FOLDS)
    fold_results = []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        sc    = MinMaxScaler()
        X_tr  = sc.fit_transform(X_tr)
        X_val = sc.transform(X_val)

        clf   = RandomForestClassifier(**params)
        clf.fit(X_tr, y_tr)
        pred  = clf.predict(X_val)
        prob  = clf.predict_proba(X_val)[:, 1]

        fold_results.append({
            'fold'     : fold,
            'recall'   : recall_score(y_val, pred, zero_division=0),
            'f1'       : f1_score(y_val, pred, zero_division=0),
            'precision': precision_score(y_val, pred, zero_division=0),
            'auc'      : roc_auc_score(y_val, prob)
                         if len(np.unique(y_val)) > 1 else float('nan'),
        })

    df_cv = pd.DataFrame(fold_results)
    print(f"\n  [{label}] Cross-Validation ({N_FOLDS}-fold TimeSeriesSplit)")
    for m in ['recall', 'f1', 'precision', 'auc']:
        v = df_cv[m].dropna()
        print(f"    {m:9s}: {v.mean():.4f} ± {v.std():.4f}")
    return df_cv


# =============================================================================
# 7. PLOTS
# =============================================================================
def plot_confusion_matrices(cm_A, cm_B, out_dir, filename='confusion_matrices.png'):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, cm, title in zip(
        axes, [cm_A, cm_B],
        ['Model A — Baseline RF', 'Model B — RF-AMM']
    ):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Normal', 'Flood'],
                    yticklabels=['Normal', 'Flood'])
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
    plt.tight_layout()
    path = os.path.join(out_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_cv_comparison(cv_A, cv_B, out_dir):
    metrics  = ['recall', 'f1', 'precision', 'auc']
    x, w     = np.arange(len(metrics)), 0.32
    means_A  = [cv_A[m].mean() for m in metrics]
    means_B  = [cv_B[m].mean() for m in metrics]
    stds_A   = [cv_A[m].std()  for m in metrics]
    stds_B   = [cv_B[m].std()  for m in metrics]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, means_A, w, yerr=stds_A, label='Model A (Baseline)',
           color='#5E82B4', capsize=4, error_kw={'elinewidth': 1.2})
    ax.bar(x + w/2, means_B, w, yerr=stds_B, label='Model B (RF-AMM)',
           color='#E87040', capsize=4, error_kw={'elinewidth': 1.2})
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('Cross-validation comparison (mean ± std, 5-fold TimeSeriesSplit)')
    ax.legend()
    ax.axhline(0.5, color='gray', linewidth=0.7, linestyle='--', alpha=0.5)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    path = os.path.join(out_dir, 'cv_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_feature_importance(model_A, model_B, feat_A, feat_B, out_dir, top_n=15):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, model, features, title in zip(
        axes, [model_A, model_B], [feat_A, feat_B],
        ['Model A — Baseline RF', 'Model B — RF-AMM']
    ):
        imp    = pd.Series(model.feature_importances_, index=features)
        imp    = imp.nlargest(top_n).sort_values()
        colors = ['#E87040' if f == 'API' else '#5E82B4' for f in imp.index]
        imp.plot.barh(ax=ax, color=colors)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Gini importance')

    from matplotlib.patches import Patch
    axes[1].legend(
        handles=[Patch(facecolor='#E87040', label='API feature'),
                 Patch(facecolor='#5E82B4', label='Other features')],
        loc='lower right', fontsize=9
    )
    plt.suptitle(f'Top-{top_n} feature importances (Gini)', y=1.01, fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, 'feature_importance.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_api_series(df, out_dir):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(df[COL_DATE], df[COL_RR], color='#5E82B4',
                 linewidth=0.8, label='Daily RR')
    axes[0].axhline(FLOOD_THRESHOLD, color='red', linewidth=0.8,
                    linestyle='--', label=f'Flood threshold ({FLOOD_THRESHOLD} mm/day)')
    axes[0].set_ylabel('RR (mm/day)')
    axes[0].legend(fontsize=8)
    axes[0].set_title('Daily Rainfall and Antecedent Precipitation Index — DKI Jakarta')

    axes[1].plot(df[COL_DATE], df['API'], color='#E87040',
                 linewidth=0.9, label='API')
    axes[1].set_ylabel('API (mm-equivalent)')
    axes[1].legend(fontsize=8)
    axes[1].set_xlabel('Date')

    plt.tight_layout()
    path = os.path.join(out_dir, 'api_series.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# 8. RESULTS EXPORT
# =============================================================================
def export_results(metrics_A, metrics_B, cv_A, cv_B,
                   opt_A, opt_B, wilcoxon_stat, wilcoxon_p, out_dir):
    rows = []
    for metric in ['recall', 'f1', 'precision', 'accuracy', 'auc', 'csi']:
        rows.append({
            'Metric'         : metric.upper(),
            'Model A (test)' : f"{metrics_A[metric]:.4f}",
            'Model B (test)' : f"{metrics_B[metric]:.4f}",
            'Delta (B-A)'    : f"{metrics_B[metric] - metrics_A[metric]:+.4f}",
            'CV A mean±std'  : (f"{cv_A[metric].mean():.4f}±{cv_A[metric].std():.4f}"
                                if metric in cv_A.columns else '—'),
            'CV B mean±std'  : (f"{cv_B[metric].mean():.4f}±{cv_B[metric].std():.4f}"
                                if metric in cv_B.columns else '—'),
        })
    df_results = pd.DataFrame(rows)

    cm_rows = pd.DataFrame([
        {'Model': 'A (Baseline)',
         'TP': metrics_A['tp'], 'FP': metrics_A['fp'],
         'FN': metrics_A['fn'], 'TN': metrics_A['tn']},
        {'Model': 'B (RF-AMM)',
         'TP': metrics_B['tp'], 'FP': metrics_B['fp'],
         'FN': metrics_B['fn'], 'TN': metrics_B['tn']},
    ])

    opt_rows = pd.DataFrame([
        {'Model': 'A (Baseline)', **opt_A},
        {'Model': 'B (RF-AMM)',   **opt_B},
    ])

    stat_row = pd.DataFrame([{
        'Test'                  : 'Wilcoxon signed-rank (F1, 5-fold)',
        'Statistic'             : f"{wilcoxon_stat:.4f}",
        'p-value'               : f"{wilcoxon_p:.4f}",
        'Significant (α=0.05)'  : 'YES' if wilcoxon_p < 0.05 else 'NO',
    }])

    path = os.path.join(out_dir, 'results_summary.xlsx')
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        df_results.to_excel(writer, sheet_name='Metrics Comparison',  index=False)
        cm_rows.to_excel(   writer, sheet_name='Confusion Matrices',  index=False)
        opt_rows.to_excel(  writer, sheet_name='Optimal Threshold',   index=False)
        stat_row.to_excel(  writer, sheet_name='Statistical Test',    index=False)
        cv_A.to_excel(      writer, sheet_name='CV Model A',          index=False)
        cv_B.to_excel(      writer, sheet_name='CV Model B',          index=False)

    print(f"  Saved: {path}")
    return df_results


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 65)
    print("  RF vs RF-AMM — Flood Prediction Experiment")
    print("=" * 65)

    # Step 1 — Load
    print("\n[1] Loading BMKG data...")
    df = load_bmkg_files(DATA_DIR)

    # Step 2 — Preprocess
    print("\n[2] Preprocessing...")
    df = impute_missing(df)
    df = encode_circular_wind(df)
    print(f"  Columns after preprocessing: {[c for c in df.columns if c != COL_DATE]}")

    # Step 3 — Feature engineering
    print("\n[3] Feature engineering...")
    feat_cols = [c for c in df.columns if c != COL_DATE]
    df, lag_cols = add_lag_features(df, feat_cols, lags=(1, 2, 3))
    df = compute_api(df, k=K_DECAY)
    df = create_target(df)
    plot_api_series(df.copy(), OUTPUT_DIR)

    # Step 4 — Prepare
    print("\n[4] Preparing datasets...")
    (X_A_tr, X_A_te, X_B_tr, X_B_te,
     y_tr, y_te, feat_A, feat_B) = prepare_datasets(df)

    # Step 5 — Cross-validation
    print("\n[5] Running cross-validation...")
    cv_A = run_cv(X_A_tr, y_tr, RF_PARAMS, label="Model A (Baseline)")
    cv_B = run_cv(X_B_tr, y_tr, RF_PARAMS, label="Model B (RF-AMM)")

    # Step 6 — Train final models
    print("\n[6] Training final models...")
    model_A = RandomForestClassifier(**RF_PARAMS)
    model_A.fit(X_A_tr, y_tr)
    model_B = RandomForestClassifier(**RF_PARAMS)
    model_B.fit(X_B_tr, y_tr)

    # Step 7 — Evaluate (default threshold 0.5)
    print("\n[7] Evaluating on test set (threshold = 0.5)...")
    pred_A   = model_A.predict(X_A_te)
    prob_A   = model_A.predict_proba(X_A_te)[:, 1]
    metrics_A = evaluate(y_te, pred_A, prob_A, label="Model A (Baseline RF)")

    pred_B   = model_B.predict(X_B_te)
    prob_B   = model_B.predict_proba(X_B_te)[:, 1]
    metrics_B = evaluate(y_te, pred_B, prob_B, label="Model B (RF-AMM)")

    # Step 7b — Optimal threshold
    print("\n[7b] Optimal threshold analysis...")
    opt_A = evaluate_optimal_threshold(y_te, prob_A, label="Model A")
    opt_B = evaluate_optimal_threshold(y_te, prob_B, label="Model B")

    # Step 8 — Wilcoxon test
    print("\n[8] Wilcoxon signed-rank test on CV F1 scores...")
    try:
        stat, p_val = wilcoxon(cv_B['f1'].values, cv_A['f1'].values)
        print(f"  Statistic: {stat:.4f}  |  p-value: {p_val:.4f}")
        sig = 'SIGNIFICANT (p < 0.05)' if p_val < 0.05 else 'NOT significant (p >= 0.05)'
        print(f"  Result: {sig}")
    except Exception as e:
        print(f"  Wilcoxon test skipped: {e}")
        stat, p_val = float('nan'), float('nan')

    # Step 9 — Plots
    print("\n[9] Generating plots...")
    # Default threshold
    plot_confusion_matrices(
        confusion_matrix(y_te, pred_A),
        confusion_matrix(y_te, pred_B),
        OUTPUT_DIR, filename='confusion_matrices_default.png'
    )
    # Optimal threshold
    pred_A_opt = (prob_A >= opt_A['threshold']).astype(int)
    pred_B_opt = (prob_B >= opt_B['threshold']).astype(int)
    plot_confusion_matrices(
        confusion_matrix(y_te, pred_A_opt),
        confusion_matrix(y_te, pred_B_opt),
        OUTPUT_DIR, filename='confusion_matrices_optimal.png'
    )
    plot_cv_comparison(cv_A, cv_B, OUTPUT_DIR)
    plot_feature_importance(model_A, model_B, feat_A, feat_B, OUTPUT_DIR)

    # Step 10 — Export
    print("\n[10] Exporting results...")
    df_results = export_results(
        metrics_A, metrics_B, cv_A, cv_B,
        opt_A, opt_B, stat, p_val, OUTPUT_DIR
    )

    print("\n" + "=" * 65)
    print("  EXPERIMENT COMPLETE")
    print("=" * 65)
    print(df_results.to_string(index=False))
    print(f"\n  Outputs saved to: {os.path.abspath(OUTPUT_DIR)}/")
    print("=" * 65)


if __name__ == "__main__":
    main()