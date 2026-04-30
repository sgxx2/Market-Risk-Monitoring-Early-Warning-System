"""
Batch Model Comparison for SP500 Crash Prediction

This script tests multiple feature combinations and saves all results to a single Excel file.

Feature Z-Score Rules (Dynamic - based on feature name patterns):
- NO z-score:
  * logret (return features)
  * diff (difference features)
  * Regime status: P_Calm, P_Transitional, P_Stress, Delta_P_Stress
  * Interaction features: *INTERACT*, *STRESS*, *SPIKE*, SPREAD_VIX_HIGH
  * RQA features: *RR_60d, *DET_60d
  * Ratio features: *RATIO*, SPX_MAXDD_20D (is a ratio)
  * CREDIT_FACTOR (spread, already a diff)

- YES z-score:
  * range features (*_range, *_range_5d, *_range_10d, *_range_q252)
  * SPX_VOL_20D* (volatility level)
  * RATE_SLOPE_10Y_3M* (interest rate slope)
  * VIX_price* (VIX price rolling - is price level)

- EXCEPTION: Delta_P_Stress_5d_z252 (manually specified with z-score)

Feature Groups:
1. Group 1: 14 manually selected features
2. Group 2: Regime probability features
3. Group 3: Group 1 without regime-related features
4. Group 4: Top 20 features from feature_importance_sp500.csv (z-score rules auto-applied)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import ParameterGrid
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("BATCH MODEL COMPARISON FOR SP500 CRASH PREDICTION")
print("="*70)

# ============================================================
# Load Data
# ============================================================

print("\n[1] Loading data...")

features = pd.read_csv('features.csv', index_col=0, parse_dates=True)
print(f"    Features shape: {features.shape}")

target_labels = pd.read_csv('target_label.csv', index_col=0, parse_dates=True)
print(f"    Target labels shape: {target_labels.shape}")

regime_labels = pd.read_csv('regime_label.csv', index_col=0, parse_dates=True)
print(f"    Regime labels shape: {regime_labels.shape}")

# Load feature importance
feature_importance = pd.read_csv('feature_importance_sp500.csv')
print(f"    Feature importance loaded: {len(feature_importance)} features")

# ============================================================
# Helper Functions
# ============================================================

def calc_zscore_252(series, min_periods=200):
    """Calculate 252-day rolling z-score"""
    rolling_mean = series.rolling(window=252, min_periods=min_periods).mean()
    rolling_std = series.rolling(window=252, min_periods=min_periods).std()
    zscore = (series - rolling_mean) / rolling_std.clip(lower=1e-8)
    return zscore

def ensure_zscore_features(features_df, feature_list):
    """Ensure all required z-score features exist"""
    # Features that need z-score conversion (range features)
    range_features_to_zscore = {
        'IEF_range_5d': 'IEF_range_5d_z252',
        'EUROSTOXX50_range': 'EUROSTOXX50_range_z252',
        'COMMODITY_range': 'COMMODITY_range_z252',
        'HYG_range': 'HYG_range_z252',
        'SP500_range': 'SP500_range_z252',
        'EEM_range': 'EEM_range_z252',
        'EEM_range_5d': 'EEM_range_5d_z252',
        'HYG_range_5d': 'HYG_range_5d_z252',
        'EUROSTOXX50_range_5d': 'EUROSTOXX50_range_5d_z252',
        'NASDAQ100_range': 'NASDAQ100_range_z252',
        'IEF_range': 'IEF_range_z252',
        'SP500_range_5d': 'SP500_range_z252',
        'USDJPY_range_5d': 'USDJPY_range_5d_z252',
        'EURUSD_range': 'EURUSD_range_z252',
    }

    created = []
    for orig_feat, zscore_feat in range_features_to_zscore.items():
        if zscore_feat in feature_list and zscore_feat not in features_df.columns:
            if orig_feat in features_df.columns:
                features_df[zscore_feat] = calc_zscore_252(features_df[orig_feat])
                created.append(zscore_feat)

    return features_df, created

def run_model_comparison(X_train, y_train, X_val, y_val, X_test, y_test, feature_names, group_name):
    """
    Workflow per group:
    1) Train different hyperparameters on Train set.
    2) Select best model by Validation PR-AUC (across LR/RF/LGB).
    3) Refit each tuned model on Train+Validation and evaluate all on Test.
    4) Select best model by Validation PR-AUC and report Test metrics + baseline.
    """
    print(f"\n    Training models for {group_name}...")

    X_train_raw = X_train.values
    X_val_raw = X_val.values
    X_test_raw = X_test.values

    # Scaler for LR tuning on train/val
    scaler_train = StandardScaler()
    X_train_scaled = scaler_train.fit_transform(X_train.values)
    X_val_scaled = scaler_train.transform(X_val.values)

    X_train_val = pd.concat([X_train, X_val], axis=0)
    y_train_val = pd.concat([y_train, y_val], axis=0)

    results = {
        'Group': group_name,
        'Num_Features': len(feature_names),
        'Features': ', '.join(feature_names[:5]) + ('...' if len(feature_names) > 5 else ''),
        'Train_Samples': len(X_train),
        'Val_Samples': len(X_val),
        'TrainVal_Samples': len(X_train_val),
        'Test_Samples': len(X_test),
        'Test_Positive_Rate': y_test.mean()
    }

    candidates = []

    # ---- L1 Logistic Regression: tune on train, select by val ----
    print("      - Tuning L1 Logistic Regression on Train, scoring on Val...", end=' ', flush=True)
    lr_param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'class_weight': [None, 'balanced']
    }
    best_lr_score = -1
    best_lr_params = None
    for params in ParameterGrid(lr_param_grid):
        model = LogisticRegression(penalty='l1', solver='saga', max_iter=2000, random_state=42, **params)
        model.fit(X_train_scaled, y_train)
        y_val_prob = model.predict_proba(X_val_scaled)[:, 1]
        score = average_precision_score(y_val, y_val_prob)
        if score > best_lr_score:
            best_lr_score = score
            best_lr_params = params
    print(f"best val PR-AUC={best_lr_score:.4f}")
    candidates.append({'model_name': 'L1 Logistic', 'val_pr_auc': best_lr_score, 'params': best_lr_params})

    # ---- Random Forest ----
    print("      - Tuning Random Forest on Train, scoring on Val...", end=' ', flush=True)
    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 7, 10],
        'min_samples_split': [5, 10],
        'class_weight': [None, 'balanced'],
        'min_samples_leaf': [2],
    }
    best_rf_score = -1
    best_rf_params = None
    for params in ParameterGrid(rf_param_grid):
        model = RandomForestClassifier(random_state=42, n_jobs=-1, **params)
        model.fit(X_train_raw, y_train)
        y_val_prob = model.predict_proba(X_val_raw)[:, 1]
        score = average_precision_score(y_val, y_val_prob)
        if score > best_rf_score:
            best_rf_score = score
            best_rf_params = params
    print(f"best val PR-AUC={best_rf_score:.4f}")
    candidates.append({'model_name': 'Random Forest', 'val_pr_auc': best_rf_score, 'params': best_rf_params})

    # ---- LightGBM ----
    print("      - Tuning LightGBM on Train, scoring on Val...", end=' ', flush=True)
    lgb_param_grid = {
        'n_estimators': [200, 400],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05],
        'num_leaves': [8, 15, 31],
        'min_child_samples': [30, 40, 50],
        'scale_pos_weight': [15, 20, 25, 30],
        'subsample': [0.8],
        'colsample_bytree': [0.8]
    }
    best_lgb_score = -1
    best_lgb_params = None
    for params in ParameterGrid(lgb_param_grid):
        model = lgb.LGBMClassifier(random_state=42, verbose=-1, **params)
        model.fit(X_train_raw, y_train)
        y_val_prob = model.predict_proba(X_val_raw)[:, 1]
        score = average_precision_score(y_val, y_val_prob)
        if score > best_lgb_score:
            best_lgb_score = score
            best_lgb_params = params
    print(f"best val PR-AUC={best_lgb_score:.4f}")
    candidates.append({'model_name': 'LightGBM', 'val_pr_auc': best_lgb_score, 'params': best_lgb_params})

    # Keep per-model validation results
    results['LR_Val_PR_AUC'] = best_lr_score
    results['RF_Val_PR_AUC'] = best_rf_score
    results['LGB_Val_PR_AUC'] = best_lgb_score

    # Refit all tuned models on Train+Validation and evaluate on Test
    n_top = int(len(y_test) * 0.10)
    total_positives = int(np.array(y_test).sum())
    y_test_array = np.array(y_test)

    def safe_roc_auc(y_true, y_prob):
        if pd.Series(y_true).nunique() < 2:
            return np.nan
        return roc_auc_score(y_true, y_prob)

    def eval_test_metrics(y_prob, prefix):
        if n_top > 0:
            top_indices = np.argsort(y_prob)[-n_top:]
            tp_in_top = int(y_test_array[top_indices].sum())
        else:
            tp_in_top = 0

        results[f'{prefix}_Test_ROC_AUC'] = safe_roc_auc(y_test, y_prob)
        results[f'{prefix}_Test_PR_AUC'] = average_precision_score(y_test, y_prob)
        results[f'{prefix}_Top10%_Count'] = n_top
        results[f'{prefix}_TP@10%'] = tp_in_top
        results[f'{prefix}_Recall@10%'] = tp_in_top / total_positives if total_positives > 0 else 0
        results[f'{prefix}_Precision@10%'] = tp_in_top / n_top if n_top > 0 else 0

    # LR refit + test
    scaler_train_val_lr = StandardScaler()
    X_train_val_scaled_lr = scaler_train_val_lr.fit_transform(X_train_val.values)
    X_test_scaled_lr = scaler_train_val_lr.transform(X_test.values)
    lr_final = LogisticRegression(
        penalty='l1', solver='saga', max_iter=2000, random_state=42, **best_lr_params
    )
    lr_final.fit(X_train_val_scaled_lr, y_train_val)
    y_test_prob_lr = lr_final.predict_proba(X_test_scaled_lr)[:, 1]
    eval_test_metrics(y_test_prob_lr, 'LR')

    # RF refit + test
    rf_final = RandomForestClassifier(random_state=42, n_jobs=-1, **best_rf_params)
    rf_final.fit(X_train_val.values, y_train_val)
    y_test_prob_rf = rf_final.predict_proba(X_test_raw)[:, 1]
    eval_test_metrics(y_test_prob_rf, 'RF')

    # LGB refit + test
    lgb_final = lgb.LGBMClassifier(random_state=42, verbose=-1, **best_lgb_params)
    lgb_final.fit(X_train_val.values, y_train_val)
    y_test_prob_lgb = lgb_final.predict_proba(X_test_raw)[:, 1]
    eval_test_metrics(y_test_prob_lgb, 'LGB')

    # Baseline (random ranking / class-prior expectation)
    # For random ranking: expected AP ~= positive rate, ROC-AUC ~= 0.5
    results['Baseline_Model'] = 'RandomRank'
    results['Baseline_Test_ROC_AUC'] = 0.5 if pd.Series(y_test).nunique() >= 2 else np.nan
    results['Baseline_Test_PR_AUC'] = float(y_test.mean()) if len(y_test) > 0 else np.nan
    results['Baseline_Top10%_Count'] = n_top
    expected_tp = total_positives * 0.10 if n_top > 0 else 0.0
    results['Baseline_TP@10%'] = expected_tp
    results['Baseline_Recall@10%'] = 0.10 if (total_positives > 0 and n_top > 0) else 0
    results['Baseline_Precision@10%'] = float(y_test.mean()) if n_top > 0 else 0
    results['Total_Crashes'] = total_positives

    # Select best model by Validation PR-AUC
    best_candidate = max(candidates, key=lambda x: x['val_pr_auc'])
    best_model_name = best_candidate['model_name']
    best_params = best_candidate['params']

    results['Best_Model'] = best_model_name
    results['Best_Val_PR_AUC'] = best_candidate['val_pr_auc']
    results['Best_Params'] = str(best_params)

    print(f"      - Selected by Val PR-AUC: {best_model_name} ({results['Best_Val_PR_AUC']:.4f})")
    # Map selected model's test metrics
    if best_model_name == 'L1 Logistic':
        prefix = 'LR'
    elif best_model_name == 'Random Forest':
        prefix = 'RF'
    else:
        prefix = 'LGB'

    results['Best_Test_ROC_AUC'] = results[f'{prefix}_Test_ROC_AUC']
    results['Best_Test_PR_AUC'] = results[f'{prefix}_Test_PR_AUC']
    results['Best_Top10%_Count'] = results[f'{prefix}_Top10%_Count']
    results['Best_TP@10%'] = results[f'{prefix}_TP@10%']
    results['Best_Recall@10%'] = results[f'{prefix}_Recall@10%']
    results['Best_Precision@10%'] = results[f'{prefix}_Precision@10%']

    return results, feature_names


# ============================================================
# Define Feature Groups
# ============================================================

print("\n[2] Defining feature groups...")

# ==============================================
# Feature Z-Score Rules:
# - NO z-score: return, diff, regime status, interaction features, RQA features
# - YES z-score: range, level, price features
# ==============================================

# Add regime probability features to main features dataframe (NO z-score for regime)
for feat in ['P_Calm', 'P_Transitional', 'P_Stress']:
    if feat in regime_labels.columns:
        features[feat] = regime_labels[feat]

# Calculate Delta_P_Stress_5d_z252 for model input
# Step 1: 5-day rolling mean, Step 2: 252-day z-score on the rolling result
if 'Delta_P_Stress' in regime_labels.columns:
    Delta_P_Stress = regime_labels['Delta_P_Stress'].reindex(features.index)
    # 5-day rolling mean
    Delta_P_Stress_5d = Delta_P_Stress.rolling(window=5, min_periods=4).mean()
    # 252-day z-score on the 5d rolling result
    rolling_mean = Delta_P_Stress_5d.rolling(window=252, min_periods=202).mean()
    rolling_std = Delta_P_Stress_5d.rolling(window=252, min_periods=202).std()
    features['Delta_P_Stress_5d_z252'] = (Delta_P_Stress_5d - rolling_mean) / rolling_std.clip(lower=1e-8)
    print(f"    Created Delta_P_Stress_5d_z252 (z-score on 5d rolling)")

# Group 1: 14 selected features (applying z-score rules)
# - range/level features: need z-score
# - return/interaction/RQA/regime: no z-score
group1_features = [
    'CREDIT_FACTOR_RR_60d',      # RQA feature - NO z-score
    'SPX_VOL_STRESS',            # interaction feature - NO z-score
    'EUROSTOXX50_range_z252',    # range - z-score
    'DXY_SPREAD_INTERACT',       # interaction feature - NO z-score
    'COMMODITY_range_z252',      # range - z-score
    'VIX_RR_60d',                # RQA feature - NO z-score
    'HYG_range_z252',            # range - z-score
    'HYG_RANGE_STRESS',          # interaction feature - NO z-score
    'HYG_logret',                # return - NO z-score
    'SP500_range_z252',          # range - z-score
    'VIX_VOL_RATIO',             # ratio - NO z-score
    'P_Stress',                  # regime status - NO z-score
    'IEF_range_5d_z252',         # range - z-score
    'NASDAQ_RANGE_STRESS',       # interaction feature - NO z-score
]
print(f"    Group 1: {len(group1_features)} features (14 selected features)")

# Group 2: Regime probability features only (use Delta_P_Stress_5d_z252 for model input)
group2_features = ['P_Calm', 'P_Transitional', 'P_Stress', 'Delta_P_Stress_5d_z252']
print(f"    Group 2: {len(group2_features)} features (Regime probability only, Delta_P_Stress as 5d rolling + 252d z-score)")

# Group 3: Group 1 without regime-related features
regime_related = ['P_Stress', 'SPX_VOL_STRESS', 'HYG_RANGE_STRESS', 'NASDAQ_RANGE_STRESS']
group3_features = [f for f in group1_features if f not in regime_related]
print(f"    Group 3: {len(group3_features)} features (Group 1 without regime-related)")

# ============================================================
# Dynamic Z-Score Rule Function
# ============================================================

def needs_zscore(feature_name):
    """
    Determine if a feature needs z-score based on its name.

    Rules:
    - NO z-score: logret, diff, regime status, interaction, RQA, ratio, CREDIT_FACTOR,
                  SPREAD_VIX_HIGH, SPX_VIX_SPIKE, Delta_P_Stress, SPX_MAXDD_20D (is ratio)
    - YES z-score: range features, SPX_VOL_20D, RATE_SLOPE_10Y_3M, VIX_price (is price level)

    Returns:
    --------
    bool : True if needs z-score, False otherwise
    """
    # Patterns that should NOT have z-score
    no_zscore_patterns = [
        'logret',           # return features
        'diff',             # difference features
        'P_Calm', 'P_Transitional', 'P_Stress',  # regime status
        'Delta_P_Stress',   # regime change (except z252 version)
        'INTERACT',         # interaction features
        'STRESS',           # interaction features (SPX_VOL_STRESS, HYG_RANGE_STRESS, etc.)
        'SPIKE',            # interaction features
        'RR_60d', 'DET_60d', # RQA features
        'CREDIT_FACTOR',    # spread (already a diff)
        'RATIO',            # ratio features
        'SPREAD_VIX_HIGH',  # interaction feature
        'SPX_MAXDD_20D',    # max drawdown is already a ratio
    ]

    # Check if feature matches any no-zscore pattern
    for pattern in no_zscore_patterns:
        if pattern in feature_name:
            # Exception: Delta_P_Stress_5d_z252 already has z-score
            if feature_name == 'Delta_P_Stress_5d_z252':
                return False
            return False

    # Patterns that NEED z-score
    zscore_patterns = [
        '_range',           # range features
        'SPX_VOL_20D',      # volatility level
        'RATE_SLOPE_10Y_3M', # interest rate slope level
        'VIX_price',        # VIX price level rolling
    ]

    for pattern in zscore_patterns:
        if pattern in feature_name:
            # Already has z-score suffix
            if '_z252' in feature_name:
                return False
            return True

    return False


def get_zscore_name(feature_name):
    """Get the z-score version name of a feature."""
    if '_z252' in feature_name:
        return feature_name
    return f"{feature_name}_z252"


# Group 4: Top 20 features from feature_importance_sp500.csv
# Apply z-score dynamically based on feature type
top20_raw = feature_importance.head(20)['Feature'].tolist()
print(f"    Group 4: Top 20 from feature_importance_sp500.csv")

group4_features = []
for f in top20_raw:
    if needs_zscore(f):
        group4_features.append(get_zscore_name(f))
    else:
        group4_features.append(f)

print(f"    Group 4 features (after z-score rules): {group4_features}")

# ============================================================
# Create Z-Score Features Dynamically
# ============================================================

print("\n[3] Creating z-score features for range/level features...")

# Find all features that need z-score and create them
zscore_created = []
for col in features.columns:
    if needs_zscore(col):
        zscore_name = get_zscore_name(col)
        if zscore_name not in features.columns:
            features[zscore_name] = calc_zscore_252(features[col])
            zscore_created.append(zscore_name)

if zscore_created:
    print(f"    Created {len(zscore_created)} z-score features:")
    for f in zscore_created[:10]:
        print(f"      - {f}")
    if len(zscore_created) > 10:
        print(f"      ... and {len(zscore_created) - 10} more")
else:
    print("    All required z-score features already exist")

print("\n    Z-Score Rules Applied:")
print("    - NO z-score: logret, diff, regime, interaction, RQA, ratio, CREDIT_FACTOR, SPX_MAXDD_20D")
print("    - YES z-score: range, SPX_VOL_20D, RATE_SLOPE_10Y_3M, VIX_price")

# ============================================================
# Data Split Setup
# ============================================================

print("\n[4] Setting up data split...")

# Keep test set fixed by Data_Split=='Test'
test_index = regime_labels[regime_labels['Data_Split'] == 'Test'].index
test_start_date = test_index.min()
test_end_date = test_index.max()
test_count = len(test_index)

print(f"    Fixed Test window: {test_start_date.date()} to {test_end_date.date()} ({test_count} rows in regime_label)")
print("    Train/Validation split rule: after excluding fixed test rows, split remaining data 2:1 by time order")

# Get target labels
y_all = target_labels['SP500_Label'].copy()

# ============================================================
# Run Batch Comparison
# ============================================================

print("\n" + "="*70)
print("[5] RUNNING BATCH MODEL COMPARISON")
print("="*70)

all_results = []
all_feature_details = {}
prepared_groups = []

feature_groups = [
    ('Group1_14Features', group1_features),
    ('Group2_RegimeProb', group2_features),
    ('Group3_NoRegime', group3_features),
    ('Group4_Top20', group4_features),
]

# Pass 1: Prepare datasets for each group
for group_name, feature_list in feature_groups:
    print(f"\n{'='*70}")
    print(f"Preparing data: {group_name}")
    print(f"{'='*70}")
    print(f"Features ({len(feature_list)}):")
    for i, f in enumerate(feature_list, 1):
        print(f"    {i:2d}. {f}")

    available_features = [f for f in feature_list if f in features.columns]
    missing_features = [f for f in feature_list if f not in features.columns]

    if missing_features:
        print(f"\n    WARNING: Missing features: {missing_features}")

    if len(available_features) == 0:
        print(f"    SKIPPED: No available features")
        continue

    X_all = features[available_features].copy()

    common_idx = X_all.index.intersection(y_all.dropna().index)
    X_all = X_all.loc[common_idx]
    y_aligned = y_all.loc[common_idx]

    valid_mask = X_all.notna().all(axis=1) & y_aligned.notna()
    X_all = X_all[valid_mask]
    y_aligned = y_aligned[valid_mask].astype(int)

    print(f"\n    Data after alignment: {len(X_all)} samples")
    print(f"    Date range: {X_all.index[0].date()} to {X_all.index[-1].date()}")
    print(f"    Positive samples: {y_aligned.sum()} ({100*y_aligned.mean():.2f}%)")

    # Sort by time before split
    X_all = X_all.sort_index()
    y_aligned = y_aligned.loc[X_all.index]

    # Fixed test set (date membership does not move)
    test_mask = X_all.index.isin(test_index)
    X_test = X_all[test_mask]
    y_test = y_aligned[test_mask]

    # Remaining rows are split into train/val with 2:1 ratio (time ordered)
    X_non_test = X_all[~test_mask]
    y_non_test = y_aligned[~test_mask]

    n_non_test = len(X_non_test)
    n_train = int(n_non_test * (2.0 / 3.0))
    n_train = max(1, min(n_train, n_non_test - 1)) if n_non_test >= 2 else 0

    X_train = X_non_test.iloc[:n_train]
    y_train = y_non_test.iloc[:n_train]
    X_val = X_non_test.iloc[n_train:]
    y_val = y_non_test.iloc[n_train:]

    print(f"    Split sizes (2:1 from non-test): Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    print(f"    Fixed test coverage in this group: {len(X_test)}/{test_count}")

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        print(f"    SKIPPED: Insufficient data in one or more splits")
        continue

    prepared_groups.append({
        'group_name': group_name,
        'available_features': available_features,
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test
    })

if len(prepared_groups) == 0:
    raise RuntimeError('No valid groups available after data preparation.')

# Pass 2: Train/tune/select using train/val, then evaluate on each group's fixed test rows
for g in prepared_groups:
    group_name = g['group_name']
    print(f"\n{'='*70}")
    print(f"Processing: {group_name}")
    print(f"{'='*70}")

    print(f"    Fixed Test samples used for evaluation: {len(g['X_test'])}")

    results, used_features = run_model_comparison(
        g['X_train'], g['y_train'],
        g['X_val'], g['y_val'],
        g['X_test'], g['y_test'],
        g['available_features'], group_name
    )

    all_results.append(results)
    all_feature_details[group_name] = used_features

    print(f"\n    Summary for {group_name}:")
    print(f"      Best Model (selected by Val PR-AUC): {results['Best_Model']} ({results['Best_Val_PR_AUC']:.4f})")
    print("      Test performance by model:")
    print(f"        LR : ROC-AUC={results['LR_Test_ROC_AUC']:.4f}, PR-AUC={results['LR_Test_PR_AUC']:.4f}, "
          f"R@10%={results['LR_Recall@10%']:.4f}, P@10%={results['LR_Precision@10%']:.4f}")
    print(f"        RF : ROC-AUC={results['RF_Test_ROC_AUC']:.4f}, PR-AUC={results['RF_Test_PR_AUC']:.4f}, "
          f"R@10%={results['RF_Recall@10%']:.4f}, P@10%={results['RF_Precision@10%']:.4f}")
    print(f"        LGB: ROC-AUC={results['LGB_Test_ROC_AUC']:.4f}, PR-AUC={results['LGB_Test_PR_AUC']:.4f}, "
          f"R@10%={results['LGB_Recall@10%']:.4f}, P@10%={results['LGB_Precision@10%']:.4f}")
    print(f"      Baseline ({results['Baseline_Model']}): "
          f"ROC-AUC={results['Baseline_Test_ROC_AUC']:.4f}, PR-AUC={results['Baseline_Test_PR_AUC']:.4f}, "
          f"R@10%={results['Baseline_Recall@10%']:.4f}, P@10%={results['Baseline_Precision@10%']:.4f}")
    print(f"      Selected Best on Test: ROC-AUC={results['Best_Test_ROC_AUC']:.4f}, PR-AUC={results['Best_Test_PR_AUC']:.4f}")
    print(f"      Test Recall@10%: {results['Best_Recall@10%']:.4f} ({results['Best_TP@10%']}/{results['Total_Crashes']})")
    print(f"      Test Precision@10%: {results['Best_Precision@10%']:.4f} ({results['Best_TP@10%']}/{results['Best_Top10%_Count']})")


# ============================================================
# Summary and Save Results
# ============================================================

print("\n" + "="*70)
print("[6] FINAL SUMMARY")
print("="*70)

results_df = pd.DataFrame(all_results)

# Build one compact summary table (Baseline + best model per group)
def _first_valid(series, default=np.nan):
    valid = series.dropna()
    return valid.iloc[0] if len(valid) > 0 else default

baseline_summary = {
    'Group': 'Baseline',
    'Best_Model': 'RandomRank',
    'Test_ROC_AUC': _first_valid(results_df['Baseline_Test_ROC_AUC']),
    'Test_PR_AUC': _first_valid(results_df['Baseline_Test_PR_AUC']),
    'Top10%_Count': _first_valid(results_df['Baseline_Top10%_Count'], default=0),
    'TP@10%': _first_valid(results_df['Baseline_TP@10%'], default=0),
    'Recall@10%': _first_valid(results_df['Baseline_Recall@10%'], default=0),
    'Precision@10%': _first_valid(results_df['Baseline_Precision@10%'], default=0),
}

group_summary_df = results_df[[
    'Group', 'Best_Model', 'Best_Test_ROC_AUC', 'Best_Test_PR_AUC',
    'Best_Top10%_Count', 'Best_TP@10%', 'Best_Recall@10%', 'Best_Precision@10%'
]].rename(columns={
    'Best_Test_ROC_AUC': 'Test_ROC_AUC',
    'Best_Test_PR_AUC': 'Test_PR_AUC',
    'Best_Top10%_Count': 'Top10%_Count',
    'Best_TP@10%': 'TP@10%',
    'Best_Recall@10%': 'Recall@10%',
    'Best_Precision@10%': 'Precision@10%',
})

model_summary_df = pd.concat([pd.DataFrame([baseline_summary]), group_summary_df], ignore_index=True)

# Tables aligned with the screenshot layout
table_1_df = results_df[[
    'Group', 'Num_Features', 'Train_Samples', 'Val_Samples',
    'Test_Samples', 'LR_Top10%_Count', 'Test_Positive_Rate', 'Total_Crashes'
]].rename(columns={
    'LR_Top10%_Count': 'Top10%_Count',
    'Total_Crashes': 'Test_Total_Crashes',
})

table_2_df = results_df[[
    'Group',
    'LR_Test_PR_AUC', 'RF_Test_PR_AUC', 'LGB_Test_PR_AUC',
    'LR_Val_PR_AUC', 'RF_Val_PR_AUC', 'LGB_Val_PR_AUC',
    'LR_Test_ROC_AUC'
]]

table_3_df = results_df[['Group', 'RF_Test_ROC_AUC', 'LGB_Test_ROC_AUC']]

table_4_df = results_df[[
    'Group',
    'LR_TP@10%', 'LR_Recall@10%', 'LR_Precision@10%',
    'RF_TP@10%', 'RF_Recall@10%', 'RF_Precision@10%',
    'LGB_TP@10%'
]]

table_5_df = results_df[['Group', 'LGB_Recall@10%', 'LGB_Precision@10%']]

# Display compact final summary table
print("\n" + "-"*110)
print("FINAL MODEL RESULT SUMMARY (BASELINE + BEST MODEL PER GROUP)")
print("-"*110)
print(f"{'Group':<25} {'Best_Model':<15} {'Test_ROC_AUC':>12} {'Test_PR_AUC':>12} {'Top10%':>8} {'TP@10%':>8} {'R@10%':>8} {'P@10%':>8}")
print("-"*110)
for _, row in model_summary_df.iterrows():
    print(f"{row['Group']:<25} {row['Best_Model']:<15} "
          f"{row['Test_ROC_AUC']:>12.4f} {row['Test_PR_AUC']:>12.4f} "
          f"{row['Top10%_Count']:>8.0f} {row['TP@10%']:>8.1f} "
          f"{row['Recall@10%']:>8.4f} {row['Precision@10%']:>8.4f}")
print("-"*110)

# Display summary table
print("\n" + "-"*130)
print("FINAL TEST PR-AUC COMPARISON (ALL MODELS + BASELINE)")
print("-"*130)
print(f"{'Group':<25} {'#Feat':>6} {'LR_PR':>8} {'RF_PR':>8} {'LGB_PR':>8} {'Base_PR':>8} {'Best_Model':>14} {'Best_PR':>8}")
print("-"*130)
for _, row in results_df.iterrows():
    print(f"{row['Group']:<25} {row['Num_Features']:>6} "
          f"{row['LR_Test_PR_AUC']:>8.4f} {row['RF_Test_PR_AUC']:>8.4f} {row['LGB_Test_PR_AUC']:>8.4f} "
          f"{row['Baseline_Test_PR_AUC']:>8.4f} {row['Best_Model']:>14} {row['Best_Test_PR_AUC']:>8.4f}")
print("-"*130)

print("\n" + "-"*130)
print("FINAL TEST ROC-AUC COMPARISON (ALL MODELS + BASELINE)")
print("-"*130)
print(f"{'Group':<25} {'#Feat':>6} {'LR_ROC':>8} {'RF_ROC':>8} {'LGB_ROC':>8} {'Base_ROC':>8} {'Best_ROC':>8}")
print("-"*130)
for _, row in results_df.iterrows():
    print(f"{row['Group']:<25} {row['Num_Features']:>6} "
          f"{row['LR_Test_ROC_AUC']:>8.4f} {row['RF_Test_ROC_AUC']:>8.4f} {row['LGB_Test_ROC_AUC']:>8.4f} "
          f"{row['Baseline_Test_ROC_AUC']:>8.4f} {row['Best_Test_ROC_AUC']:>8.4f}")
print("-"*130)

# Save to Excel
print("\n[7] Saving results to Excel...")

with pd.ExcelWriter('model_comparison_batch_results.xlsx', engine='openpyxl') as writer:
    # Summary sheet
    results_df.to_excel(writer, sheet_name='Summary', index=False)
    model_summary_df.to_excel(writer, sheet_name='Model_Result_Summary', index=False)

    # Screenshot-style stacked tables (all 6 tables)
    image_sheet = 'Image_Tables'
    start_row = 0
    for df in [table_1_df, table_2_df, table_3_df, table_4_df, table_5_df, model_summary_df]:
        df.to_excel(writer, sheet_name=image_sheet, index=False, startrow=start_row)
        start_row += len(df) + 3

    # Feature details for each group
    feature_df = pd.DataFrame([
        {'Group': group, 'Feature_Index': i+1, 'Feature': feat}
        for group, feats in all_feature_details.items()
        for i, feat in enumerate(feats)
    ])
    feature_df.to_excel(writer, sheet_name='Feature_Details', index=False)

    # Detailed metrics
    metrics_cols = ['Group', 'Num_Features', 'Features',
                    'Train_Samples', 'Val_Samples', 'TrainVal_Samples', 'Test_Samples', 'Test_Positive_Rate',
                    'LR_Val_PR_AUC', 'RF_Val_PR_AUC', 'LGB_Val_PR_AUC',
                    'LR_Test_ROC_AUC', 'LR_Test_PR_AUC', 'LR_Recall@10%', 'LR_Precision@10%', 'LR_TP@10%', 'LR_Top10%_Count',
                    'RF_Test_ROC_AUC', 'RF_Test_PR_AUC', 'RF_Recall@10%', 'RF_Precision@10%', 'RF_TP@10%', 'RF_Top10%_Count',
                    'LGB_Test_ROC_AUC', 'LGB_Test_PR_AUC', 'LGB_Recall@10%', 'LGB_Precision@10%', 'LGB_TP@10%', 'LGB_Top10%_Count',
                    'Baseline_Model', 'Baseline_Test_ROC_AUC', 'Baseline_Test_PR_AUC',
                    'Baseline_Recall@10%', 'Baseline_Precision@10%', 'Baseline_TP@10%', 'Baseline_Top10%_Count',
                    'Best_Model', 'Best_Params', 'Best_Val_PR_AUC',
                    'Best_Test_ROC_AUC', 'Best_Test_PR_AUC',
                    'Best_Recall@10%', 'Best_Precision@10%', 'Best_TP@10%', 'Best_Top10%_Count',
                    'Total_Crashes']
    available_cols = [c for c in metrics_cols if c in results_df.columns]
    results_df[available_cols].to_excel(writer, sheet_name='Detailed_Metrics', index=False)

print(f"    Results saved to 'model_comparison_batch_results.xlsx'")

print("\n" + "="*70)
print("BATCH COMPARISON COMPLETED!")
print("="*70)

