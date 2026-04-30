"""
Feature Selection and Model Preparation

This script performs:
1. Feature correlation analysis - identify highly correlated feature pairs (|corr| > 0.8)
2. Remove highly correlated features - keep the one with longer data length
3. Single feature predictive power evaluation - PR-AUC and ROC-AUC for each feature
   against SP500 and NASDAQ100 crash labels

IMPORTANT: All calculations are done on TRAINING data only (first 50%)
           to prevent data leakage from validation/test sets.

Output:
- feature_correlation_pairs.csv: Highly correlated feature pairs
- features_to_remove.csv: Features removed due to high correlation
- feature_importance_sp500.csv: Feature ranking for SP500 crash prediction
- feature_importance_nasdaq100.csv: Feature ranking for NASDAQ100 crash prediction
- modelling_results.xlsx: Comprehensive results in Excel format
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Load Data
# ============================================================

print("Loading data...")

# Load features
features = pd.read_csv('features.csv', index_col=0, parse_dates=True)
print(f"Features shape: {features.shape}")

# Load target labels
target_labels = pd.read_csv('target_label.csv', index_col=0, parse_dates=True)
print(f"Target labels shape: {target_labels.shape}")

# ============================================================
# Split Data: Use regime_label.csv split points
# ============================================================

print("\n" + "="*60)
print("Data Split: Using regime_label.csv Split Points")
print("="*60)

# Load regime_label.csv to get split boundaries
regime_labels = pd.read_csv('regime_label.csv', index_col=0, parse_dates=True)

# Get split boundaries from regime_label.csv
train_end_date = regime_labels[regime_labels['Data_Split'] == 'Train'].index[-1]
val_start_date = regime_labels[regime_labels['Data_Split'] == 'Validation'].index[0]
val_end_date = regime_labels[regime_labels['Data_Split'] == 'Validation'].index[-1]
test_start_date = regime_labels[regime_labels['Data_Split'] == 'Test'].index[0]

print(f"Split boundaries from regime_label.csv:")
print(f"  Training ends: {train_end_date.date()}")
print(f"  Validation: {val_start_date.date()} to {val_end_date.date()}")
print(f"  Test starts: {test_start_date.date()}")

# ============================================================
# Add Regime Probability Features to features DataFrame
# ============================================================

print("\nAdding regime probability features from regime_label.csv...")
# Use Delta_P_Stress_5d (5-day rolling) for feature selection instead of raw Delta_P_Stress
regime_features = ['P_Calm', 'P_Transitional', 'P_Stress', 'Delta_P_Stress_5d']
for feat in regime_features:
    # Check if feature already exists in features (from features.py)
    if feat in features.columns:
        valid_count = features[feat].notna().sum()
        print(f"  {feat} already in features: {valid_count} valid values")
    elif feat in regime_labels.columns:
        features[feat] = regime_labels[feat]
        valid_count = features[feat].notna().sum()
        print(f"  Added {feat}: {valid_count} valid values")
    elif feat == 'Delta_P_Stress_5d' and 'Delta_P_Stress' in regime_labels.columns:
        # Calculate 5d rolling if not already in features
        Delta_P_Stress = regime_labels['Delta_P_Stress']
        features['Delta_P_Stress_5d'] = Delta_P_Stress.rolling(window=5, min_periods=4).mean()
        valid_count = features['Delta_P_Stress_5d'].notna().sum()
        print(f"  Created Delta_P_Stress_5d (5d rolling): {valid_count} valid values")
    else:
        print(f"  Warning: {feat} not found in regime_label.csv")

# Align features and target labels by common index
common_idx = features.index.intersection(target_labels.index)
features_aligned = features.loc[common_idx].sort_index()
target_aligned = target_labels.loc[common_idx].sort_index()

# Split data based on regime_label.csv dates
# Training: all data up to train_end_date (including data before regime_label start)
train_mask = features_aligned.index <= train_end_date
val_mask = (features_aligned.index >= val_start_date) & (features_aligned.index <= val_end_date)
test_mask = features_aligned.index >= test_start_date

train_features = features_aligned[train_mask]
train_labels = target_aligned[train_mask]
val_features = features_aligned[val_mask]
val_labels = target_aligned[val_mask]
test_features = features_aligned[test_mask]
test_labels = target_aligned[test_mask]

n_total = len(features_aligned)
n_train = len(train_features)
n_val = len(val_features)
n_test = len(test_features)

print(f"\nTotal aligned samples: {n_total}")
print(f"Training set: {n_train} samples ({100*n_train/n_total:.1f}%)")
print(f"  Date range: {train_features.index[0].date()} to {train_features.index[-1].date()}")
print(f"Validation set: {n_val} samples ({100*n_val/n_total:.1f}%) - NOT USED in feature selection")
print(f"  Date range: {val_features.index[0].date()} to {val_features.index[-1].date()}")
print(f"Test set: {n_test} samples ({100*n_test/n_total:.1f}%) - NOT USED in feature selection")
print(f"  Date range: {test_features.index[0].date()} to {test_features.index[-1].date()}")

# ============================================================
# 1. Feature Correlation Analysis (Training Data Only)
# ============================================================

print("\n" + "="*60)
print("1. Feature Correlation Analysis (Training Data Only)")
print("="*60)

# Calculate correlation matrix (only numeric columns)
numeric_features = train_features.select_dtypes(include=[np.number])
print(f"\nNumber of numeric features: {numeric_features.shape[1]}")

# Drop columns with all NaN
numeric_features = numeric_features.dropna(axis=1, how='all')
print(f"Features after dropping all-NaN columns: {numeric_features.shape[1]}")

# Calculate valid data length for each feature
feature_lengths = {}
for col in numeric_features.columns:
    feature_lengths[col] = numeric_features[col].notna().sum()

print(f"Calculated data lengths for {len(feature_lengths)} features")

# Calculate correlation matrix
print("\nCalculating correlation matrix...")
corr_matrix = numeric_features.corr()

# Find highly correlated pairs (|correlation| > 0.9)
print("Calculating all feature pair correlations...")

all_corr_pairs = []
high_corr_pairs = []
feature_names = corr_matrix.columns.tolist()
corr_array = corr_matrix.values

# Get upper triangle indices (excluding diagonal)
upper_tri_indices = np.triu_indices_from(corr_array, k=1)
feat1_names = np.array([feature_names[i] for i in upper_tri_rows])
feat2_names = np.array([feature_names[j] for j in upper_tri_cols])

feat1_lengths = np.array([feature_lengths[name] for name in feat1_names])
for i in range(len(feature_names)):
    for j in range(i + 1, len(feature_names)):
        corr_value = corr_matrix.iloc[i, j]
            # Determine which feature to keep (longer data) and which to remove
            if len1 >= len2:
                keep_feature = feat1
                remove_feature = feat2
            pair_info = {
                'Feature_1': feat1,
                'Feature_2': feat2,
                'Correlation': corr_value,
            all_corr_pairs.append(pair_info)
keep_feat1 = feat1_lengths >= feat2_lengths  # True if keep feat1, False if keep feat2
keep_features = np.where(keep_feat1, feat1_names, feat2_names)
remove_features = np.where(keep_feat1, feat2_names, feat1_names)

# Filter out NaN values
            # Also track high correlation pairs for feature removal
            if abs(corr_value) > 0.8:
                high_corr_pairs.append(pair_info)

# Create DataFrame for all pairs
all_corr_pairs = pd.DataFrame({
    'Feature_1': feat1_names,
    'Feature_2': feat2_names,
    'Correlation': corr_values,
# Create DataFrame for all pairs and sort by absolute correlation (descending)
all_corr_df = pd.DataFrame(all_corr_pairs)
if len(all_corr_df) > 0:
    all_corr_df = all_corr_df.sort_values('Abs_Correlation', ascending=False)
    all_corr_df = all_corr_df.reset_index(drop=True)
# Create DataFrame for high correlation pairs
high_corr_df = pd.DataFrame(high_corr_pairs)
if len(high_corr_df) > 0:
    high_corr_df = high_corr_df.sort_values('Abs_Correlation', ascending=False)
    high_corr_df = high_corr_df.reset_index(drop=True)
    print("-" * 110)
    print(f"{'Feature_1':<30} {'Feature_2':<30} {'Corr':>8} {'Len1':>6} {'Len2':>6} {'Remove':<25}")
    print("-" * 110)
    for idx, row in all_corr_df.head(20).iterrows():
        remove_note = row['Remove'] if abs(row['Correlation']) > 0.8 else ''
        print(f"{row['Feature_1']:<30} {row['Feature_2']:<30} {row['Correlation']:>8.4f} {row['Length_1']:>6} {row['Length_2']:>6} {remove_note:<25}")

# Save ALL correlation pairs to CSV (sorted by absolute correlation)
all_corr_df.to_csv('feature_correlation_pairs.csv', index=False)
print(f"\nAll {len(all_corr_df)} correlation pairs saved to 'feature_correlation_pairs.csv' (sorted by |corr|)")

# ============================================================
# 2. Remove Highly Correlated Features
# ============================================================

print("\n" + "="*60)
print("2. Removing Highly Correlated Features")
print("="*60)

# Collect features to remove (keep the one with longer data)
features_to_remove = set()

if len(high_corr_df) > 0:
    for _, row in high_corr_df.iterrows():
        features_to_remove.add(row['Remove'])

print(f"\nFeatures to remove: {len(features_to_remove)}")

# Save removed features list
removed_features_df = pd.DataFrame({
    'Feature': list(features_to_remove),
    'Reason': 'High correlation (>0.8) with longer feature'
})
removed_features_df = removed_features_df.sort_values('Feature')
removed_features_df.to_csv('features_to_remove.csv', index=False)
print(f"Removed features saved to 'features_to_remove.csv'")

# Create filtered feature set
filtered_features = numeric_features.drop(columns=list(features_to_remove), errors='ignore')
print(f"\nOriginal features: {numeric_features.shape[1]}")
print(f"Features after removing high correlation: {filtered_features.shape[1]}")

# ============================================================
# 3. Single Feature Predictive Power (PR-AUC, ROC-AUC)
#    Using Training Data Only and Filtered Features
# ============================================================

print("\n" + "="*60)
print("3. Single Feature Predictive Power Evaluation")
print("   (Training Data Only, After Removing Correlated Features)")
print("="*60)

def calculate_feature_auc(features_df, labels_series, label_name):
    """
    Calculate PR-AUC and ROC-AUC for each feature against a binary label.

    Parameters:
    -----------
    features_df : DataFrame
        Feature matrix (training data only)
    labels_series : Series
        Binary labels (0/1) (training data only)
    label_name : str
        Name of the label for display

    Returns:
    --------
    results_df : DataFrame
        Feature importance rankings with PR-AUC and ROC-AUC
    """
    print(f"\nCalculating AUC metrics for {label_name}...")

    # Align features and labels
    common_idx = features_df.index.intersection(labels_series.dropna().index)
    X = features_df.loc[common_idx]
    y = labels_series.loc[common_idx]

    # Remove rows where label is NaN
    valid_mask = y.notna()
    X = X[valid_mask]
    y = y[valid_mask].astype(int)

    print(f"  Total samples: {len(y)}")
    print(f"  Positive samples (Label=1): {y.sum()} ({100*y.mean():.2f}%)")

    results = []

    for col in X.columns:
        feature_values = X[col]

        # Skip if too many NaN
        valid_feature_mask = feature_values.notna()
        if valid_feature_mask.sum() < 100:
            continue

        y_valid = y[valid_feature_mask]
        x_valid = feature_values[valid_feature_mask]

        # Skip if only one class present
        if y_valid.nunique() < 2:
            continue

        try:
            # Calculate ROC-AUC
            roc_auc = roc_auc_score(y_valid, x_valid)

            # Calculate PR-AUC (Average Precision)
            pr_auc = average_precision_score(y_valid, x_valid)

            # Also calculate for negated feature (in case negative correlation)
            roc_auc_neg = roc_auc_score(y_valid, -x_valid)
            pr_auc_neg = average_precision_score(y_valid, -x_valid)

            # Use the better direction
            if roc_auc_neg > roc_auc:
                roc_auc = roc_auc_neg
                pr_auc = pr_auc_neg
                direction = 'Negative'
            else:
                direction = 'Positive'

            results.append({
                'Feature': col,
                'ROC_AUC': roc_auc,
                'PR_AUC': pr_auc,
                'Direction': direction,
                'Valid_Samples': valid_feature_mask.sum(),
                'Positive_Rate': y_valid.mean()
            })

        except Exception as e:
            # Skip features that cause errors
            continue

    # Create DataFrame and sort by PR-AUC (more relevant for imbalanced data)
    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        results_df = results_df.sort_values('PR_AUC', ascending=False)
        results_df = results_df.reset_index(drop=True)
        results_df['Rank_PR_AUC'] = range(1, len(results_df) + 1)

        # Also add ROC-AUC rank
        results_df['Rank_ROC_AUC'] = results_df['ROC_AUC'].rank(ascending=False).astype(int)

    return results_df

# ----- SP500 Label -----
print("\n----- SP500 Crash Prediction -----")
sp500_results = calculate_feature_auc(
    filtered_features,
    train_labels['SP500_Label'],
    'SP500_Label'
)

if len(sp500_results) > 0:
    print(f"\nTop 20 features for SP500 crash prediction (by PR-AUC):")
    print("-" * 100)
    print(f"{'Rank':<6} {'Feature':<40} {'PR_AUC':>10} {'ROC_AUC':>10} {'Direction':>10}")
    print("-" * 100)
    for idx, row in sp500_results.head(20).iterrows():
        print(f"{row['Rank_PR_AUC']:<6} {row['Feature']:<40} {row['PR_AUC']:>10.4f} {row['ROC_AUC']:>10.4f} {row['Direction']:>10}")

    # Save to CSV
    sp500_results.to_csv('feature_importance_sp500.csv', index=False)
    print(f"\nSP500 feature importance saved to 'feature_importance_sp500.csv'")

# ----- NASDAQ100 Label -----
print("\n----- NASDAQ100 Crash Prediction -----")
nasdaq_results = calculate_feature_auc(
    filtered_features,
    train_labels['NASDAQ100_Label'],
    'NASDAQ100_Label'
)

if len(nasdaq_results) > 0:
    print(f"\nTop 20 features for NASDAQ100 crash prediction (by PR-AUC):")
    print("-" * 100)
    print(f"{'Rank':<6} {'Feature':<40} {'PR_AUC':>10} {'ROC_AUC':>10} {'Direction':>10}")
    print("-" * 100)
    for idx, row in nasdaq_results.head(20).iterrows():
        print(f"{row['Rank_PR_AUC']:<6} {row['Feature']:<40} {row['PR_AUC']:>10.4f} {row['ROC_AUC']:>10.4f} {row['Direction']:>10}")

    # Save to CSV
    nasdaq_results.to_csv('feature_importance_nasdaq100.csv', index=False)
    print(f"\nNASDAQ100 feature importance saved to 'feature_importance_nasdaq100.csv'")

# ============================================================
# 4. Save Comprehensive Results to Excel
# ============================================================

print("\n" + "="*60)
print("4. Saving Comprehensive Results")
print("="*60)

with pd.ExcelWriter('feature_selection_results.xlsx', engine='openpyxl') as writer:
    # Sheet 1: All correlation pairs (sorted by absolute correlation)
    if len(all_corr_df) > 0:
        all_corr_df.to_excel(writer, sheet_name='All_Correlation_Pairs', index=False)

    # Sheet 2: Highly correlated pairs only (|corr| > 0.8)
    if len(high_corr_df) > 0:
        high_corr_df.to_excel(writer, sheet_name='High_Correlation_Pairs', index=False)

    # Sheet 2: Removed features
    if len(removed_features_df) > 0:
        removed_features_df.to_excel(writer, sheet_name='Removed_Features', index=False)

    # Sheet 3: SP500 feature importance
    if len(sp500_results) > 0:
        sp500_results.to_excel(writer, sheet_name='SP500_Feature_Importance', index=False)

    # Sheet 4: NASDAQ100 feature importance
    if len(nasdaq_results) > 0:
        nasdaq_results.to_excel(writer, sheet_name='NASDAQ100_Feature_Importance', index=False)

    # Sheet 5: Summary statistics
    summary_data = {
        'Metric': [
            'Total Features (Original)',
            'Features After Removing High Correlation',
            'Features Removed',
            'Highly Correlated Pairs (|corr|>0.8)',
            'Training Samples',
            'Training Date Range Start',
            'Training Date Range End',
            'SP500 - Best PR-AUC Feature',
            'SP500 - Best PR-AUC Score',
            'SP500 - Best ROC-AUC Feature',
            'SP500 - Best ROC-AUC Score',
            'NASDAQ100 - Best PR-AUC Feature',
            'NASDAQ100 - Best PR-AUC Score',
            'NASDAQ100 - Best ROC-AUC Feature',
            'NASDAQ100 - Best ROC-AUC Score',
        ],
        'Value': [
            numeric_features.shape[1],
            filtered_features.shape[1],
            len(features_to_remove),
            len(high_corr_df),
            n_train,
            str(train_features.index[0].date()),
            str(train_features.index[-1].date()),
            sp500_results.iloc[0]['Feature'] if len(sp500_results) > 0 else 'N/A',
            f"{sp500_results.iloc[0]['PR_AUC']:.4f}" if len(sp500_results) > 0 else 'N/A',
            sp500_results.sort_values('ROC_AUC', ascending=False).iloc[0]['Feature'] if len(sp500_results) > 0 else 'N/A',
            f"{sp500_results['ROC_AUC'].max():.4f}" if len(sp500_results) > 0 else 'N/A',
            nasdaq_results.iloc[0]['Feature'] if len(nasdaq_results) > 0 else 'N/A',
            f"{nasdaq_results.iloc[0]['PR_AUC']:.4f}" if len(nasdaq_results) > 0 else 'N/A',
            nasdaq_results.sort_values('ROC_AUC', ascending=False).iloc[0]['Feature'] if len(nasdaq_results) > 0 else 'N/A',
            f"{nasdaq_results['ROC_AUC'].max():.4f}" if len(nasdaq_results) > 0 else 'N/A',
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_excel(writer, sheet_name='Summary', index=False)

print("Results saved to 'feature_selection_results.xlsx'")

# ============================================================
# 5. Summary
# ============================================================

print("\n" + "="*60)
print("SUMMARY")
print("="*60)

print(f"\n0. Data Split (aligned with regime_label.csv):")
print(f"   - Training samples used: {n_train}")
print(f"   - Training date range: {train_features.index[0].date()} to {train_features.index[-1].date()}")
print(f"   - Validation samples: {n_val} (NOT USED)")
print(f"   - Test samples: {n_test} (NOT USED)")

print(f"\n1. Correlation Analysis:")
print(f"   - Total features analyzed: {numeric_features.shape[1]}")
print(f"   - Total feature pairs: {len(all_corr_df)}")
print(f"   - Highly correlated pairs (|corr| > 0.8): {len(high_corr_df)}")
print(f"   - Features removed (shorter data length): {len(features_to_remove)}")
print(f"   - Features remaining: {filtered_features.shape[1]}")

if len(sp500_results) > 0:
    print(f"\n2. SP500 Crash Prediction (Training Only):")
    print(f"   - Features evaluated: {len(sp500_results)}")
    print(f"   - Best PR-AUC: {sp500_results.iloc[0]['Feature']} ({sp500_results.iloc[0]['PR_AUC']:.4f})")
    best_roc_sp500 = sp500_results.sort_values('ROC_AUC', ascending=False).iloc[0]
    print(f"   - Best ROC-AUC: {best_roc_sp500['Feature']} ({best_roc_sp500['ROC_AUC']:.4f})")

if len(nasdaq_results) > 0:
    print(f"\n3. NASDAQ100 Crash Prediction (Training Only):")
    print(f"   - Features evaluated: {len(nasdaq_results)}")
    print(f"   - Best PR-AUC: {nasdaq_results.iloc[0]['Feature']} ({nasdaq_results.iloc[0]['PR_AUC']:.4f})")
    best_roc_nasdaq = nasdaq_results.sort_values('ROC_AUC', ascending=False).iloc[0]
    print(f"   - Best ROC-AUC: {best_roc_nasdaq['Feature']} ({best_roc_nasdaq['ROC_AUC']:.4f})")

print("\n" + "="*60)
print("Output Files:")
print("="*60)
print("  - feature_correlation_pairs.csv")
print("  - features_to_remove.csv")
print("  - feature_importance_sp500.csv")
print("  - feature_importance_nasdaq100.csv")
print("  - modelling_results.xlsx")

print("\nDone!")

