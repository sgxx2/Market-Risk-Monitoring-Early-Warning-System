"""
Feature Engineering for Data Mining Project
- Calculate log returns for price/index data
- Calculate differences for interest rates and VIX
- Construct credit factor: f_t = r_t^HYG - r_t^IEF
- Range features: log(High/Low)
- Interest rate slope: 10Y - 3M
- SPX volatility and max drawdown (20 day)
- Rolling features: 5d, 10d, 252d quantile(0.90)
- Interaction features for credit spread
"""

import pandas as pd
import numpy as np

# ============================================================
# Load Data
# ============================================================

# Read the combine_data.csv with multi-index columns
df = pd.read_csv('combine_data.csv', header=[0, 1], index_col=0, parse_dates=True)

print("Data loaded successfully.")
print(f"Original shape: {df.shape}")
print(f"Date range: {df.index[0]} to {df.index[-1]}")

# ============================================================
# Data Alignment: Use SPX as baseline
# ============================================================

print("\n" + "="*60)
print("Data Alignment Process")
print("="*60)

# Step 1: Remove all rows where SP500 Close is NaN
sp500_close = df['Close']['SP500']
valid_sp500_mask = sp500_close.notna()
df_aligned = df[valid_sp500_mask].copy()

print(f"Step 1: Removed rows where SP500 is missing")
print(f"  - Original rows: {len(df)}")
print(f"  - After removing SP500 NaN rows: {len(df_aligned)}")
print(f"  - Rows removed: {len(df) - len(df_aligned)}")

# Step 2: For other symbols, fill missing values with forward fill
def fill_missing_with_ffill(series):
    """
    Fill missing values using forward fill method.
    Only fill values that are after the first valid data point.
    """
    if series.isna().sum() == 0:
        return series

    # Use forward fill
    filled = series.ffill()
    return filled

print(f"\nStep 2: Filling missing values for other symbols using forward fill...")

# Get all columns at level 0 (Close, High, Low, Open, Volume)
for price_type in df_aligned.columns.get_level_values(0).unique():
    for symbol in df_aligned[price_type].columns:
        if symbol == 'SP500':
            continue  # SP500 is our baseline, skip

        col_data = df_aligned[(price_type, symbol)]
        missing_before = col_data.isna().sum()

        if missing_before > 0:
            # Fill missing values with forward fill
            df_aligned[(price_type, symbol)] = fill_missing_with_ffill(col_data)
            missing_after = df_aligned[(price_type, symbol)].isna().sum()

            if missing_before != missing_after:
                print(f"  - {price_type}/{symbol}: {missing_before} -> {missing_after} missing values")

# Summary of remaining missing values
print(f"\nAlignment complete. Final shape: {df_aligned.shape}")

# Check remaining missing values in Close prices
close_missing = df_aligned['Close'].isna().sum()
print(f"\nRemaining missing values in Close prices:")
for symbol in close_missing.index:
    if close_missing[symbol] > 0:
        print(f"  - {symbol}: {close_missing[symbol]}")

# Extract Close, High, Low prices for all symbols (after alignment)
close_data = df_aligned['Close']
high_data = df_aligned['High']
low_data = df_aligned['Low']

print(f"\nAligned data:")
print(f"Shape: {close_data.shape}")
print(f"Symbols: {list(close_data.columns)}")
print(f"Date range: {close_data.index[0]} to {close_data.index[-1]}")

# ============================================================
# Define Symbol Categories (from visualization.py)
# ============================================================

# Symbols that should use difference instead of log return
# Interest rates and VIX are levels, not prices
difference_symbols = ['RATE_10Y', 'RATE_3M', 'VIX']

# All other symbols use log return
log_return_symbols = [col for col in close_data.columns if col not in difference_symbols]

print(f"\nLog Return symbols: {log_return_symbols}")
print(f"Difference symbols: {difference_symbols}")

# ============================================================
# Calculate Log Returns and Differences
# ============================================================

# Calculate daily log returns: ln(P_t / P_{t-1})
log_returns = np.log(close_data / close_data.shift(1))

# Calculate differences: P_t - P_{t-1}
differences = close_data.diff()

# Combine into features DataFrame
# Use log returns for most symbols, differences for RATE_10Y, RATE_3M, VIX
features = pd.DataFrame(index=close_data.index)

for symbol in close_data.columns:
    if symbol in difference_symbols:
        features[f'{symbol}_diff'] = differences[symbol]
    else:
        features[f'{symbol}_logret'] = log_returns[symbol]

print(f"\nFeatures created: {list(features.columns)}")

# ============================================================
# 1. Range Features: log(High/Low) for every symbol
# ============================================================

for symbol in close_data.columns:
    if symbol in high_data.columns and symbol in low_data.columns:
        high = high_data[symbol]
        low = low_data[symbol]

        # Data quality check: High should be >= Low
        # Fix invalid cases where High < Low by swapping or using Close
        invalid_mask = high < low
        if invalid_mask.sum() > 0:
            print(f"  Warning: {symbol} has {invalid_mask.sum()} cases where High < Low, setting range to 0")

        # Calculate range ratio, handle invalid cases
        range_ratio = high / low
        range_ratio = range_ratio.where(~invalid_mask, 1.0)  # Set invalid to 1 (log=0)
        range_ratio = range_ratio.replace([np.inf, -np.inf], np.nan)
        range_ratio = range_ratio.clip(lower=1.0)  # Ensure ratio >= 1

        # Additional check: cap extreme range values (e.g., due to data errors)
        # A daily range > 50% (log > 0.4) is extremely rare, likely data error
        log_range = np.log(range_ratio)
        extreme_mask = log_range > 0.4
        if extreme_mask.sum() > 0:
            print(f"  Warning: {symbol} has {extreme_mask.sum()} extreme range values (>0.4), capping to median")
            median_range = log_range[log_range <= 0.4].median()
            log_range = log_range.where(~extreme_mask, median_range)

        features[f'{symbol}_range'] = log_range

print(f"Range features added: {[col for col in features.columns if '_range' in col]}")

# ============================================================
# 2. Interest Rate Slope: 10Y - 3M
# ============================================================

features['RATE_SLOPE_10Y_3M'] = close_data['RATE_10Y'] - close_data['RATE_3M']

print("Interest rate slope (10Y - 3M) added.")

# ============================================================
# 3. SPX Return Volatility (20 day) and Maximum Drawdown (20 day)
# ============================================================

# SPX 20-day rolling volatility (annualized)
# Note: Use dropna() to handle holiday gaps, then reindex back
# min_periods = 80% of 20 = 16
spx_returns = log_returns['SP500']
spx_returns_clean = spx_returns.dropna()
spx_vol_clean = spx_returns_clean.rolling(window=20, min_periods=16).std() * np.sqrt(252)
features['SPX_VOL_20D'] = spx_vol_clean.reindex(features.index)

# SPX 20-day maximum drawdown
def rolling_max_drawdown(prices, window=20):
    """Calculate rolling maximum drawdown over a given window"""
    # Use dropna to handle gaps, then reindex
    # min_periods = 80% of window
    min_p = int(window * 0.8)
    prices_clean = prices.dropna()
    rolling_max = prices_clean.rolling(window=window, min_periods=min_p).max()
    drawdown = (prices_clean - rolling_max) / rolling_max
    max_drawdown = drawdown.rolling(window=window, min_periods=min_p).min()
    return max_drawdown

spx_maxdd_clean = rolling_max_drawdown(close_data['SP500'], window=20)
features['SPX_MAXDD_20D'] = spx_maxdd_clean.reindex(features.index)

print("SPX volatility (20d) and max drawdown (20d) added.")

# ============================================================
# Construct Credit Factor
# f_t = r_t^HYG - r_t^IEF
# Credit spread factor: High Yield Bond return minus Treasury Bond return
# ============================================================

# Credit factor represents the excess return of high yield bonds over safe treasury bonds
# When credit spreads widen (credit risk increases), HYG underperforms IEF, f_t < 0
# When credit spreads narrow (credit risk decreases), HYG outperforms IEF, f_t > 0

features['CREDIT_FACTOR'] = log_returns['HYG'] - log_returns['IEF']

print("\nCredit Factor (f_t = r_t^HYG - r_t^IEF) constructed.")

# ============================================================
# 4. Rolling Features: 5d, 10d mean and 252d quantile(0.90)
# For VIX, use price level instead of return for rolling
# ============================================================

# Define base features for rolling (exclude already rolling features)
base_feature_cols = [col for col in features.columns if not any(x in col for x in ['_5d', '_10d', '_252d'])]

print(f"\nCreating rolling features for: {base_feature_cols}")

# Store rolling features separately to avoid modifying during iteration
rolling_features = {}

for col in base_feature_cols:
    # For VIX, use price level for rolling calculations
    if 'VIX' in col and 'diff' in col:
        # Use VIX price instead of diff for rolling
        base_series = close_data['VIX']
        col_prefix = 'VIX_price'
    else:
        base_series = features[col]
        col_prefix = col

    # 5-day rolling mean (min_periods = 80% of 5 = 4)
    rolling_features[f'{col_prefix}_5d'] = base_series.rolling(window=5, min_periods=4).mean()

    # 10-day rolling mean (min_periods = 80% of 10 = 8)
    rolling_features[f'{col_prefix}_10d'] = base_series.rolling(window=10, min_periods=8).mean()

    # 252-day rolling 90th percentile (min_periods = 80% of 252 = 202)
    rolling_features[f'{col_prefix}_q252'] = base_series.rolling(window=252, min_periods=202).quantile(0.90)

# Add rolling features to main dataframe
for col, series in rolling_features.items():
    features[col] = series

print(f"Rolling features added. Total features now: {len(features.columns)}")

# ============================================================
# 5. Interaction Features for Credit Spread
# ============================================================

# Get required base features - calculate directly without relying on rolling features
vix_price = close_data['VIX']
dVIX_1 = close_data['VIX'].diff(1)  # 1-day VIX change
spx_ret_1 = log_returns['SP500']  # 1-day SPX return
dxy_ret_20 = log_returns['DXY'].rolling(window=20, min_periods=16).sum()  # 20-day cumulative DXY return

# Calculate spread 252-day 90th percentile (min_periods = 80% of 252 = 202)
minp = 202
# Feature 1: spread_q252 × vix_gate
# VIX gate maps VIX level to 0~1 (below 80th percentile = 0, above 95th percentile = 1)
vix_thr_80 = vix_price.rolling(252, min_periods=minp).quantile(0.80)
vix_thr_95 = vix_price.rolling(252, min_periods=minp).quantile(0.95)

# Map VIX high level to 0~1 (below 80% = 0, above 95% = 1, linear in between)
vix_gate = ((vix_price - vix_thr_80) / (vix_thr_95 - vix_thr_80)).clip(0, 1)

spread = features['CREDIT_FACTOR']
s80 = spread.rolling(252, min_periods=minp).quantile(0.80)
s95 = spread.rolling(252, min_periods=minp).quantile(0.95)
spread_gate = ((spread - s80) / (s95 - s80)).clip(0, 1)

features['SPREAD_VIX_HIGH'] = spread_gate * vix_gate

# Feature 2: (-spx_ret_1) × max(dVIX_1, 0)
# Captures the interaction between negative SPX returns and VIX spikes
features['SPX_VIX_SPIKE'] = (-spx_ret_1) * np.maximum(dVIX_1, 0)

# Feature 3: dxy_ret_20 × spread_z
# Captures interaction between USD strength and credit spread (standardized)
spread_rolling_mean = features['CREDIT_FACTOR'].rolling(window=252, min_periods=minp).mean()
spread_rolling_std = features['CREDIT_FACTOR'].rolling(window=252, min_periods=minp).std()
spread_z = (features['CREDIT_FACTOR'] - spread_rolling_mean) / spread_rolling_std
features['DXY_SPREAD_INTERACT'] = dxy_ret_20 * spread_z

print("Interaction features for credit spread added:")
print("  - SPREAD_VIX_HIGH: spread_gate × vix_gate (both mapped to 0~1 based on 252d 80th-95th percentile)")
print("  - SPX_VIX_SPIKE: (-spx_ret_1) × max(dVIX_1, 0)")
print("  - DXY_SPREAD_INTERACT: dxy_ret_20 × spread_z (z-score with 252d rolling mean/std)")

# ============================================================
# 5b. Additional Features: VIX/Vol Ratio and Z-Score × P_Stress
# ============================================================

print("\nAdding VIX/Vol ratio and Z-Score × P_Stress interaction features...")

# Helper function for 252-day rolling z-score
def calc_zscore_252(series, min_periods=202):
    """Calculate 252-day rolling z-score: (x - rolling_mean) / rolling_std"""
    rolling_mean = series.rolling(window=252, min_periods=min_periods).mean()
    rolling_std = series.rolling(window=252, min_periods=min_periods).std()
    zscore = (series - rolling_mean) / rolling_std.clip(lower=1e-8)
    return zscore

# Load P_Stress from regime_label.csv
try:
    regime_labels = pd.read_csv('regime_label.csv', index_col=0, parse_dates=True)
    P_Stress = regime_labels['P_Stress'].reindex(features.index)
    print(f"  Loaded P_Stress from regime_label.csv: {P_Stress.notna().sum()} valid values")

    # Also load Delta_P_Stress and create rolling/zscore versions
    if 'Delta_P_Stress' in regime_labels.columns:
        Delta_P_Stress = regime_labels['Delta_P_Stress'].reindex(features.index)
        print(f"  Loaded Delta_P_Stress from regime_label.csv: {Delta_P_Stress.notna().sum()} valid values")

        # Step 1: Delta_P_Stress 5-day rolling mean (for feature selection)
        Delta_P_Stress_5d = Delta_P_Stress.rolling(window=5, min_periods=4).mean()
        features['Delta_P_Stress_5d'] = Delta_P_Stress_5d
        print("  - Delta_P_Stress_5d: 5-day rolling mean of Delta_P_Stress")

        # Step 2: 252-day z-score on the 5d rolling result (for model input)
        # z-score is calculated on Delta_P_Stress_5d, NOT on raw Delta_P_Stress
        features['Delta_P_Stress_5d_z252'] = calc_zscore_252(Delta_P_Stress_5d)
        print("  - Delta_P_Stress_5d_z252: 252-day z-score of Delta_P_Stress_5d (z-score on 5d rolling)")
except Exception as e:
    print(f"  Warning: Could not load P_Stress from regime_label.csv: {e}")
    P_Stress = pd.Series(np.nan, index=features.index)

# Feature 1: VIX Close Price / SPX_VOL_20D
# Ratio of implied volatility (VIX) to realized volatility (SPX_VOL_20D)
# High ratio indicates VIX is elevated relative to actual market volatility
features['VIX_VOL_RATIO'] = vix_price / features['SPX_VOL_20D']
# Handle inf values
features['VIX_VOL_RATIO'] = features['VIX_VOL_RATIO'].replace([np.inf, -np.inf], np.nan)
print("  - VIX_VOL_RATIO: VIX Close / SPX_VOL_20D")

# Feature 2: SPX_VOL_20D_zscore_252 × P_Stress
# Interaction between volatility z-score and stress probability
spx_vol_zscore = calc_zscore_252(features['SPX_VOL_20D'])
features['SPX_VOL_STRESS'] = spx_vol_zscore * P_Stress
print("  - SPX_VOL_STRESS: SPX_VOL_20D_zscore_252 × P_Stress")

# Feature 3: HYG_range_5d_zscore_252 × P_Stress
# Interaction between HYG range z-score and stress probability
if 'HYG_range_5d' in rolling_features:
    hyg_range_5d = rolling_features['HYG_range_5d']
elif 'HYG_range' in features.columns:
    hyg_range_5d = features['HYG_range'].rolling(window=5, min_periods=4).mean()
else:
    hyg_range_5d = pd.Series(np.nan, index=features.index)
hyg_range_5d_zscore = calc_zscore_252(hyg_range_5d)
features['HYG_RANGE_STRESS'] = hyg_range_5d_zscore * P_Stress
print("  - HYG_RANGE_STRESS: HYG_range_5d_zscore_252 × P_Stress")

# Feature 4: NASDAQ100_range_zscore_252 × P_Stress
# Interaction between NASDAQ100 range z-score and stress probability
if 'NASDAQ100_range' in features.columns:
    nasdaq_range_zscore = calc_zscore_252(features['NASDAQ100_range'])
    features['NASDAQ_RANGE_STRESS'] = nasdaq_range_zscore * P_Stress
    print("  - NASDAQ_RANGE_STRESS: NASDAQ100_range_zscore_252 × P_Stress")
else:
    print("  Warning: NASDAQ100_range not found, skipping NASDAQ_RANGE_STRESS")

# ============================================================
# 6. Recurrence Plot Features: RR and DET (60-day window)
# For SP500_logret, VIX_diff, CREDIT_FACTOR, DXY_logret
# ============================================================

print("\nCalculating Recurrence Plot features (RR, DET) with 60-day window...")

def compute_rr_det(window_data):
    """
    Compute Recurrence Rate (RR) and Determinism (DET) for a given window.

    Parameters:
    -----------
    window_data : array-like
        The time series window data

    Returns:
    --------
    rr : float
        Recurrence Rate
    det : float
        Determinism
    """
    data = window_data.dropna().values
    n = len(data)

    if n < 10:  # Need minimum points for meaningful calculation
        return np.nan, np.nan

    # Compute pairwise distance matrix
    distance_matrix = np.abs(data.reshape(-1, 1) - data.reshape(1, -1))

    # Use epsilon = 0.5 * std as threshold
    epsilon = 0.5 * np.std(data)

    if epsilon == 0:
        return np.nan, np.nan

    # Create recurrence matrix
    R = (distance_matrix <= epsilon).astype(int)

    # Calculate Recurrence Rate (RR)
    # RR = (sum of R excluding diagonal) / (N * (N-1))
    total_recurrence = np.sum(R) - n  # Exclude diagonal
    rr = total_recurrence / (n * (n - 1)) if n > 1 else 0

    # Calculate Determinism (DET)
    # DET = (sum of recurrence points in diagonal lines of length >= 2) / total_recurrence
    if total_recurrence == 0:
        det = 0
    else:
        diag_sum = 0
        # Check diagonal lines (both upper and lower triangular)
        for k in range(2, n):
            # Upper diagonal
            diag = np.diag(R, k)
            diag_str = ''.join(map(str, diag))
            for run in diag_str.split('0'):
                if len(run) >= 2:
                    diag_sum += len(run)
            # Lower diagonal (symmetric, so same count)
            diag = np.diag(R, -k)
            diag_str = ''.join(map(str, diag))
            for run in diag_str.split('0'):
                if len(run) >= 2:
                    diag_sum += len(run)
        det = diag_sum / total_recurrence if total_recurrence > 0 else 0

    return rr, det

# Features to compute RR and DET
rp_features = ['SP500_logret', 'VIX_diff', 'CREDIT_FACTOR', 'DXY_logret']
window_size = 60
min_periods_rp = int(window_size * 0.8)  # 80% = 48

for feature_name in rp_features:
    if feature_name not in features.columns:
        print(f"  Warning: {feature_name} not found, skipping RR/DET calculation")
        continue

    print(f"  Calculating RR and DET for {feature_name}...")

    series = features[feature_name]

    # Initialize result arrays
    rr_values = np.full(len(series), np.nan)
    det_values = np.full(len(series), np.nan)

    # Rolling window calculation
    for i in range(window_size - 1, len(series)):
        window = series.iloc[i - window_size + 1:i + 1]
        valid_count = window.notna().sum()

        if valid_count >= min_periods_rp:
            rr, det = compute_rr_det(window)
            rr_values[i] = rr
            det_values[i] = det

    # Add to features DataFrame
    feature_prefix = feature_name.replace('_logret', '').replace('_diff', '')
    features[f'{feature_prefix}_RR_60d'] = rr_values
    features[f'{feature_prefix}_DET_60d'] = det_values

print("Recurrence Plot features added:")
print("  - SP500_RR_60d, SP500_DET_60d")
print("  - VIX_RR_60d, VIX_DET_60d")
print("  - CREDIT_FACTOR_RR_60d, CREDIT_FACTOR_DET_60d")
print("  - DXY_RR_60d, DXY_DET_60d")

# ============================================================
# Summary Statistics
# ============================================================

print("\n" + "="*60)
print("Feature Summary Statistics")
print("="*60)
print(features.describe().T.to_string())

# ============================================================
# Save Features to CSV
# ============================================================

# Drop the first row (NaN due to differencing/log return calculation)
features_clean = features.dropna(how='all')

features_clean.to_csv('features.csv')
print(f"\nFeatures saved to 'features.csv'")
print(f"Shape: {features_clean.shape}")

# ============================================================
# Additional Feature Analysis
# ============================================================

print("\n" + "="*60)
print("Missing Values per Feature")
print("="*60)
missing_counts = features_clean.isnull().sum()
print(missing_counts[missing_counts > 0].to_string() if missing_counts.sum() > 0 else "No missing values in features.")

print("\n" + "="*60)
print("Credit Factor Statistics")
print("="*60)
credit_stats = features_clean['CREDIT_FACTOR'].describe()
print(credit_stats.to_string())
print(f"\nSkewness: {features_clean['CREDIT_FACTOR'].skew():.4f}")
print(f"Kurtosis: {features_clean['CREDIT_FACTOR'].kurtosis():.4f}")

# ============================================================
# 6. Summarize and Describe Features in Excel File
# ============================================================

print("\n" + "="*60)
print("Creating Feature Summary Excel File")
print("="*60)

# Create Excel writer
with pd.ExcelWriter('feature_summary.xlsx', engine='openpyxl') as writer:

    # Sheet 1: Feature Overview
    feature_overview = []
    for col in features_clean.columns:
        # Determine feature category
        if '_logret' in col:
            category = 'Log Return'
        elif '_diff' in col:
            category = 'Difference'
        elif '_range' in col:
            category = 'Range (log H/L)'
        elif 'RATE_SLOPE' in col:
            category = 'Interest Rate Slope'
        elif 'SPX_VOL' in col:
            category = 'Volatility'
        elif 'SPX_MAXDD' in col:
            category = 'Max Drawdown'
        elif 'CREDIT_FACTOR' in col and col == 'CREDIT_FACTOR':
            category = 'Credit Spread'
        elif '_RR_60d' in col:
            category = 'Recurrence Rate (60d)'
        elif '_DET_60d' in col:
            category = 'Determinism (60d)'
        elif '_5d' in col:
            category = 'Rolling 5D Mean'
        elif '_10d' in col:
            category = 'Rolling 10D Mean'
        elif '_q252' in col:
            category = 'Rolling 252D Q90'
        elif 'SPREAD_VIX_HIGH' in col:
            category = 'Interaction (Spread×VIX)'
        elif 'SPX_VIX_SPIKE' in col:
            category = 'Interaction (SPX×VIX)'
        elif 'DXY_SPREAD' in col:
            category = 'Interaction (DXY×Spread)'
        elif 'VIX_VOL_RATIO' in col:
            category = 'Vol Ratio (Implied/Realized)'
        elif '_STRESS' in col:
            category = 'Interaction (Z-Score×P_Stress)'
        else:
            category = 'Other'

        # Get statistics
        series = features_clean[col].dropna()
        feature_overview.append({
            'Feature': col,
            'Category': category,
            'Count': len(series),
            'Missing': features_clean[col].isnull().sum(),
            'Missing_Pct': f"{100*features_clean[col].isnull().sum()/len(features_clean):.2f}%",
            'Mean': series.mean() if len(series) > 0 else np.nan,
            'Std': series.std() if len(series) > 0 else np.nan,
            'Min': series.min() if len(series) > 0 else np.nan,
            'Q25': series.quantile(0.25) if len(series) > 0 else np.nan,
            'Median': series.median() if len(series) > 0 else np.nan,
            'Q75': series.quantile(0.75) if len(series) > 0 else np.nan,
            'Max': series.max() if len(series) > 0 else np.nan,
            'Skewness': series.skew() if len(series) > 0 else np.nan,
            'Kurtosis': series.kurtosis() if len(series) > 0 else np.nan
        })

    overview_df = pd.DataFrame(feature_overview)
    overview_df.to_excel(writer, sheet_name='Feature_Overview', index=False)

    # Sheet 2: Feature Categories Summary
    category_summary = overview_df.groupby('Category').agg({
        'Feature': 'count',
        'Missing_Pct': lambda x: x.iloc[0]  # Just take first as example
    }).reset_index()
    category_summary.columns = ['Category', 'Feature_Count', 'Example_Missing_Pct']
    category_summary.to_excel(writer, sheet_name='Category_Summary', index=False)

    # Sheet 3: Correlation Matrix (for key features only)
    key_features = [col for col in features_clean.columns if not any(x in col for x in ['_5d', '_10d', '_q252'])]
    if len(key_features) > 0:
        corr_matrix = features_clean[key_features].corr()
        corr_matrix.to_excel(writer, sheet_name='Correlation_Matrix')

    # Sheet 4: Feature Descriptions
    feature_descriptions = [
        {'Feature_Pattern': '*_logret', 'Description': 'Daily log return: ln(P_t / P_{t-1})'},
        {'Feature_Pattern': '*_diff', 'Description': 'Daily difference: P_t - P_{t-1}, used for rates and VIX'},
        {'Feature_Pattern': '*_range', 'Description': 'Daily range: log(High / Low), measures intraday volatility'},
        {'Feature_Pattern': 'RATE_SLOPE_10Y_3M', 'Description': 'Interest rate slope: 10Y yield - 3M yield, yield curve steepness'},
        {'Feature_Pattern': 'SPX_VOL_20D', 'Description': 'S&P 500 20-day rolling volatility (annualized)'},
        {'Feature_Pattern': 'SPX_MAXDD_20D', 'Description': 'S&P 500 20-day rolling maximum drawdown'},
        {'Feature_Pattern': 'CREDIT_FACTOR', 'Description': 'Credit spread factor: r_HYG - r_IEF, high yield vs treasury return'},
        {'Feature_Pattern': '*_5d', 'Description': '5-day rolling mean (min_periods=4)'},
        {'Feature_Pattern': '*_10d', 'Description': '10-day rolling mean (min_periods=8)'},
        {'Feature_Pattern': '*_q252', 'Description': '252-day rolling 90th percentile (min_periods=202)'},
        {'Feature_Pattern': 'SPREAD_VIX_HIGH', 'Description': 'spread_gate × vix_gate, where gates map values to 0~1 based on 252d rolling 80th-95th percentile'},
        {'Feature_Pattern': 'SPX_VIX_SPIKE', 'Description': '(-SPX_ret_1) × max(dVIX_1, 0), captures SPX loss × VIX spike interaction'},
        {'Feature_Pattern': 'DXY_SPREAD_INTERACT', 'Description': 'DXY_ret_20 × spread_z, where spread_z = (spread - rolling_mean) / rolling_std (252d)'},
        {'Feature_Pattern': 'VIX_VOL_RATIO', 'Description': 'VIX Close / SPX_VOL_20D, ratio of implied volatility to realized volatility'},
        {'Feature_Pattern': 'SPX_VOL_STRESS', 'Description': 'SPX_VOL_20D_zscore_252 × P_Stress, volatility z-score interaction with stress probability'},
        {'Feature_Pattern': 'HYG_RANGE_STRESS', 'Description': 'HYG_range_5d_zscore_252 × P_Stress, HYG range z-score interaction with stress probability'},
        {'Feature_Pattern': 'NASDAQ_RANGE_STRESS', 'Description': 'NASDAQ100_range_zscore_252 × P_Stress, NASDAQ range z-score interaction with stress probability'},
        {'Feature_Pattern': '*_RR_60d', 'Description': 'Recurrence Rate from 60-day window recurrence plot, ε = 0.5 × std'},
        {'Feature_Pattern': '*_DET_60d', 'Description': 'Determinism from 60-day window recurrence plot, proportion of recurrence points in diagonal lines'}
    ]
    desc_df = pd.DataFrame(feature_descriptions)
    desc_df.to_excel(writer, sheet_name='Feature_Descriptions', index=False)

    # Sheet 5: Sample Data (first 100 rows)
    features_clean.head(100).to_excel(writer, sheet_name='Sample_Data')

print(f"Feature summary saved to 'feature_summary.xlsx'")
print(f"Total features: {len(features_clean.columns)}")
print(f"Total observations: {len(features_clean)}")



