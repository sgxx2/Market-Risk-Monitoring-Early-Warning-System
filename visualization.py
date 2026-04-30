import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from scipy import stats

matplotlib.use('Agg')  # Use non-interactive backend

# Read the combine_data.csv with multi-index columns
df = pd.read_csv('combine_data.csv', header=[0, 1], index_col=0, parse_dates=True)

# Extract Close prices for all symbols
close_data = df['Close']

# Dataset description for reference
dataset_info = {
    'EEM': 'iShares MSCI Emerging Markets ETF',
    'EUROSTOXX50': 'Euro Stoxx 50 Index',
    'NASDAQ100': 'NASDAQ 100 Index',
    'NIKKEI225': 'Nikkei 225 Index (Japan)',
    'SP500': 'S&P 500 Index',
    'VIX': 'CBOE Volatility Index',
    'IEF': 'iShares 7-10 Year Treasury Bond ETF',
    'RATE_10Y': '10-Year Treasury Note Yield',
    'RATE_3M': '3-Month Treasury Bill Rate',
    'DXY': 'US Dollar Index',
    'EURUSD': 'Euro/US Dollar Exchange Rate',
    'USDCNH': 'US Dollar/Chinese Yuan Offshore Exchange Rate',
    'USDJPY': 'US Dollar/Japanese Yen Exchange Rate',
    'HYG': 'iShares iBoxx High Yield Corporate Bond ETF',
    'COMMODITY': 'S&P GSCI Commodity Index'
}

# Symbols with very different scales (interest rates are in decimal form ~0.01-0.17)
# These will use the secondary y-axis
secondary_axis_symbols = ['RATE_10Y', 'RATE_3M']

# Get all symbols
symbols = list(close_data.columns)

# Create figure with two y-axes
fig, ax1 = plt.subplots(figsize=(18, 12))
ax2 = ax1.twinx()  # Secondary y-axis for interest rates

# Define colors for each symbol
colors = plt.cm.tab20(np.linspace(0, 1, len(symbols)))

# Store line objects for legend
lines1 = []
lines2 = []
labels1 = []
labels2 = []

for i, symbol in enumerate(symbols):
    series = close_data[symbol].dropna()
    if len(series) > 0:
        # Determine which axis to use
        if symbol in secondary_axis_symbols:
            # Use secondary axis for interest rates (scale: 0-0.2)
            line, = ax2.plot(series.index, series.values, label=symbol,
                           color=colors[i], linewidth=1, linestyle='--')
            lines2.append(line)
            labels2.append(symbol)
            # Add annotation at the end of line
            ax2.annotate(symbol,
                        xy=(series.index[-1], series.iloc[-1]),
                        xytext=(5, 0), textcoords='offset points',
                        fontsize=8, color=colors[i], fontweight='bold')
        else:
            # Use primary axis for other symbols
            line, = ax1.plot(series.index, series.values, label=symbol,
                           color=colors[i], linewidth=1)
            lines1.append(line)
            labels1.append(symbol)
            # Add annotation at the end of line
            ax1.annotate(symbol,
                        xy=(series.index[-1], series.iloc[-1]),
                        xytext=(5, 0), textcoords='offset points',
                        fontsize=8, color=colors[i], fontweight='bold')

# Configure primary axis (left)
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Price / Index Value', fontsize=12, color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.set_yscale('log')  # Log scale for better visualization across different magnitudes
ax1.grid(True, alpha=0.3)

# Configure secondary axis (right) for interest rates
ax2.set_ylabel('Interest Rate (Decimal)', fontsize=12, color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Combine legends
all_lines = lines1 + lines2
all_labels = labels1 + labels2
ax1.legend(all_lines, all_labels, loc='upper left', bbox_to_anchor=(1.08, 1), fontsize=9)

ax1.set_title('Daily Close Time Series for All Symbols\n(Left Axis: Prices/Indices in Log Scale | Right Axis: Interest Rates)',
              fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('daily_close_timeseries_all_symbols.png', dpi=150, bbox_inches='tight')
plt.close()

print("Figure saved as 'daily_close_timeseries_all_symbols.png'")

# ============================================================
# Second Graph: Daily Returns Time Series (Subplots for each symbol)
# ============================================================

# Symbols that should use difference instead of log return
difference_symbols = ['RATE_10Y', 'RATE_3M', 'VIX']

# Calculate daily log returns for most symbols: ln(P_t / P_{t-1})
log_returns_data = np.log(close_data / close_data.shift(1))

# Calculate differences for interest rates and VIX: P_t - P_{t-1}
diff_data = close_data.diff()

# Calculate grid size for subplots
n_symbols = len(symbols)
n_cols = 3
n_rows = int(np.ceil(n_symbols / n_cols))

# Create figure with subplots (sharex=False for individual x-axes)
fig2, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows), sharex=False)
axes = axes.flatten()  # Flatten to 1D array for easy indexing

for i, symbol in enumerate(symbols):
    ax = axes[i]

    # Use difference for interest rates and VIX, log return for others
    if symbol in difference_symbols:
        series = diff_data[symbol].dropna()
        y_label = 'Difference'
        data_type = 'Diff'
    else:
        series = log_returns_data[symbol].dropna()
        y_label = 'Log Return'
        data_type = 'LogRet'

    if len(series) > 0:
        # Plot data
        ax.plot(series.index, series.values, color=colors[i], linewidth=0.6, alpha=0.8)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)  # Add zero line

        # Get date range for this symbol
        start_date = series.index[0].strftime('%Y-%m-%d')
        end_date = series.index[-1].strftime('%Y-%m-%d')
        ax.set_title(f'{symbol} [{data_type}] ({start_date} to {end_date})', fontsize=10, fontweight='bold', color=colors[i])
        ax.grid(True, alpha=0.3)
        ax.set_ylabel(y_label, fontsize=9)
        ax.set_xlabel('Date', fontsize=8)

        # Rotate x-axis labels for better readability
        ax.tick_params(axis='x', rotation=45, labelsize=7)

        # Add statistics annotation
        mean_ret = series.mean()
        std_ret = series.std()
        ax.annotate(f'μ={mean_ret:.4f}\nσ={std_ret:.4f}',
                    xy=(0.02, 0.95), xycoords='axes fraction',
                    fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Hide empty subplots
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)


fig2.suptitle('Daily Log Returns / Differences Time Series for All Symbols\n(Log Return for prices/indices, Difference for interest rates & VIX)', fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('daily_returns_timeseries_all_symbols.png', dpi=150, bbox_inches='tight')
plt.close()

print("Figure saved as 'daily_returns_timeseries_all_symbols.png'")

# ============================================================
# Third Graph: Correlation Heatmap for Log Returns and Differences
# ============================================================

# Combine log returns and differences into one DataFrame for correlation
combined_returns = pd.DataFrame()

for symbol in symbols:
    if symbol in difference_symbols:
        combined_returns[symbol] = diff_data[symbol]
    else:
        combined_returns[symbol] = log_returns_data[symbol]

# Calculate correlation matrix
corr_matrix = combined_returns.corr()

# Create heatmap figure
fig3, ax3 = plt.subplots(figsize=(14, 12))

# Create heatmap using imshow
im = ax3.imshow(corr_matrix.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)

# Set ticks and labels
ax3.set_xticks(np.arange(len(symbols)))
ax3.set_yticks(np.arange(len(symbols)))
ax3.set_xticklabels(symbols, fontsize=10, rotation=45, ha='right')
ax3.set_yticklabels(symbols, fontsize=10)

# Add colorbar
cbar = ax3.figure.colorbar(im, ax=ax3, shrink=0.8)
cbar.ax.set_ylabel('Correlation', rotation=-90, va='bottom', fontsize=12)

# Add correlation values as text annotations
for i in range(len(symbols)):
    for j in range(len(symbols)):
        value = corr_matrix.iloc[i, j]
        text_color = 'white' if abs(value) > 0.5 else 'black'
        ax3.text(j, i, f'{value:.2f}', ha='center', va='center',
                color=text_color, fontsize=8)

ax3.set_title('Correlation Heatmap of Daily Log Returns / Differences\n(Log Return for prices/indices, Difference for interest rates & VIX)',
              fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

print("Figure saved as 'correlation_heatmap.png'")

# ============================================================
# Fourth Graph: Pairwise Plot (Scatter Plot Matrix)
# ============================================================

# Use the combined_returns DataFrame (already contains log returns and differences)
# Drop NaN values for pairwise plotting
pairwise_data = combined_returns.dropna()

n_vars = len(symbols)

# Create pairwise plot figure
fig4, axes4 = plt.subplots(n_vars, n_vars, figsize=(24, 24))

# Define colors for histograms
hist_colors = plt.cm.tab20(np.linspace(0, 1, n_vars))

for i, var_i in enumerate(symbols):
    for j, var_j in enumerate(symbols):
        ax = axes4[i, j]

        if i == j:
            # Diagonal: histogram
            ax.hist(pairwise_data[var_i].dropna(), bins=50, color=hist_colors[i],
                   alpha=0.7, edgecolor='black', linewidth=0.3)
            ax.set_ylabel('Frequency', fontsize=7)
        else:
            # Off-diagonal: scatter plot
            # Get common valid data points
            valid_mask = pairwise_data[[var_i, var_j]].dropna().index
            x_data = pairwise_data.loc[valid_mask, var_j]
            y_data = pairwise_data.loc[valid_mask, var_i]

            ax.scatter(x_data, y_data, alpha=0.3, s=2, c='steelblue')

            # Add regression line
            if len(x_data) > 2:
                z = np.polyfit(x_data, y_data, 1)
                p = np.poly1d(z)
                x_line = np.linspace(x_data.min(), x_data.max(), 100)
                ax.plot(x_line, p(x_line), color='red', linewidth=1, alpha=0.7)

        # Set labels
        if i == n_vars - 1:
            ax.set_xlabel(var_j, fontsize=8, rotation=45, ha='right')
        if j == 0:
            ax.set_ylabel(var_i, fontsize=8)

        # Remove ticks for cleaner look (except edges)
        if i != n_vars - 1:
            ax.set_xticklabels([])
        else:
            ax.tick_params(axis='x', labelsize=6, rotation=45)
        if j != 0:
            ax.set_yticklabels([])
        else:
            ax.tick_params(axis='y', labelsize=6)

        ax.grid(True, alpha=0.2)

fig4.suptitle('Pairwise Plot of Daily Log Returns / Differences\n(Diagonal: Histograms | Off-diagonal: Scatter plots with regression lines)',
              fontsize=16, fontweight='bold', y=1.01)

plt.tight_layout()
plt.savefig('pairwise_plot.png', dpi=150, bbox_inches='tight')
plt.close()

print("Figure saved as 'pairwise_plot.png'")

# ============================================================
# Fifth Graph: KDE Plot for Every Symbol
# ============================================================

# Create figure with subplots for KDE plots
fig5, axes5 = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
axes5 = axes5.flatten()

for i, symbol in enumerate(symbols):
    ax = axes5[i]

    # Use difference for interest rates and VIX, log return for others
    if symbol in difference_symbols:
        series = diff_data[symbol].dropna()
        x_label = 'Difference'
        data_type = 'Diff'
    else:
        series = log_returns_data[symbol].dropna()
        x_label = 'Log Return'
        data_type = 'LogRet'

    if len(series) > 0:
        # Create KDE
        kde = stats.gaussian_kde(series.values)
        x_range = np.linspace(series.min(), series.max(), 500)
        kde_values = kde(x_range)

        # Plot KDE
        ax.fill_between(x_range, kde_values, alpha=0.5, color=colors[i])
        ax.plot(x_range, kde_values, color=colors[i], linewidth=2, label='Actual KDE')

        # Overlay normal distribution for comparison
        mean_val = series.mean()
        std_val = series.std()
        normal_pdf = stats.norm.pdf(x_range, mean_val, std_val)
        ax.plot(x_range, normal_pdf, color='black', linewidth=2, linestyle='--', label='Normal Dist.')

        # Add histogram for reference (normalized)
        ax.hist(series.values, bins=50, density=True, alpha=0.3,
               color=colors[i], edgecolor='black', linewidth=0.3)

        # Add vertical line at mean
        mean_val = series.mean()
        ax.axvline(x=mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.4f}')

        # Add vertical line at median
        median_val = series.median()
        ax.axvline(x=median_val, color='green', linestyle=':', linewidth=1.5, label=f'Median: {median_val:.4f}')

        # Statistics
        std_val = series.std()
        skew_val = series.skew()
        kurt_val = series.kurtosis()

        ax.set_title(f'{symbol} [{data_type}] Distribution', fontsize=10, fontweight='bold', color=colors[i])
        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=7)

        # Add statistics annotation
        ax.annotate(f'σ={std_val:.4f}\nSkew={skew_val:.2f}\nKurt={kurt_val:.2f}',
                    xy=(0.02, 0.95), xycoords='axes fraction',
                    fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Hide empty subplots
for j in range(i + 1, len(axes5)):
    axes5[j].set_visible(False)

fig5.suptitle('KDE (Kernel Density Estimation) Plots for All Symbols\n(Log Return for prices/indices, Difference for interest rates & VIX)',
              fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('kde_plots.png', dpi=150, bbox_inches='tight')
plt.close()

print("Figure saved as 'kde_plots.png'")

# ============================================================
# Sixth: Detect Outliers and Abnormal Data for Every Symbol
# ============================================================

# Create a dictionary to store outlier information for each symbol
outlier_results = {}

# Methods for outlier detection:
# 1. IQR method (Interquartile Range)
# 2. Z-score method (values beyond 3 standard deviations)
# 3. Modified Z-score using MAD (Median Absolute Deviation)

def detect_outliers_iqr(data, multiplier=1.5):
    """Detect outliers using IQR method"""
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    return outliers, lower_bound, upper_bound

def detect_outliers_zscore(data, threshold=3):
    """Detect outliers using Z-score method"""
    mean = data.mean()
    std = data.std()
    z_scores = (data - mean) / std
    outliers = data[np.abs(z_scores) > threshold]
    return outliers, z_scores[np.abs(z_scores) > threshold]

def detect_outliers_mad(data, threshold=3.5):
    """Detect outliers using Modified Z-score (MAD method)"""
    median = data.median()
    mad = np.median(np.abs(data - median))
    if mad == 0:
        mad = 1e-10  # Avoid division by zero
    modified_z_scores = 0.6745 * (data - median) / mad
    outliers = data[np.abs(modified_z_scores) > threshold]
    return outliers, modified_z_scores[np.abs(modified_z_scores) > threshold]

# Create Excel writer
with pd.ExcelWriter('outliers_analysis.xlsx', engine='openpyxl') as writer:

    # Summary sheet data
    summary_data = []

    for symbol in symbols:
        # Use difference for interest rates and VIX, log return for others
        if symbol in difference_symbols:
            series = diff_data[symbol].dropna()
            data_type = 'Difference'
        else:
            series = log_returns_data[symbol].dropna()
            data_type = 'Log Return'

        if len(series) == 0:
            continue

        # Detect outliers using all three methods
        iqr_outliers, iqr_lower, iqr_upper = detect_outliers_iqr(series)
        zscore_outliers, zscore_values = detect_outliers_zscore(series)
        mad_outliers, mad_scores = detect_outliers_mad(series)

        # Create DataFrame for this symbol's outliers
        symbol_outliers = pd.DataFrame()

        # IQR outliers
        if len(iqr_outliers) > 0:
            iqr_df = pd.DataFrame({
                'Date': iqr_outliers.index,
                'Value': iqr_outliers.values,
                'Method': 'IQR',
                'Lower_Bound': iqr_lower,
                'Upper_Bound': iqr_upper,
                'Direction': ['Below' if v < iqr_lower else 'Above' for v in iqr_outliers.values]
            })
            symbol_outliers = pd.concat([symbol_outliers, iqr_df], ignore_index=True)

        # Z-score outliers
        if len(zscore_outliers) > 0:
            zscore_df = pd.DataFrame({
                'Date': zscore_outliers.index,
                'Value': zscore_outliers.values,
                'Method': 'Z-Score',
                'Z_Score': zscore_values.values,
                'Direction': ['Negative' if z < 0 else 'Positive' for z in zscore_values.values]
            })
            symbol_outliers = pd.concat([symbol_outliers, zscore_df], ignore_index=True)

        # MAD outliers
        if len(mad_outliers) > 0:
            mad_df = pd.DataFrame({
                'Date': mad_outliers.index,
                'Value': mad_outliers.values,
                'Method': 'MAD',
                'Modified_Z_Score': mad_scores.values,
                'Direction': ['Negative' if z < 0 else 'Positive' for z in mad_scores.values]
            })
            symbol_outliers = pd.concat([symbol_outliers, mad_df], ignore_index=True)

        # Sort by date
        if len(symbol_outliers) > 0:
            symbol_outliers = symbol_outliers.sort_values('Date').reset_index(drop=True)
            # Write to Excel sheet
            symbol_outliers.to_excel(writer, sheet_name=symbol, index=False)

        # Add to summary
        summary_data.append({
            'Symbol': symbol,
            'Data_Type': data_type,
            'Total_Observations': len(series),
            'IQR_Outliers': len(iqr_outliers),
            'ZScore_Outliers': len(zscore_outliers),
            'MAD_Outliers': len(mad_outliers),
            'IQR_Outlier_Pct': f"{100*len(iqr_outliers)/len(series):.2f}%",
            'ZScore_Outlier_Pct': f"{100*len(zscore_outliers)/len(series):.2f}%",
            'MAD_Outlier_Pct': f"{100*len(mad_outliers)/len(series):.2f}%",
            'Mean': series.mean(),
            'Std': series.std(),
            'Min': series.min(),
            'Max': series.max(),
            'Skewness': series.skew(),
            'Kurtosis': series.kurtosis()
        })

        # Store for reference
        outlier_results[symbol] = {
            'iqr': iqr_outliers,
            'zscore': zscore_outliers,
            'mad': mad_outliers
        }

    # Write summary sheet
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_excel(writer, sheet_name='Summary', index=False)

    # Create a combined outliers sheet (all symbols together)
    all_outliers = []
    for symbol in symbols:
        if symbol in difference_symbols:
            series = diff_data[symbol].dropna()
            data_type = 'Difference'
        else:
            series = log_returns_data[symbol].dropna()
            data_type = 'Log Return'

        if len(series) == 0:
            continue

        iqr_outliers, iqr_lower, iqr_upper = detect_outliers_iqr(series)

        for date, value in iqr_outliers.items():
            all_outliers.append({
                'Symbol': symbol,
                'Date': date,
                'Value': value,
                'Data_Type': data_type,
                'IQR_Lower': iqr_lower,
                'IQR_Upper': iqr_upper
            })

    if all_outliers:
        all_outliers_df = pd.DataFrame(all_outliers)
        all_outliers_df = all_outliers_df.sort_values(['Date', 'Symbol']).reset_index(drop=True)
        all_outliers_df.to_excel(writer, sheet_name='All_Outliers_Combined', index=False)

print("Outliers analysis saved to 'outliers_analysis.xlsx'")

# ============================================================
# Seventh Graph: Recurrence Plots for Key Features
# ============================================================

print("\nGenerating Recurrence Plots...")

# Load features data
features = pd.read_csv('features.csv', index_col=0, parse_dates=True)

# Features to plot
recurrence_features = ['SP500_logret', 'VIX_diff', 'CREDIT_FACTOR', 'DXY_logret']

def compute_recurrence_matrix(series):
    """
    Compute the recurrence matrix for a time series.

    Parameters:
    -----------
    series : array-like
        The time series data

    Returns:
    --------
    recurrence_matrix : numpy array
        Binary matrix where 1 indicates recurrence
    epsilon : float
        The threshold used (0.5 * std)
    """
    # Clean data
    data = series.dropna().values
    n = len(data)

    # Compute pairwise distance matrix
    # Using Euclidean distance: |x_i - x_j|
    distance_matrix = np.abs(data.reshape(-1, 1) - data.reshape(1, -1))

    # Use 0.5 * std as threshold
    epsilon = 0.5 * np.std(data)

    # Create recurrence matrix
    recurrence_matrix = (distance_matrix <= epsilon).astype(int)

    return recurrence_matrix, epsilon

# Create figure with 2x2 subplots
fig_rp, axes_rp = plt.subplots(2, 2, figsize=(14, 14))
axes_rp = axes_rp.flatten()

# Colors for each plot
rp_colors = ['Blues', 'Reds', 'Greens', 'Purples']

for i, feature in enumerate(recurrence_features):
    ax = axes_rp[i]

    if feature not in features.columns:
        ax.text(0.5, 0.5, f'{feature}\nNot Found', ha='center', va='center', fontsize=14)
        ax.set_title(f'{feature} - Not Available')
        continue

    series = features[feature].dropna()

    if len(series) == 0:
        ax.text(0.5, 0.5, f'{feature}\nNo Data', ha='center', va='center', fontsize=14)
        ax.set_title(f'{feature} - No Data')
        continue

    # Use 60-day time window for recurrence plot
    window_days = 60
    if len(series) > window_days:
        series = series.tail(window_days)

    # Compute recurrence matrix
    rec_matrix, epsilon = compute_recurrence_matrix(series)

    # Plot recurrence matrix
    im = ax.imshow(rec_matrix, cmap=rp_colors[i], origin='lower', aspect='equal')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Recurrence', fontsize=9)

    # Calculate recurrence rate (RR)
    n = len(rec_matrix)
    rr = (np.sum(rec_matrix) - n) / (n * (n - 1))  # Exclude diagonal

    # Calculate determinism (DET) - proportion of recurrence points forming diagonal lines
    # Simplified: count diagonal structures
    diag_sum = 0
    for k in range(2, n):
        diag = np.diag(rec_matrix, k)
        # Count consecutive 1s of length >= 2
        diag_str = ''.join(map(str, diag))
        for run in diag_str.split('0'):
            if len(run) >= 2:
                diag_sum += len(run)
    det = diag_sum / (np.sum(rec_matrix) - n) if (np.sum(rec_matrix) - n) > 0 else 0

    # Set labels and title
    ax.set_xlabel('Time Index', fontsize=10)
    ax.set_ylabel('Time Index', fontsize=10)
    ax.set_title(f'{feature}\n(n={len(series)}, ε={epsilon:.4f})',
                 fontsize=11, fontweight='bold')

    # Add statistics annotation
    ax.annotate(f'RR={rr:.3f}\nDET={det:.3f}',
                xy=(0.02, 0.98), xycoords='axes fraction',
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Add date range info
    date_range = f'{series.index[0].strftime("%Y-%m-%d")} to {series.index[-1].strftime("%Y-%m-%d")}'
    ax.annotate(date_range,
                xy=(0.98, 0.02), xycoords='axes fraction',
                fontsize=8, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

fig_rp.suptitle('Recurrence Plots for Key Financial Features\n(60-Day Time Window, ε = 0.5 × std)',
                fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('recurrence_plots.png', dpi=150, bbox_inches='tight')
plt.close()

print("Figure saved as 'recurrence_plots.png'")


# ============================================================
# Eighth Graph: Target Crash Days Timeline
# ============================================================

print("\nGenerating target crash day timeline plot...")

try:
    target_df = pd.read_csv('target_label.csv', index_col=0, parse_dates=True)

    # Prefer max-drawdown style column if available, fallback to 20d forward return.
    metric_candidates = [
        'SP500_forward_maxdd_20d',
        'SP500_20d_forward_maxdd',
        'SP500_20d_maxdd',
        'SP500_20d_forward_return',
    ]
    metric_col = next((c for c in metric_candidates if c in target_df.columns), None)
    if metric_col is None:
        raise ValueError("No SP500 20d drawdown/forward-return metric found in target_label.csv")

    metric_s = target_df[metric_col].dropna()
    sp500_label = target_df.get('SP500_Label', pd.Series(index=target_df.index, dtype=float))
    sp500_crash_dates = target_df.index[(sp500_label == 1) & target_df.index.isin(metric_s.index)]
    focus_start = pd.Timestamp('2007-01-01')
    latest_date = target_df.index.max()
    sp500_crash_dates_focus = sp500_crash_dates[sp500_crash_dates >= focus_start]
    metric_focus = metric_s[metric_s.index >= focus_start]

    fig8, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(18, 8), sharex=False, gridspec_kw={'height_ratios': [3, 2]}
    )
    cmap = plt.cm.RdYlGn  # continuous colormap: larger drawdown (more negative) tends toward red
    norm = plt.Normalize(vmin=float(metric_s.min()), vmax=float(metric_s.max()))

    # Top panel: full-period drawdown metric with crash-day markers
    ax_top.plot(metric_s.index, metric_s.values, color='gray', linewidth=1.0, alpha=0.6, label=f'{metric_col} (all days)')
    if len(sp500_crash_dates) > 0:
        ax_top.scatter(
            sp500_crash_dates,
            metric_s.loc[sp500_crash_dates].values,
            c=metric_s.loc[sp500_crash_dates].values,
            cmap=cmap,
            norm=norm,
            s=22,
            alpha=0.95,
            edgecolors='black',
            linewidths=0.2,
            label=f'SP500 Label=1 (n={len(sp500_crash_dates)})',
            zorder=3,
        )
    if metric_col == 'SP500_20d_forward_return':
        ax_top.axhline(y=-0.06, color='black', linestyle='--', linewidth=1.0, label='Crash Threshold (-6%)')
    ax_top.set_ylabel('20d Drawdown Metric')
    ax_top.set_title(f'SP500 Crash Days (Full Period) | Metric: {metric_col}')
    ax_top.legend(loc='upper left', fontsize=9)
    ax_top.grid(True, alpha=0.3)

    # Bottom panel: zoom from 2007 to latest with same metric
    ax_bottom.plot(metric_focus.index, metric_focus.values, color='gray', linewidth=1.0, alpha=0.6, label=f'{metric_col} (2007+)')
    if len(sp500_crash_dates_focus) > 0:
        ax_bottom.scatter(
            sp500_crash_dates_focus,
            metric_s.loc[sp500_crash_dates_focus].values,
            c=metric_s.loc[sp500_crash_dates_focus].values,
            cmap=cmap,
            norm=norm,
            marker='o',
            s=26,
            alpha=0.95,
            edgecolors='black',
            linewidths=0.2,
            label=f'SP500 Label=1 (2007+, n={len(sp500_crash_dates_focus)})'
        )
    if metric_col == 'SP500_20d_forward_return':
        ax_bottom.axhline(y=-0.06, color='black', linestyle='--', linewidth=1.0, label='Crash Threshold (-6%)')

    ax_bottom.set_xlim(focus_start, latest_date)
    ax_bottom.set_xlabel('Date')
    ax_bottom.set_ylabel('20d Drawdown Metric')
    ax_bottom.grid(True, alpha=0.3)
    ax_bottom.set_title(f'SP500 Crash Days (2007-01-01 to {latest_date.date()})')
    ax_bottom.legend(loc='upper left', fontsize=9)

    # Shared continuous colorbar for drawdown magnitude
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # Reserve right margin for colorbar to avoid overlap/clutter on the right side
    fig8.subplots_adjust(right=0.90, hspace=0.20)
    cax = fig8.add_axes([0.92, 0.15, 0.015, 0.70])  # [left, bottom, width, height]
    cbar = fig8.colorbar(sm, cax=cax)
    cbar.set_label(f'{metric_col} (drawdown magnitude)', fontsize=9)

    plt.savefig('target_crash_timeline.png', dpi=150)
    plt.close()
    print("Figure saved as 'target_crash_timeline.png'")

except Exception as e:
    print(f"Warning: Failed to generate target crash timeline plot: {e}")
