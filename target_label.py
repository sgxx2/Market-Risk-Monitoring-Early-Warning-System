"""
Generate Target Labels for Data Mining Project

Target: Predict if SP500/NASDAQ100 will drop more than 6% in the next 20 trading days

Labels:
- Label = 1 if (20d forward return < -6%)
- Label = 0 otherwise

Output: target_label.csv
"""

import pandas as pd
import numpy as np

# ============================================================
# Load Data
# ============================================================

print("Loading data...")

# Read the combine_data.csv with multi-index columns
df = pd.read_csv('combine_data.csv', header=[0, 1], index_col=0, parse_dates=True)

# Extract SP500 and NASDAQ100 Close price
sp500_close = df['Close']['SP500'].dropna()
nasdaq_close = df['Close']['NASDAQ100'].dropna()

print(f"SP500 data loaded.")
print(f"  Date range: {sp500_close.index[0]} to {sp500_close.index[-1]}")
print(f"  Total trading days: {len(sp500_close)}")

print(f"\nNASDAQ100 data loaded.")
print(f"  Date range: {nasdaq_close.index[0]} to {nasdaq_close.index[-1]}")
print(f"  Total trading days: {len(nasdaq_close)}")

# ============================================================
# Calculate Features
# ============================================================

print("\nCalculating features...")

# Create target DataFrame using SP500 index as base
target_df = pd.DataFrame(index=sp500_close.index)

# ----- SP500 Features -----
# SP500 Close price
target_df['SP500_Close'] = sp500_close

# SP500 daily log return: ln(P_t / P_{t-1})
target_df['SP500_logret'] = np.log(sp500_close / sp500_close.shift(1))

# SP500 20-day return (backward looking): (P_t - P_{t-20}) / P_{t-20}
target_df['SP500_20d_return'] = (sp500_close - sp500_close.shift(20)) / sp500_close.shift(20)

# SP500 20-day forward return: (P_{t+20} - P_t) / P_t
sp500_20d_forward_return = (sp500_close.shift(-20) - sp500_close) / sp500_close
target_df['SP500_20d_forward_return'] = sp500_20d_forward_return

# SP500 Label = 1 if 20d forward return < -6%
threshold = -0.06
target_df['SP500_Label'] = (sp500_20d_forward_return < threshold).astype(int)
target_df.loc[target_df['SP500_20d_forward_return'].isna(), 'SP500_Label'] = np.nan

# ----- NASDAQ100 Features -----
# NASDAQ100 Close price
target_df['NASDAQ100_Close'] = nasdaq_close

# NASDAQ100 daily log return: ln(P_t / P_{t-1})
target_df['NASDAQ100_logret'] = np.log(nasdaq_close / nasdaq_close.shift(1))

# NASDAQ100 20-day return (backward looking): (P_t - P_{t-20}) / P_{t-20}
target_df['NASDAQ100_20d_return'] = (nasdaq_close - nasdaq_close.shift(20)) / nasdaq_close.shift(20)

# NASDAQ100 20-day forward return: (P_{t+20} - P_t) / P_t
nasdaq_20d_forward_return = (nasdaq_close.shift(-20) - nasdaq_close) / nasdaq_close
target_df['NASDAQ100_20d_forward_return'] = nasdaq_20d_forward_return

# NASDAQ100 Label = 1 if 20d forward return < -6%
target_df['NASDAQ100_Label'] = (nasdaq_20d_forward_return < threshold).astype(int)
target_df.loc[target_df['NASDAQ100_20d_forward_return'].isna(), 'NASDAQ100_Label'] = np.nan

# ============================================================
# Statistics
# ============================================================

print("\n" + "="*60)
print("Target Label Statistics")
print("="*60)

# ----- SP500 Statistics -----
print("\n----- SP500 -----")
sp500_label_counts = target_df['SP500_Label'].value_counts(dropna=True)
sp500_total_labeled = sp500_label_counts.sum()

print(f"Total observations: {len(target_df)}")
print(f"Labeled observations: {int(sp500_total_labeled)}")
print(f"Unlabeled (last 20 days): {target_df['SP500_Label'].isna().sum()}")

print(f"\nLabel distribution:")
print(f"  Label = 0 (No crash): {int(sp500_label_counts.get(0, 0))} ({100*sp500_label_counts.get(0, 0)/sp500_total_labeled:.2f}%)")
print(f"  Label = 1 (Crash >6%): {int(sp500_label_counts.get(1, 0))} ({100*sp500_label_counts.get(1, 0)/sp500_total_labeled:.2f}%)")

# ----- NASDAQ100 Statistics -----
print("\n----- NASDAQ100 -----")
nasdaq_label_counts = target_df['NASDAQ100_Label'].value_counts(dropna=True)
nasdaq_total_labeled = nasdaq_label_counts.sum()

print(f"Total observations with NASDAQ data: {target_df['NASDAQ100_Close'].notna().sum()}")
print(f"Labeled observations: {int(nasdaq_total_labeled)}")
print(f"Unlabeled: {target_df['NASDAQ100_Label'].isna().sum()}")

print(f"\nLabel distribution:")
print(f"  Label = 0 (No crash): {int(nasdaq_label_counts.get(0, 0))} ({100*nasdaq_label_counts.get(0, 0)/nasdaq_total_labeled:.2f}%)")
print(f"  Label = 1 (Crash >6%): {int(nasdaq_label_counts.get(1, 0))} ({100*nasdaq_label_counts.get(1, 0)/nasdaq_total_labeled:.2f}%)")

# ============================================================
# Show Crash Periods
# ============================================================

def show_crash_periods(target_df, label_col, forward_return_col, index_name):
    """Display crash periods for a given index"""
    crash_dates = target_df[target_df[label_col] == 1].index
    print(f"\n" + "="*60)
    print(f"{index_name}: Days labeled as 1 (crash >6% in next 20 days): {len(crash_dates)}")
    print("="*60)

    if len(crash_dates) > 0:
        crash_df = target_df[target_df[label_col] == 1][[forward_return_col]].copy()
        crash_df['Return_Pct'] = crash_df[forward_return_col] * 100

        print("\nCrash periods (showing start and end of consecutive crash labels):")
        print("-" * 60)

        crash_dates_list = crash_df.index.tolist()
        periods = []
        start_date = crash_dates_list[0]
        prev_date = crash_dates_list[0]

        for i, date in enumerate(crash_dates_list[1:], 1):
            days_diff = (date - prev_date).days
            if days_diff > 7:
                periods.append((start_date, prev_date))
                start_date = date
            prev_date = date
        periods.append((start_date, prev_date))

        print(f"{'Period':<8} {'Start Date':<12} {'End Date':<12} {'Days':<6} {'Min Return':<12}")
        print("-" * 60)

        for idx, (start, end) in enumerate(periods, 1):
            period_data = crash_df.loc[start:end]
            min_ret = period_data['Return_Pct'].min()
            days = len(period_data)
            print(f"{idx:<8} {start.strftime('%Y-%m-%d'):<12} {end.strftime('%Y-%m-%d'):<12} {days:<6} {min_ret:>8.2f}%")

show_crash_periods(target_df, 'SP500_Label', 'SP500_20d_forward_return', 'SP500')
show_crash_periods(target_df, 'NASDAQ100_Label', 'NASDAQ100_20d_forward_return', 'NASDAQ100')

# ============================================================
# Save to CSV
# ============================================================

print("\n" + "="*60)
print("Saving target labels...")
print("="*60)

target_df.to_csv('target_label.csv')

print(f"\nTarget labels saved to 'target_label.csv'")
print(f"Columns: {list(target_df.columns)}")
print(f"Shape: {target_df.shape}")

# Show sample data
print("\nSample data (first 5 rows):")
print(target_df.head().to_string())

print("\nSample data (last 5 rows):")
print(target_df.tail().to_string())

# ============================================================
# KDE Plot for Return Distributions
# 4 plots: SP500 Crash, SP500 All, NASDAQ100 Crash, NASDAQ100 All
# ============================================================

import matplotlib.pyplot as plt
from scipy import stats

print("\n" + "="*60)
print("Generating KDE plots for return distributions...")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Get all returns data
sp500_all_returns = target_df['SP500_20d_forward_return'].dropna() * 100
nasdaq_all_returns = target_df['NASDAQ100_20d_forward_return'].dropna() * 100

# Get crash returns data
sp500_crash_returns = target_df[target_df['SP500_Label'] == 1]['SP500_20d_forward_return'].dropna() * 100
nasdaq_crash_returns = target_df[target_df['NASDAQ100_Label'] == 1]['NASDAQ100_20d_forward_return'].dropna() * 100

# ----- Plot 1: SP500 Crash Distribution -----
ax1 = axes[0, 0]
sp500_crash_returns.plot(kind='kde', ax=ax1, color='blue', linewidth=2, label='Crash KDE')
ax1.hist(sp500_crash_returns, bins=30, density=True, alpha=0.3, color='blue')

# Normal distribution with crash sample parameters
sp500_crash_mean = sp500_crash_returns.mean()
sp500_crash_std = sp500_crash_returns.std()
x_range1 = np.linspace(sp500_crash_returns.min() - 5, -5, 200)
normal_pdf1 = stats.norm.pdf(x_range1, sp500_crash_mean, sp500_crash_std)
ax1.plot(x_range1, normal_pdf1, 'r--', linewidth=2, label=f'Normal (μ={sp500_crash_mean:.1f}%, σ={sp500_crash_std:.1f}%)')

ax1.axvline(x=threshold * 100, color='green', linestyle=':', linewidth=1.5, label=f"Threshold ({threshold*100:.0f}%)")
ax1.set_xlabel('20-Day Forward Return (%)', fontsize=10)
ax1.set_ylabel('Density', fontsize=10)
ax1.set_title(f'SP500 Crash Distribution (Label=1, n={len(sp500_crash_returns)})', fontsize=11, fontweight='bold')
ax1.legend(loc='upper left', fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.annotate(f'Mean: {sp500_crash_mean:.2f}%\nStd: {sp500_crash_std:.2f}%\nMin: {sp500_crash_returns.min():.2f}%',
             xy=(0.98, 0.98), xycoords='axes fraction', fontsize=8, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# ----- Plot 2: SP500 All Distribution -----
ax2 = axes[0, 1]
sp500_all_returns.plot(kind='kde', ax=ax2, color='blue', linewidth=2, label='All Returns KDE')
ax2.hist(sp500_all_returns, bins=50, density=True, alpha=0.3, color='blue')

# Normal distribution with full sample parameters
sp500_all_mean = sp500_all_returns.mean()
sp500_all_std = sp500_all_returns.std()
x_range2 = np.linspace(sp500_all_returns.min() - 5, sp500_all_returns.max() + 5, 200)
normal_pdf2 = stats.norm.pdf(x_range2, sp500_all_mean, sp500_all_std)
ax2.plot(x_range2, normal_pdf2, 'r--', linewidth=2, label=f'Normal (μ={sp500_all_mean:.1f}%, σ={sp500_all_std:.1f}%)')

ax2.axvline(x=threshold * 100, color='green', linestyle=':', linewidth=1.5, label=f"Threshold ({threshold*100:.0f}%)")
ax2.set_xlabel('20-Day Forward Return (%)', fontsize=10)
ax2.set_ylabel('Density', fontsize=10)
ax2.set_title(f'SP500 All Returns Distribution (n={len(sp500_all_returns)})', fontsize=11, fontweight='bold')
ax2.legend(loc='upper left', fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.annotate(f'Mean: {sp500_all_mean:.2f}%\nStd: {sp500_all_std:.2f}%\nSkew: {sp500_all_returns.skew():.2f}\nKurt: {sp500_all_returns.kurtosis():.2f}',
             xy=(0.98, 0.98), xycoords='axes fraction', fontsize=8, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# ----- Plot 3: NASDAQ100 Crash Distribution -----
ax3 = axes[1, 0]
nasdaq_crash_returns.plot(kind='kde', ax=ax3, color='purple', linewidth=2, label='Crash KDE')
ax3.hist(nasdaq_crash_returns, bins=30, density=True, alpha=0.3, color='purple')

# Normal distribution with crash sample parameters
nasdaq_crash_mean = nasdaq_crash_returns.mean()
nasdaq_crash_std = nasdaq_crash_returns.std()
x_range3 = np.linspace(nasdaq_crash_returns.min() - 5, -5, 200)
normal_pdf3 = stats.norm.pdf(x_range3, nasdaq_crash_mean, nasdaq_crash_std)
ax3.plot(x_range3, normal_pdf3, 'r--', linewidth=2, label=f'Normal (μ={nasdaq_crash_mean:.1f}%, σ={nasdaq_crash_std:.1f}%)')

ax3.axvline(x=threshold * 100, color='green', linestyle=':', linewidth=1.5, label=f"Threshold ({threshold*100:.0f}%)")
ax3.set_xlabel('20-Day Forward Return (%)', fontsize=10)
ax3.set_ylabel('Density', fontsize=10)
ax3.set_title(f'NASDAQ100 Crash Distribution (Label=1, n={len(nasdaq_crash_returns)})', fontsize=11, fontweight='bold')
ax3.legend(loc='upper left', fontsize=8)
ax3.grid(True, alpha=0.3)
ax3.annotate(f'Mean: {nasdaq_crash_mean:.2f}%\nStd: {nasdaq_crash_std:.2f}%\nMin: {nasdaq_crash_returns.min():.2f}%',
             xy=(0.98, 0.98), xycoords='axes fraction', fontsize=8, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# ----- Plot 4: NASDAQ100 All Distribution -----
ax4 = axes[1, 1]
nasdaq_all_returns.plot(kind='kde', ax=ax4, color='purple', linewidth=2, label='All Returns KDE')
ax4.hist(nasdaq_all_returns, bins=50, density=True, alpha=0.3, color='purple')

# Normal distribution with full sample parameters
nasdaq_all_mean = nasdaq_all_returns.mean()
nasdaq_all_std = nasdaq_all_returns.std()
x_range4 = np.linspace(nasdaq_all_returns.min() - 5, nasdaq_all_returns.max() + 5, 200)
normal_pdf4 = stats.norm.pdf(x_range4, nasdaq_all_mean, nasdaq_all_std)
ax4.plot(x_range4, normal_pdf4, 'r--', linewidth=2, label=f'Normal (μ={nasdaq_all_mean:.1f}%, σ={nasdaq_all_std:.1f}%)')

ax4.axvline(x=threshold * 100, color='green', linestyle=':', linewidth=1.5, label=f"Threshold ({threshold*100:.0f}%)")
ax4.set_xlabel('20-Day Forward Return (%)', fontsize=10)
ax4.set_ylabel('Density', fontsize=10)
ax4.set_title(f'NASDAQ100 All Returns Distribution (n={len(nasdaq_all_returns)})', fontsize=11, fontweight='bold')
ax4.legend(loc='upper left', fontsize=8)
ax4.grid(True, alpha=0.3)
ax4.annotate(f'Mean: {nasdaq_all_mean:.2f}%\nStd: {nasdaq_all_std:.2f}%\nSkew: {nasdaq_all_returns.skew():.2f}\nKurt: {nasdaq_all_returns.kurtosis():.2f}',
             xy=(0.98, 0.98), xycoords='axes fraction', fontsize=8, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.suptitle('20-Day Forward Return Distributions: Crash Events vs All Data\nCompared with Normal Distribution',
             fontsize=13, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('crash_distribution_kde.png', dpi=150, bbox_inches='tight')
plt.close()

print("KDE plot saved as 'crash_distribution_kde.png'")

