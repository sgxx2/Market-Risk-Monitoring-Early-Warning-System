"""
Generate Regime Labels using Hidden Markov Model (HMM)

This script uses HMM to identify market regimes (states) for each trading day.

Variables:
- SPX_VOL_20D: S&P 500 20-day rolling volatility
- VIX Close: VIX index close price
- CREDIT_FACTOR: Credit spread factor (HYG - IEF return)
- RATE_SLOPE_10Y_3M: Interest rate slope (10Y - 3M yield)

States (3 total):
- Stress: Highest average VIX
- Transitional: Medium average VIX
- Calm: Lowest average VIX

HMM Estimation:
- Rolling window: 1200 trading days (~5 years)
- Re-estimate monthly (every ~21 trading days)
- Forward-only algorithm for state probabilities (no data leakage)

Output: regime_label.csv with P(Stress, t) and deltaP
"""

import pandas as pd
import numpy as np
from hmmlearn import hmm
from scipy.stats import multivariate_normal
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Load Data
# ============================================================

print("Loading data...")

# Load features data
features = pd.read_csv('features.csv', index_col=0, parse_dates=True)

# Load combine_data for VIX close
combine_data = pd.read_csv('combine_data.csv', header=[0, 1], index_col=0, parse_dates=True)
vix_close = combine_data['Close']['VIX']

print(f"Features data shape: {features.shape}")
print(f"VIX close data shape: {vix_close.shape}")

# ============================================================
# Prepare Variables
# ============================================================

print("\nPreparing variables...")

# Select required variables
regime_df = pd.DataFrame(index=features.index)

# SPX_VOL_20D from features
regime_df['SPX_VOL_20D'] = features['SPX_VOL_20D']

# VIX Close from combine_data
regime_df['VIX_Close'] = vix_close

# CREDIT_FACTOR from features
regime_df['CREDIT_FACTOR'] = features['CREDIT_FACTOR']

# RATE_SLOPE_10Y_3M from features
regime_df['RATE_SLOPE_10Y_3M'] = features['RATE_SLOPE_10Y_3M']

# Align data - use the latest starting symbol as base
print("\nData ranges before alignment:")
for col in regime_df.columns:
    valid_data = regime_df[col].dropna()
    if len(valid_data) > 0:
        print(f"  {col}: {valid_data.index[0]} to {valid_data.index[-1]} ({len(valid_data)} points)")

# Drop rows with any NaN
regime_df_aligned = regime_df.dropna()

print(f"\nAligned data range: {regime_df_aligned.index[0]} to {regime_df_aligned.index[-1]}")
print(f"Aligned data points: {len(regime_df_aligned)}")

# ============================================================
# Calculate 250-day Rolling Z-Score (Prevent Data Leakage)
# ============================================================

print("\nCalculating 250-day rolling z-score (preventing data leakage)...")

rolling_window = 250
min_periods = 200  # 80% of window

# Create z-score DataFrame
zscore_df = pd.DataFrame(index=regime_df_aligned.index)

for col in ['SPX_VOL_20D', 'VIX_Close', 'CREDIT_FACTOR', 'RATE_SLOPE_10Y_3M']:
    series = regime_df_aligned[col]
    
    # Rolling z-score: (x_t - rolling_mean_{1:t}) / rolling_std_{1:t}
    # This only uses data from time 1 to t, no future data
    rolling_mean = series.rolling(window=rolling_window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window=rolling_window, min_periods=min_periods).std()
    
    zscore = (series - rolling_mean) / rolling_std.clip(lower=1e-8)
    zscore_df[f'{col}_zscore'] = zscore

# Drop NaN from z-score calculation
zscore_df = zscore_df.dropna()

print(f"Z-score data range: {zscore_df.index[0]} to {zscore_df.index[-1]}")
print(f"Z-score data points: {len(zscore_df)}")

# ============================================================
# Rolling Window HMM Estimation Parameters
# ============================================================

print("\n" + "="*60)
print("Rolling Window HMM Estimation")
print("="*60)

HMM_WINDOW = 1200  # 1200 trading days (~5 years) for HMM estimation
REFIT_INTERVAL = 21  # Re-estimate monthly (~21 trading days)
N_STATES = 3

print(f"HMM estimation window: {HMM_WINDOW} trading days")
print(f"Re-estimation interval: {REFIT_INTERVAL} trading days (monthly)")
print(f"Number of states: {N_STATES}")

# ============================================================
# Forward-Only Algorithm for State Probabilities
# ============================================================

def compute_log_emission(model, x):
    """Compute log emission probability for a single observation x."""
    n_states = model.n_components
    log_emission = np.zeros(n_states)
    for i in range(n_states):
        try:
            mvn = multivariate_normal(mean=model.means_[i], cov=model.covars_[i])
            log_emission[i] = mvn.logpdf(x)
        except:
            log_emission[i] = -1e10
    return log_emission


def forward_step(log_alpha_prev, log_transmat, log_emission_today):
    """
    Incremental forward step: alpha_today = emission_today × (alpha_yesterday × transition)

    In log space:
    log_alpha_today[j] = log_emission_today[j] + logsumexp(log_alpha_prev + log_transmat[:, j])

    Returns normalized state probabilities and log_alpha for next step.
    """
    n_states = len(log_alpha_prev)
    log_alpha_today = np.zeros(n_states)

    for j in range(n_states):
        log_alpha_today[j] = log_emission_today[j] + np.logaddexp.reduce(
            log_alpha_prev + log_transmat[:, j]
        )

    # Normalize to get probabilities
    log_norm = np.logaddexp.reduce(log_alpha_today)
    state_probs = np.exp(log_alpha_today - log_norm)

    return state_probs, log_alpha_today


def initialize_forward(model, x_first):
    """Initialize forward algorithm with first observation."""
    log_startprob = np.log(model.startprob_ + 1e-300)
    log_emission = compute_log_emission(model, x_first)

    log_alpha = log_startprob + log_emission
    log_norm = np.logaddexp.reduce(log_alpha)
    state_probs = np.exp(log_alpha - log_norm)

    return state_probs, log_alpha


def identify_states(model, X_train, vix_zscore_idx):
    """
    Identify which HMM state corresponds to Stress/Transitional/Calm
    based on average VIX z-score.
    """
    train_states = model.predict(X_train)

    state_vix_means = {}
    for state in range(model.n_components):
        state_mask = train_states == state
        if state_mask.sum() > 0:
            state_vix_means[state] = X_train[state_mask, vix_zscore_idx].mean()
        else:
            state_vix_means[state] = 0

    # Sort by VIX mean: highest = Stress
    sorted_states = sorted(state_vix_means.items(), key=lambda x: x[1], reverse=True)
    stress_state = sorted_states[0][0]
    transitional_state = sorted_states[1][0]
    calm_state = sorted_states[2][0]

    return stress_state, transitional_state, calm_state


def fit_hmm(X_train):
    """Fit HMM model on training data."""
    model = hmm.GaussianHMM(
        n_components=N_STATES,
        covariance_type='full',
        n_iter=1000,
        random_state=42,
        tol=1e-4
    )
    model.fit(X_train)
    return model

# ============================================================
# Rolling Window HMM Estimation with Incremental Forward
# ============================================================

print("\nPerforming rolling window HMM estimation with incremental forward...")

all_data = zscore_df.values
all_dates = zscore_df.index
n_total = len(all_data)
vix_zscore_idx = zscore_df.columns.get_loc("VIX_Close_zscore")

# Initialize output arrays
p_calm_all = np.full(n_total, np.nan)
p_trans_all = np.full(n_total, np.nan)
p_stress_all = np.full(n_total, np.nan)
predicted_states_all = np.full(n_total, -1, dtype=int)
state_labels_all = [''] * n_total

# Track estimation points
estimation_dates = []
estimation_log_likelihoods = []

# Start from first point where we have enough data for HMM window
start_idx = HMM_WINDOW

print(f"\nFirst estimation at index {start_idx} ({all_dates[start_idx].date()})")
print(f"Total points to process: {n_total - start_idx}")

# Keep track of current model and states
current_model = None
current_stress_state = None
current_trans_state = None
current_calm_state = None
current_log_transmat = None
last_fit_idx = -REFIT_INTERVAL  # Ensure first fit happens

# Keep track of log_alpha for incremental forward
log_alpha_prev = None

# Store first fitted model for backfilling
first_model = None
first_stress_state = None
first_trans_state = None
first_calm_state = None

# Process each day
for t in range(start_idx, n_total):
    need_reinit_forward = False

    # Check if we need to refit the model
    if t - last_fit_idx >= REFIT_INTERVAL:
        # Fit HMM on window [t - HMM_WINDOW, t)
        window_start = t - HMM_WINDOW
        X_window = all_data[window_start:t]

        try:
            current_model = fit_hmm(X_window)
            current_stress_state, current_trans_state, current_calm_state = identify_states(
                current_model, X_window, vix_zscore_idx
            )
            current_log_transmat = np.log(current_model.transmat_ + 1e-300)

            log_likelihood = current_model.score(X_window)
            estimation_dates.append(all_dates[t])
            estimation_log_likelihoods.append(log_likelihood)

            last_fit_idx = t
            need_reinit_forward = True  # Need to reinitialize forward after model refit

            # Store first model for backfilling
            if first_model is None:
                first_model = current_model
                first_stress_state = current_stress_state
                first_trans_state = current_trans_state
                first_calm_state = current_calm_state
                print(f"  First model fitted at {all_dates[t].date()}: LL={log_likelihood:.2f}")

            if len(estimation_dates) % 10 == 1:
                print(f"  Fitted at {all_dates[t].date()}: LL={log_likelihood:.2f}, "
                      f"Stress={current_stress_state}, Trans={current_trans_state}, Calm={current_calm_state}")
        except Exception as e:
            print(f"  Warning: HMM fit failed at {all_dates[t].date()}: {e}")
            # Keep using previous model

    # Predict state probabilities using incremental forward
    if current_model is not None:
        try:
            x_today = all_data[t]

            if need_reinit_forward or log_alpha_prev is None:
                # Reinitialize forward algorithm after model refit
                # Run forward on the last few observations to get stable alpha
                warmup_len = min(50, t - (t - HMM_WINDOW))  # Use 50 days warmup
                warmup_start = t - warmup_len

                # Initialize with first warmup observation
                state_probs, log_alpha_prev = initialize_forward(current_model, all_data[warmup_start])

                # Run forward through warmup period
                for w in range(warmup_start + 1, t):
                    log_emission = compute_log_emission(current_model, all_data[w])
                    state_probs, log_alpha_prev = forward_step(log_alpha_prev, current_log_transmat, log_emission)

                # Now compute for today
                log_emission_today = compute_log_emission(current_model, x_today)
                state_probs, log_alpha_prev = forward_step(log_alpha_prev, current_log_transmat, log_emission_today)
            else:
                # Incremental forward: just one step
                # alpha_today = emission_today × (alpha_yesterday × transition)
                log_emission_today = compute_log_emission(current_model, x_today)
                state_probs, log_alpha_prev = forward_step(log_alpha_prev, current_log_transmat, log_emission_today)

            # Store results
            p_calm_all[t] = state_probs[current_calm_state]
            p_trans_all[t] = state_probs[current_trans_state]
            p_stress_all[t] = state_probs[current_stress_state]
            predicted_states_all[t] = np.argmax(state_probs)

            # Map to state label
            pred_state = np.argmax(state_probs)
            if pred_state == current_stress_state:
                state_labels_all[t] = 'Stress'
            elif pred_state == current_trans_state:
                state_labels_all[t] = 'Transitional'
            else:
                state_labels_all[t] = 'Calm'

        except Exception as e:
            print(f"  Warning: Prediction failed at {all_dates[t].date()}: {e}")
            log_alpha_prev = None  # Reset on error

print(f"\nTotal HMM estimations: {len(estimation_dates)}")
print(f"Estimation period: {estimation_dates[0].date()} to {estimation_dates[-1].date()}")

# ============================================================
# Backfill First 1200 Days Using First Fitted Model
# ============================================================

print("\n" + "="*60)
print("Backfilling first 1200 days using first fitted model...")
print("="*60)

if first_model is not None:
    # Use forward algorithm on first 1200 days with the first model
    first_log_transmat = np.log(first_model.transmat_ + 1e-300)

    # Initialize forward at t=0
    state_probs, log_alpha = initialize_forward(first_model, all_data[0])

    # Store first day's results
    p_calm_all[0] = state_probs[first_calm_state]
    p_trans_all[0] = state_probs[first_trans_state]
    p_stress_all[0] = state_probs[first_stress_state]
    predicted_states_all[0] = np.argmax(state_probs)

    pred_state = np.argmax(state_probs)
    if pred_state == first_stress_state:
        state_labels_all[0] = 'Stress'
    elif pred_state == first_trans_state:
        state_labels_all[0] = 'Transitional'
    else:
        state_labels_all[0] = 'Calm'

    # Forward through days 1 to start_idx-1
    for t in range(1, start_idx):
        log_emission = compute_log_emission(first_model, all_data[t])
        state_probs, log_alpha = forward_step(log_alpha, first_log_transmat, log_emission)

        p_calm_all[t] = state_probs[first_calm_state]
        p_trans_all[t] = state_probs[first_trans_state]
        p_stress_all[t] = state_probs[first_stress_state]
        predicted_states_all[t] = np.argmax(state_probs)

        pred_state = np.argmax(state_probs)
        if pred_state == first_stress_state:
            state_labels_all[t] = 'Stress'
        elif pred_state == first_trans_state:
            state_labels_all[t] = 'Transitional'
        else:
            state_labels_all[t] = 'Calm'

    print(f"Backfilled {start_idx} days (index 0 to {start_idx-1})")
    print(f"Backfill date range: {all_dates[0].date()} to {all_dates[start_idx-1].date()}")
else:
    print("Warning: No model available for backfilling!")

# ============================================================
# Data Split (on full data including backfilled period)
# ============================================================

# Now all data should be valid
valid_mask = ~np.isnan(p_stress_all)
valid_start_idx = 0  # Start from beginning since we backfilled
valid_end_idx = n_total - 1

n_valid = n_total
n_train = int(n_valid * 0.5)
n_val = int(n_valid * 0.3)
n_test = n_valid - n_train - n_val

train_end_idx = n_train - 1
val_end_idx = train_end_idx + n_val

print("\n" + "="*60)
print("Data Split (on full data including backfilled period)")
print("="*60)
print(f"Full data range: {all_dates[0].date()} to {all_dates[-1].date()}")
print(f"Total data points: {n_valid}")
print(f"Training: {all_dates[0].date()} to {all_dates[train_end_idx].date()} ({n_train} points, 50%)")
print(f"Validation: {all_dates[train_end_idx+1].date()} to {all_dates[val_end_idx].date()} ({n_val} points, 30%)")
print(f"Test: {all_dates[val_end_idx+1].date()} to {all_dates[-1].date()} ({n_test} points, 20%)")

# ============================================================
# Create Output DataFrame
# ============================================================

print("\nCreating output DataFrame...")

output_df = pd.DataFrame(index=all_dates)

# Original aligned data
output_df['SPX_VOL_20D'] = regime_df_aligned.loc[all_dates, 'SPX_VOL_20D']
output_df['VIX_Close'] = regime_df_aligned.loc[all_dates, 'VIX_Close']
output_df['CREDIT_FACTOR'] = regime_df_aligned.loc[all_dates, 'CREDIT_FACTOR']
output_df['RATE_SLOPE_10Y_3M'] = regime_df_aligned.loc[all_dates, 'RATE_SLOPE_10Y_3M']

# Z-scores
output_df['SPX_VOL_20D_zscore'] = zscore_df['SPX_VOL_20D_zscore']
output_df['VIX_Close_zscore'] = zscore_df['VIX_Close_zscore']
output_df['CREDIT_FACTOR_zscore'] = zscore_df['CREDIT_FACTOR_zscore']
output_df['RATE_SLOPE_10Y_3M_zscore'] = zscore_df['RATE_SLOPE_10Y_3M_zscore']

# State probabilities
output_df['P_Calm'] = p_calm_all
output_df['P_Transitional'] = p_trans_all
output_df['P_Stress'] = p_stress_all

# Delta P(Stress)
delta_p = np.diff(p_stress_all, prepend=np.nan)
output_df['Delta_P_Stress'] = delta_p

# Predicted state (numeric)
output_df['Predicted_State'] = predicted_states_all

# State label (string)
output_df['Regime_Label'] = state_labels_all

# Data split indicator (on full data including backfilled period)
split_labels = [''] * n_total
for i in range(n_total):
    if i <= train_end_idx:
        split_labels[i] = 'Train'
    elif i <= val_end_idx:
        split_labels[i] = 'Validation'
    else:
        split_labels[i] = 'Test'
output_df['Data_Split'] = split_labels

# Mark backfilled vs rolling estimation
backfill_flag = [''] * n_total
for i in range(start_idx):
    backfill_flag[i] = 'Backfilled'
for i in range(start_idx, n_total):
    backfill_flag[i] = 'Rolling'
output_df['Estimation_Type'] = backfill_flag

# No need to filter - all data is valid now
# (backfill ensures all days have predictions)

# ============================================================
# Statistics
# ============================================================

print("\n" + "="*60)
print("Regime Statistics")
print("="*60)

print("\n----- Overall -----")
regime_counts = output_df['Regime_Label'].value_counts()
for regime in ['Calm', 'Transitional', 'Stress']:
    count = regime_counts.get(regime, 0)
    pct = 100 * count / len(output_df)
    print(f"  {regime}: {count} days ({pct:.1f}%)")

print("\n----- By Estimation Type -----")
for est_type in ['Backfilled', 'Rolling']:
    est_data = output_df[output_df['Estimation_Type'] == est_type]
    if len(est_data) > 0:
        print(f"\n{est_type} ({len(est_data)} days):")
        est_counts = est_data['Regime_Label'].value_counts()
        for regime in ['Calm', 'Transitional', 'Stress']:
            count = est_counts.get(regime, 0)
            pct = 100 * count / len(est_data)
            print(f"  {regime}: {count} days ({pct:.1f}%)")

print("\n----- By Data Split -----")
for split in ['Train', 'Validation', 'Test']:
    split_data = output_df[output_df['Data_Split'] == split]
    if len(split_data) > 0:
        print(f"\n{split} ({len(split_data)} days):")
        split_counts = split_data['Regime_Label'].value_counts()
        for regime in ['Calm', 'Transitional', 'Stress']:
            count = split_counts.get(regime, 0)
            pct = 100 * count / len(split_data)
            print(f"  {regime}: {count} days ({pct:.1f}%)")

print("\n----- P(Stress) Statistics -----")
print(f"  Mean: {output_df['P_Stress'].mean():.4f}")
print(f"  Std: {output_df['P_Stress'].std():.4f}")
print(f"  Min: {output_df['P_Stress'].min():.4f}")
print(f"  Max: {output_df['P_Stress'].max():.4f}")

print("\n----- Delta P(Stress) Statistics -----")
valid_delta = output_df['Delta_P_Stress'].dropna()
print(f"  Mean: {valid_delta.mean():.6f}")
print(f"  Std: {valid_delta.std():.4f}")
print(f"  Min: {valid_delta.min():.4f}")
print(f"  Max: {valid_delta.max():.4f}")

# ============================================================
# Save to CSV
# ============================================================

print("\n" + "="*60)
print("Saving regime labels...")
print("="*60)

output_df.to_csv('regime_label.csv')

print(f"\nRegime labels saved to 'regime_label.csv'")
print(f"Columns: {list(output_df.columns)}")
print(f"Shape: {output_df.shape}")

# Show sample data
print("\nSample data (first 5 rows):")
print(output_df.head().to_string())

print("\nSample data (last 5 rows):")
print(output_df.tail().to_string())

# ============================================================
# Save Estimation History
# ============================================================

estimation_history = pd.DataFrame({
    'Date': estimation_dates,
    'Log_Likelihood': estimation_log_likelihoods
})
estimation_history.to_csv('hmm_estimation_history.csv', index=False)
print(f"\nHMM estimation history saved to 'hmm_estimation_history.csv' ({len(estimation_history)} estimations)")

# ============================================================
# Visualization
# ============================================================

import matplotlib.pyplot as plt

print("\nGenerating regime visualization...")

fig, axes = plt.subplots(5, 1, figsize=(16, 16), sharex=True)

# Color mapping for regimes
colors = {'Calm': 'green', 'Transitional': 'orange', 'Stress': 'red', '': 'gray'}

# Plot 1: VIX Close with regime coloring
ax1 = axes[0]
for regime in ['Calm', 'Transitional', 'Stress']:
    mask = output_df['Regime_Label'] == regime
    ax1.scatter(output_df.index[mask], output_df['VIX_Close'][mask], 
                c=colors[regime], label=regime, alpha=0.5, s=2)
ax1.set_ylabel('VIX Close', fontsize=10)
ax1.set_title('Market Regimes Identified by Rolling Window HMM (1200-day window, monthly refit)',
              fontsize=12, fontweight='bold')
ax1.legend(loc='upper right', fontsize=8)
ax1.grid(True, alpha=0.3)

# Plot 2: P(Stress)
ax2 = axes[1]
ax2.plot(output_df.index, output_df['P_Stress'], color='red', linewidth=0.8, alpha=0.8)
ax2.fill_between(output_df.index, 0, output_df['P_Stress'], color='red', alpha=0.3)
ax2.set_ylabel('P(Stress)', fontsize=10)
ax2.set_ylim(0, 1)
ax2.axhline(y=0.5, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
ax2.grid(True, alpha=0.3)

# Plot 3: Delta P(Stress)
ax3 = axes[2]
ax3.bar(output_df.index, output_df['Delta_P_Stress'], color='purple', alpha=0.6, width=1)
ax3.set_ylabel('ΔP(Stress)', fontsize=10)
ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax3.grid(True, alpha=0.3)

# Plot 4: SPX Volatility with regime coloring
ax4 = axes[3]
for regime in ['Calm', 'Transitional', 'Stress']:
    mask = output_df['Regime_Label'] == regime
    ax4.scatter(output_df.index[mask], output_df['SPX_VOL_20D'][mask], 
                c=colors[regime], label=regime, alpha=0.5, s=2)
ax4.set_ylabel('SPX Vol 20D', fontsize=10)
ax4.grid(True, alpha=0.3)

# Plot 5: HMM Estimation Points
ax5 = axes[4]
ax5.scatter(estimation_dates, estimation_log_likelihoods, color='blue', s=10, alpha=0.7)
ax5.set_ylabel('Log-Likelihood', fontsize=10)
ax5.set_xlabel('Date', fontsize=10)
ax5.set_title('HMM Re-estimation Points', fontsize=10)
ax5.grid(True, alpha=0.3)

# Add vertical lines for data split
train_data = output_df[output_df['Data_Split'] == 'Train']
val_data = output_df[output_df['Data_Split'] == 'Validation']
test_data = output_df[output_df['Data_Split'] == 'Test']

if len(train_data) > 0 and len(val_data) > 0:
    for ax in axes[:4]:
        ax.axvline(x=train_data.index[-1], color='blue', linestyle='--', linewidth=1, alpha=0.7)
        ax.axvline(x=val_data.index[-1], color='purple', linestyle='--', linewidth=1, alpha=0.7)

plt.tight_layout()
plt.savefig('regime_hmm.png', dpi=150, bbox_inches='tight')
plt.close()

print("Visualization saved as 'regime_hmm.png'")

print("\n" + "="*60)
print("Rolling Window HMM Estimation Complete!")
print("="*60)
print(f"- Estimation window: {HMM_WINDOW} days")
print(f"- Refit interval: {REFIT_INTERVAL} days (monthly)")
print(f"- Total estimations: {len(estimation_dates)}")
print(f"- Backfilled days: {start_idx} (using first fitted model)")
print(f"- Rolling estimation days: {n_total - start_idx}")
print(f"- Output range: {output_df.index[0].date()} to {output_df.index[-1].date()}")
print(f"- Total output points: {len(output_df)}")
