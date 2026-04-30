import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

print('Generating Recurrence Plots with 60-day window...')

features = pd.read_csv('features.csv', index_col=0, parse_dates=True)
recurrence_features = ['SP500_logret', 'VIX_diff', 'CREDIT_FACTOR', 'DXY_logret']

def compute_recurrence_matrix(series):
    data = series.dropna().values
    n = len(data)
    distance_matrix = np.abs(data.reshape(-1, 1) - data.reshape(1, -1))
    # Use 0.5 * std as threshold
    epsilon = 0.5 * np.std(data)
    recurrence_matrix = (distance_matrix <= epsilon).astype(int)
    return recurrence_matrix, epsilon

fig_rp, axes_rp = plt.subplots(2, 2, figsize=(14, 14))
axes_rp = axes_rp.flatten()
rp_colors = ['Blues', 'Reds', 'Greens', 'Purples']

for i, feature in enumerate(recurrence_features):
    ax = axes_rp[i]

    if feature not in features.columns:
        ax.text(0.5, 0.5, f'{feature}\nNot Found', ha='center', va='center', fontsize=14)
        ax.set_title(f'{feature} - Not Available')
        continue

    series = features[feature].dropna()
    print(f'{feature}: {len(series)} total points')

    if len(series) == 0:
        continue

    # Use 60-day time window
    window_days = 60
    if len(series) > window_days:
        series = series.tail(window_days)

    print(f'  Using last {len(series)} days: {series.index[0].strftime("%Y-%m-%d")} to {series.index[-1].strftime("%Y-%m-%d")}')

    rec_matrix, epsilon = compute_recurrence_matrix(series)

    im = ax.imshow(rec_matrix, cmap=rp_colors[i], origin='lower', aspect='equal')
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Recurrence', fontsize=9)

    n = len(rec_matrix)
    rr = (np.sum(rec_matrix) - n) / (n * (n - 1))

    diag_sum = 0
    for k in range(2, n):
        diag = np.diag(rec_matrix, k)
        diag_str = ''.join(map(str, diag))
        for run in diag_str.split('0'):
            if len(run) >= 2:
                diag_sum += len(run)
    det = diag_sum / (np.sum(rec_matrix) - n) if (np.sum(rec_matrix) - n) > 0 else 0

    ax.set_xlabel('Time Index', fontsize=10)
    ax.set_ylabel('Time Index', fontsize=10)
    ax.set_title(f'{feature}\n(n={len(series)}, ε={epsilon:.4f})', fontsize=11, fontweight='bold')

    ax.annotate(f'RR={rr:.3f}\nDET={det:.3f}',
                xy=(0.02, 0.98), xycoords='axes fraction',
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    date_range = f'{series.index[0].strftime("%Y-%m-%d")} to {series.index[-1].strftime("%Y-%m-%d")}'
    ax.annotate(date_range,
                xy=(0.98, 0.02), xycoords='axes fraction',
                fontsize=8, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

fig_rp.suptitle('Recurrence Plots for Key Financial Features\n(60-Day Time Window, ε = 0.5 × std)',
                fontsize=14, fontweight='bold', y=0.95)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('recurrence_plots.png', dpi=150, bbox_inches='tight')
plt.close()

print('\nFigure saved as recurrence_plots.png')

