import pandas as pd

# ============================================================
# Read data from Yahoo and IBKR
# ============================================================

# Read data_yahoo.csv with MultiIndex columns (first two rows are headers)
prices = pd.read_csv('data_yahoo.csv', header=[0, 1], index_col=0, parse_dates=True)
prices.index = pd.to_datetime(prices.index)
print("data_yahoo.csv loaded:")
print(f"  Shape: {prices.shape}")
print(f"  Date range: {prices.index.min()} to {prices.index.max()}")
print(f"  Symbols: {list(prices.columns.get_level_values(1).unique())}")

# Read data_ibkr.csv with MultiIndex columns
# Note: IBKR format is (Symbol, Price_Type), Yahoo format is (Price_Type, Symbol)
ibkr_data = None
try:
    ibkr_data = pd.read_csv('data_ibkr.csv', header=[0, 1], index_col=0, parse_dates=True)
    ibkr_data.index = pd.to_datetime(ibkr_data.index)
    # IBKR format: level 0 = Symbol, level 1 = Price_Type
    print(f"\ndata_ibkr.csv loaded:")
    print(f"  Shape: {ibkr_data.shape}")
    print(f"  Date range: {ibkr_data.index.min()} to {ibkr_data.index.max()}")
    print(f"  Symbols (level 0): {list(ibkr_data.columns.get_level_values(0).unique())}")
except Exception as e:
    print(f"\nError loading data_ibkr.csv: {e}")

# ============================================================
# Merge IBKR data with Yahoo data
# IBKR data quality is better but shorter timespan
# Priority: Use IBKR data first, fill missing with Yahoo data
# ============================================================

print("\n" + "="*60)
print("Merging data: IBKR (priority) + Yahoo (fill missing)")
print("="*60)

if ibkr_data is not None:
    # Only merge FX data from IBKR (EURUSD, USDJPY, USDCNH)
    # IBKR format: level 0 = Symbol, level 1 = Price_Type
    fx_symbols = ['EURUSD', 'USDJPY', 'USDCNH']
    all_ibkr_symbols = list(ibkr_data.columns.get_level_values(0).unique())
    ibkr_symbols = [s for s in fx_symbols if s in all_ibkr_symbols]
    excluded_symbols = [s for s in all_ibkr_symbols if s not in fx_symbols]

    print(f"\nIBKR FX symbols to merge: {ibkr_symbols}")
    if excluded_symbols:
        print(f"Excluded IBKR symbols (non-FX): {excluded_symbols}")

    for symbol in ibkr_symbols:
        print(f"\n--- Processing {symbol} ---")

        # Get IBKR data for this symbol (level 0 = symbol)
        ibkr_cols = [col for col in ibkr_data.columns if col[0] == symbol]
        if not ibkr_cols:
            continue

        ibkr_sym = ibkr_data[ibkr_cols].copy()
        # Convert IBKR format (Symbol, Price_Type) to Yahoo format (Price_Type, Symbol)
        ibkr_sym.columns = pd.MultiIndex.from_tuples(
            [(col[1], col[0]) for col in ibkr_sym.columns]
        )
        ibkr_sym = ibkr_sym[~ibkr_sym.index.duplicated(keep='first')]
        ibkr_count = ibkr_sym.dropna(how='all').shape[0]
        print(f"  IBKR {symbol}: {ibkr_count} valid rows")

        if ibkr_count == 0:
            continue

        ibkr_valid = ibkr_sym.dropna(how='all')
        print(f"  IBKR date range: {ibkr_valid.index.min()} to {ibkr_valid.index.max()}")

        # Check if Yahoo data exists for this symbol
        yahoo_cols = [col for col in prices.columns if col[1] == symbol]

        if yahoo_cols:
            yahoo_sym = prices[yahoo_cols].copy()
            yahoo_sym = yahoo_sym[~yahoo_sym.index.duplicated(keep='first')]
            yahoo_count = yahoo_sym.dropna(how='all').shape[0]
            print(f"  Yahoo {symbol}: {yahoo_count} valid rows")

            # Merge: IBKR takes priority, Yahoo fills missing
            all_dates = yahoo_sym.index.union(ibkr_sym.index)
            yahoo_aligned = yahoo_sym.reindex(all_dates)
            ibkr_aligned = ibkr_sym.reindex(all_dates)

            merged = ibkr_aligned.combine_first(yahoo_aligned)

            # Count sources
            ibkr_dates = ibkr_valid.index
            merged_valid = merged.dropna(how='all')
            yahoo_only_dates = merged_valid.index.difference(ibkr_dates)
            print(f"  Merged: {len(merged_valid)} total rows")
            print(f"    - From IBKR: {len(ibkr_dates)} rows")
            print(f"    - From Yahoo (fill): {len(yahoo_only_dates)} rows")

            # Update prices DataFrame
            for col in merged.columns:
                prices[col] = merged[col]
        else:
            # No Yahoo data, just add IBKR data
            print(f"  Yahoo {symbol} not found, using IBKR data only")
            for col in ibkr_sym.columns:
                prices[col] = ibkr_sym[col]

# ============================================================
# Also check for USDCNH in usdcnh_daily.csv if not in IBKR
# ============================================================

usdcnh_cols = [col for col in prices.columns if col[1] == 'USDCNH']
if not usdcnh_cols:
    print("\n--- Processing USDCNH from usdcnh_daily.csv ---")
    try:
        usdcnh_raw = pd.read_csv('usdcnh_daily.csv', index_col='date', parse_dates=True)
        usdcnh_df = pd.DataFrame(index=usdcnh_raw.index)
        for price_type in ['Close', 'High', 'Low', 'Open', 'Volume']:
            if price_type in usdcnh_raw.columns:
                usdcnh_df[(price_type, 'USDCNH')] = usdcnh_raw[price_type]
        usdcnh_df = usdcnh_df[~usdcnh_df.index.duplicated(keep='first')]
        print(f"  USDCNH from usdcnh_daily.csv: {usdcnh_df.dropna(how='all').shape[0]} valid rows")

        for col in usdcnh_df.columns:
            prices[col] = usdcnh_df[col]
    except Exception as e:
        print(f"  Error loading usdcnh_daily.csv: {e}")

# ============================================================
# Sort and save
# ============================================================

# Sort columns
prices = prices.sort_index(axis=1)

# Sort by date
prices = prices.sort_index()

# Save to combine_data.csv
prices.to_csv('combine_data.csv')

print("\n" + "="*60)
print("Merged data saved to combine_data.csv")
print("="*60)
print(f"Total columns: {len(prices.columns)}")
print(f"Total rows: {len(prices)}")
print(f"Date range: {prices.index.min()} to {prices.index.max()}")

print("\nSymbols in combined data:")
symbols = sorted(set([col[1] for col in prices.columns]))
for sym in symbols:
    sym_data = prices[[col for col in prices.columns if col[1] == sym]]
    valid_count = sym_data.dropna(how='all').shape[0]
    first_date = sym_data.dropna(how='all').index.min() if valid_count > 0 else 'N/A'
    last_date = sym_data.dropna(how='all').index.max() if valid_count > 0 else 'N/A'
    print(f"  {sym}: {valid_count} rows ({first_date} to {last_date})")
