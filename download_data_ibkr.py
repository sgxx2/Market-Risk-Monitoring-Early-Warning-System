"""
Download market data from Interactive Brokers.
This script downloads all data including FX pairs (EURUSD, USDJPY, USDCNH).

Output: data_ibkr.csv (combined file with MultiIndex columns)
"""

from ib_insync import *
import pandas as pd


# Contracts to download
CONTRACTS = {
    # ==================== ETFs ====================
    'SP500_ETF': {
        'symbol': 'SPY',
        'secType': 'STK',
        'exchange': 'SMART',
        'currency': 'USD',
        'description': 'S&P 500 ETF (SPY)'
    },
    'NASDAQ100_ETF': {
        'symbol': 'QQQ',
        'secType': 'STK',
        'exchange': 'SMART',
        'currency': 'USD',
        'description': 'NASDAQ 100 ETF (QQQ)'
    },
    'EUROSTOXX50_ETF': {
        'symbol': 'FEZ',
        'secType': 'STK',
        'exchange': 'SMART',
        'currency': 'USD',
        'description': 'EURO STOXX 50 ETF (FEZ)'
    },
    'NIKKEI225_ETF': {
        'symbol': 'EWJ',
        'secType': 'STK',
        'exchange': 'SMART',
        'currency': 'USD',
        'description': 'Japan ETF (EWJ, iShares MSCI Japan)'
    },
    'EEM': {
        'symbol': 'EEM',
        'secType': 'STK',
        'exchange': 'SMART',
        'currency': 'USD',
        'description': 'Emerging Markets ETF'
    },
    'IEF': {
        'symbol': 'IEF',
        'secType': 'STK',
        'exchange': 'SMART',
        'currency': 'USD',
        'description': '7-10 Year Treasury Bond ETF'
    },
    'HYG': {
        'symbol': 'HYG',
        'secType': 'STK',
        'exchange': 'SMART',
        'currency': 'USD',
        'description': 'High Yield Corporate Bond ETF'
    },
    'GSCI_ETF': {
        'symbol': 'GSG',
        'secType': 'STK',
        'exchange': 'SMART',
        'currency': 'USD',
        'description': 'S&P GSCI Commodity ETF (GSG)'
    },

    # ==================== Indices ====================
    'SP500': {
        'symbol': 'SPX',
        'secType': 'IND',
        'exchange': 'CBOE',
        'currency': 'USD',
        'description': 'S&P 500 Index'
    },
    'NASDAQ100': {
        'symbol': 'NDX',
        'secType': 'IND',
        'exchange': 'NASDAQ',
        'currency': 'USD',
        'description': 'NASDAQ 100 Index'
    },
    'EUROSTOXX50': {
        'symbol': 'ESTX50',
        'secType': 'IND',
        'exchange': 'EUREX',
        'currency': 'EUR',
        'description': 'EURO STOXX 50 Index'
    },
    'NIKKEI225': {
        'symbol': 'N225',
        'secType': 'IND',
        'exchange': 'OSE.JPN',
        'currency': 'JPY',
        'description': 'Nikkei 225 Index'
    },
    'VIX': {
        'symbol': 'VIX',
        'secType': 'IND',
        'exchange': 'CBOE',
        'currency': 'USD',
        'description': 'CBOE Volatility Index'
    },
    'DXY': {
        'symbol': 'DX',
        'secType': 'IND',
        'exchange': 'NYBOT',
        'currency': 'USD',
        'description': 'US Dollar Index'
    },

    # ==================== FX Pairs ====================
    'EURUSD': {
        'symbol': 'EUR',
        'secType': 'CASH',
        'exchange': 'IDEALPRO',
        'currency': 'USD',
        'description': 'EUR/USD Exchange Rate'
    },
    'USDJPY': {
        'symbol': 'USD',
        'secType': 'CASH',
        'exchange': 'IDEALPRO',
        'currency': 'JPY',
        'description': 'USD/JPY Exchange Rate'
    },
    'USDCNH': {
        'symbol': 'USD',
        'secType': 'CASH',
        'exchange': 'IDEALPRO',
        'currency': 'CNH',
        'description': 'USD/CNH Exchange Rate'
    },
}


def create_contract(name, contract_info):
    """Create IB contract object from contract info dictionary."""
    # For FX pairs, use Forex() class like fx_ibkr.py
    if contract_info['secType'] == 'CASH':
        # name is like 'EURUSD', 'USDJPY', 'USDCNH'
        return Forex(name)
    else:
        contract = Contract()
        contract.symbol = contract_info['symbol']
        contract.secType = contract_info['secType']
        contract.exchange = contract_info['exchange']
        contract.currency = contract_info['currency']
        return contract


def download_single_contract(ib, name, contract_info, duration='20 Y'):
    """
    Download historical data for a single contract.

    Parameters:
    -----------
    ib : IB
        Connected IB instance
    name : str
        Contract name/identifier
    contract_info : dict
        Contract specification
    duration : str
        Data duration

    Returns:
    --------
    pd.DataFrame or None
    """
    print(f"\n{'='*60}")
    print(f"Downloading {name} ({contract_info['description']})...")
    print(f"{'='*60}")

    contract = create_contract(name, contract_info)

    # Select whatToShow parameter based on contract type
    if contract_info['secType'] == 'CASH':
        what_to_show = 'MIDPOINT'
    elif contract_info['secType'] == 'IND':
        what_to_show = 'TRADES'
    else:
        what_to_show = 'TRADES'

    try:
        bars = ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr=duration,
            barSizeSetting='1 day',
            whatToShow=what_to_show,
            useRTH=False,
            formatDate=1
        )

        if not bars:
            print(f"  Warning: No data received for {name}")
            return None

        df = util.df(bars)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

        # Handle columns based on contract type
        # FX (CASH) data doesn't have volume
        if contract_info['secType'] == 'CASH':
            # FX data: only OHLC, no volume
            available_cols = []
            col_mapping = {}
            for col in ['open', 'high', 'low', 'close']:
                if col in df.columns:
                    available_cols.append(col)
                    col_mapping[col] = col.capitalize()
            df = df[available_cols]
            df.columns = [col_mapping[c] for c in available_cols]
            # Add empty Volume column for consistency
            df['Volume'] = 0
        else:
            # Standard data with OHLCV
            df = df[['open', 'high', 'low', 'close', 'volume']]
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

        print(f"  Date range: {df.index.min()} to {df.index.max()}")
        print(f"  Total {len(df)} records")

        return df

    except Exception as e:
        print(f"  Error: Failed to download {name} - {e}")
        return None


def download_all_data(
    host='127.0.0.1',
    port=7497,
    client_id=1,
    duration='20 Y',
    symbols=None,
    output_file='data_ibkr.csv'
):
    """
    Download historical data for all specified contracts and save to one combined CSV file.

    Parameters:
    -----------
    host : str
        TWS host address
    port : int
        TWS port (TWS: 7497, IB Gateway: 4001)
    client_id : int
        Client ID
    duration : str
        Data duration (e.g., '20 Y', '10 Y', '5 Y', '1 Y')
    symbols : list or None
        List of contracts to download, None means download all
    output_file : str
        Output CSV filename for combined data

    Returns:
    --------
    pd.DataFrame : Combined DataFrame with MultiIndex columns
    """
    ib = IB()
    results = {}

    try:
        print(f"Connecting to TWS ({host}:{port})...")
        ib.connect(host, port, clientId=client_id)
        print("Connected successfully!\n")

        # Determine contracts to download
        if symbols is None:
            contracts_to_download = CONTRACTS
        else:
            contracts_to_download = {k: v for k, v in CONTRACTS.items() if k in symbols}

        total = len(contracts_to_download)
        for idx, (name, contract_info) in enumerate(contracts_to_download.items(), 1):
            print(f"\nProgress: {idx}/{total}")

            df = download_single_contract(ib, name, contract_info, duration)
            if df is not None:
                results[name] = df

            # Avoid requesting too fast
            if idx < total:
                print("  Waiting 2 seconds...")
                ib.sleep(2)

        print(f"\n{'='*60}")
        print("Download completed!")
        print(f"{'='*60}")
        print(f"Successfully downloaded: {len(results)}/{total} contracts")

        if results:
            print("\nSuccessfully downloaded contracts:")
            for name in results.keys():
                print(f"  - {name}")

        failed = set(contracts_to_download.keys()) - set(results.keys())
        if failed:
            print("\nFailed to download contracts:")
            for name in failed:
                print(f"  - {name}")

        # Combine all DataFrames into one with MultiIndex columns
        if results:
            print(f"\n{'='*60}")
            print("Combining all data into one file...")
            print(f"{'='*60}")

            combined_dfs = []
            for name, df in results.items():
                # Create MultiIndex columns same as fx_ibkr.py
                df_copy = df.copy()
                # Remove duplicate index entries
                df_copy = df_copy[~df_copy.index.duplicated(keep='first')]
                # df columns are: ['Open', 'High', 'Low', 'Close', 'Volume']
                df_copy.columns = pd.MultiIndex.from_product([df_copy.columns, [name]])
                # Swap levels to get (Symbol, Price_Type) format
                df_copy.columns = df_copy.columns.swaplevel(0, 1)
                combined_dfs.append(df_copy)
                print(f"  {name}: {len(df_copy)} rows")

            if combined_dfs:
                # Use outer join to combine all data
                combined_df = combined_dfs[0]
                for df in combined_dfs[1:]:
                    combined_df = combined_df.join(df, how='outer')

                # Sort columns
                combined_df = combined_df.sort_index(axis=1)
                # Sort by date
                combined_df = combined_df.sort_index()
                combined_df.to_csv(output_file)

                print(f"\nCombined data saved to: {output_file}")
                print(f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")
                print(f"Total {len(combined_df)} rows")
                print(f"Symbols: {list(combined_df.columns.get_level_values(0).unique())}")

                return combined_df

        return None

    except Exception as e:
        print(f"Error: {e}")
        print("\nPlease check:")
        print("1. TWS or IB Gateway is running and logged in")
        print("2. API settings: TWS/IB Gateway -> Configure -> API -> Enable ActiveX and Socket Clients")
        print("3. Port is correct (TWS: 7497, IB Gateway: 4001)")
        return None

    finally:
        if ib.isConnected():
            ib.disconnect()
            print("\nTWS connection disconnected.")


def list_contracts():
    """Print list of available contracts."""
    print("\nAvailable contracts:")
    print("=" * 70)

    # Group by type
    etfs = {k: v for k, v in CONTRACTS.items() if v['secType'] == 'STK'}
    indices = {k: v for k, v in CONTRACTS.items() if v['secType'] == 'IND'}
    fx_pairs = {k: v for k, v in CONTRACTS.items() if v['secType'] == 'CASH'}

    print("\n[ETFs]")
    print("-" * 70)
    for name, info in etfs.items():
        print(f"  {name:20} - {info['description']}")

    print("\n[Indices]")
    print("-" * 70)
    for name, info in indices.items():
        print(f"  {name:20} - {info['description']}")

    print("\n[FX Pairs]")
    print("-" * 70)
    for name, info in fx_pairs.items():
        print(f"  {name:20} - {info['description']}")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1].lower() == 'list':
        list_contracts()
    else:
        # Download all data including FX pairs
        print("="*60)
        print("IBKR Data Downloader")
        print("="*60)
        print("\nThis script downloads all market data including:")
        print("  - ETFs: SPY, QQQ, FEZ, EWJ, EEM, IEF, HYG, GSG")
        print("  - Indices: SPX, NDX, ESTX50, N225, VIX, DXY")
        print("  - FX Pairs: EURUSD, USDJPY, USDCNH\n")

        results = download_all_data(
            port=7497,              # TWS port, use 4001 for IB Gateway
            duration='20 Y',        # Download 20 years of data
            symbols=None,           # None means download all
            output_file='data_ibkr.csv'
        )

