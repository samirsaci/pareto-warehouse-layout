"""
Pareto Analysis for Warehouse Layout Optimization
"""

import pandas as pd
import matplotlib.pyplot as plt


def load_data():
    """Load order data from file."""
    df = pd.read_csv('data/1-2017.csv', sep='\t')
    # Rename columns to match notebook convention
    df.columns = ['DATE FORMAT', 'ORDER_NUMBER', 'SKU', 'BOX']
    return df


def create_sample_data():
    """Create sample data for demonstration."""
    import numpy as np
    np.random.seed(42)

    n_orders = 5000
    n_lines = 15000

    skus = [f"SKU_{i}" for i in range(500)]
    sku_weights = np.random.pareto(1.5, 500)
    sku_weights = sku_weights / sku_weights.sum()

    data = []
    for i in range(n_lines):
        data.append({
            'DATE': '2017-01-01',
            'FORMAT': 'A',
            'ORDER_NUMBER': np.random.randint(1, n_orders),
            'SKU': np.random.choice(skus, p=sku_weights),
            'BOX': np.random.randint(1, 5)
        })
    return pd.DataFrame(data)


def perform_pareto_analysis(df):
    """Perform Pareto analysis on order data."""
    # BOX/SKU
    df_par = pd.DataFrame(df.groupby(['SKU'])['BOX'].sum())
    df_par.columns = ['BOX']

    # Sort Values
    df_par.sort_values(['BOX'], ascending=False, inplace=True)
    df_par.reset_index(inplace=True)

    # Cumulative Sum
    df_par['CumSum'] = df_par['BOX'].cumsum()

    # % CumSum
    df_par['%CumSum'] = (100 * df_par['CumSum'] / df_par['BOX'].sum())

    # % SKU
    df_par['%SKU'] = (100 * (df_par.index + 1).astype(float) / (df_par.index.max() + 1))

    return df_par


def calculate_thresholds(df_par):
    """Calculate key Pareto thresholds."""
    # > 80% Volume
    df_par80 = df_par[df_par['%CumSum'] > 80].copy()
    perc_sku80 = df_par80['%SKU'].min()
    perc_sum80 = df_par80['%CumSum'].min()

    # 20% SKU
    df_sku20 = df_par[df_par['%SKU'] > 20].copy()
    perc_sku20 = df_sku20['%SKU'].min()
    perc_sum20 = df_sku20['%CumSum'].min()

    # 5% SKU
    df_sku5 = df_par[df_par['%SKU'] > 5].copy()
    perc_sku5 = df_sku5['%SKU'].min()
    perc_sum5 = df_sku5['%CumSum'].min()

    return {
        'perc_sku80': perc_sku80, 'perc_sum80': perc_sum80,
        'perc_sku20': perc_sku20, 'perc_sum20': perc_sum20,
        'perc_sku5': perc_sku5, 'perc_sum5': perc_sum5
    }


def plot_pareto(df_par, thresholds):
    """Create and save Pareto chart."""
    fig, ax = plt.subplots(figsize=(20, 7.5))
    ax.plot(df_par['%SKU'], df_par['%CumSum'])
    plt.xlabel('Percentage of SKU (%)', fontsize=15)
    plt.ylabel('Percentage of Boxes Ordered (%)', fontsize=15)
    plt.title('Pareto Analysis using Cumulative Sum of Boxes Prepared (%) = f(%SKU)', fontsize=15)

    # 5% SKU
    ax.axhline(thresholds['perc_sum5'], color="black", linestyle="--", linewidth=1.0)
    ax.axvline(thresholds['perc_sku5'], color="black", linestyle="--", linewidth=1.0)

    # 80% Volume
    ax.axhline(thresholds['perc_sum80'], color="red", linestyle="--", linewidth=1.0)
    ax.axvline(thresholds['perc_sku80'], color="red", linestyle="--", linewidth=1.0)

    # 20% SKU
    ax.axhline(thresholds['perc_sum20'], color="blue", linestyle="--", linewidth=1.0)
    ax.axvline(thresholds['perc_sku20'], color="blue", linestyle="--", linewidth=1.0)

    plt.tight_layout()
    plt.savefig('pareto_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: pareto_analysis.png")


def display_results(df, df_par, thresholds):
    """Display analysis results."""
    print("=" * 60)
    print("PARETO ANALYSIS - WAREHOUSE LAYOUT OPTIMIZATION")
    print("=" * 60)

    print(f"\n--- DATA SUMMARY ---")
    print(f"Order Lines: {len(df):,}")
    print(f"Unique Orders: {df['ORDER_NUMBER'].nunique():,}")
    print(f"Unique SKUs: {len(df_par):,}")
    print(f"Total Boxes: {df_par['BOX'].sum():,.0f}")

    print(f"\n--- PARETO ANALYSIS ---")
    print(f"5% of SKUs represent {thresholds['perc_sum5']:.1f}% of volume")
    print(f"20% of SKUs represent {thresholds['perc_sum20']:.1f}% of volume")
    print(f"80% of volume comes from {thresholds['perc_sku80']:.1f}% of SKUs")

    print(f"\n--- TOP 10 SKUs ---")
    print(df_par.head(10).to_string(index=False))

    print(f"\n--- WAREHOUSE LAYOUT RECOMMENDATION ---")
    n_fast = int(len(df_par) * thresholds['perc_sku80'] / 100)
    print(f"Fast-moving zone (A): {n_fast} SKUs ({thresholds['perc_sku80']:.1f}%)")
    print(f"  -> Place closest to packing area")
    n_medium = int(len(df_par) * 0.15)
    print(f"Medium-moving zone (B): ~{n_medium} SKUs")
    print(f"  -> Standard picking locations")
    n_slow = len(df_par) - n_fast - n_medium
    print(f"Slow-moving zone (C): ~{n_slow} SKUs")
    print(f"  -> Reserve storage area")


def main():
    """Main function for Pareto analysis."""
    # Load data
    try:
        df = load_data()
        print("Data loaded from file.")
    except FileNotFoundError:
        print("Data file not found. Using sample data.")
        df = create_sample_data()

    print(f"{len(df):,} order lines for {df['ORDER_NUMBER'].nunique():,} orders")

    # Perform analysis
    df_par = perform_pareto_analysis(df)
    print(f"Pareto Analysis for {len(df_par):,} unique SKU")

    # Calculate thresholds
    thresholds = calculate_thresholds(df_par)

    # Create visualization
    plot_pareto(df_par, thresholds)

    # Display results
    display_results(df, df_par, thresholds)


if __name__ == "__main__":
    main()
