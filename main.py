import pandas as pd
import polymarket_api as api
import analysis
import plotting
import numpy as np
import warnings

# --- Configuration ---
EVENT_URL = "https://polymarket.com/event/poland-presidential-election"
FIDELITY = 720  # Lower for more granular data, higher for less
ROLLING_WINDOW = '3D'
THRESHOLDS = [0.02, 0.01]
# Use None to select all markets in the event
# Example: COLUMN_NAMES_TO_PICK = ['Will Donald Trump win the 2024 US Presidential Election?']
COLUMN_NAMES_TO_PICK = None 

def main():
    """
    Main function to run the Polymarket analysis pipeline.
    """
    # 1. Fetch data from Polymarket APIs
    print(f"Fetching markets for event: {EVENT_URL}")
    token_mapping = api.fetch_market_token_mapping(event_url=EVENT_URL)
    
    if not token_mapping:
        print("No markets with valid token IDs found. Exiting.")
        return

    print("Fetching and merging price histories...")
    df_prices = api.fetch_and_merge_price_histories(token_mapping=token_mapping, fidelity=FIDELITY)
    
    if COLUMN_NAMES_TO_PICK:
        df_prices = df_prices[COLUMN_NAMES_TO_PICK]

    # 2. Perform analysis
    print("Preparing data for analysis...")
    df_long = analysis.prepare_long_dataframe(df_prices)
    
    print("Calculating rolling belief spread...")
    df_long = analysis.calculate_rolling_belief_spread(df_long, window_size_str=ROLLING_WINDOW)

    print("Calculating convergence metrics...")
    first_nonzero = analysis.get_first_nonzero_spread(df_long)
    df_thresh = analysis.calculate_threshold_hits(df_long, first_nonzero, THRESHOLDS)
    df_half = analysis.calculate_half_life(df_long, first_nonzero)
    df_decay = analysis.calculate_decay_rate(df_long, first_nonzero)
    df_auc = analysis.calculate_auc(df_long, first_nonzero)

    # 3. Display results
    print("\n=== Threshold Hits ===")
    print(df_thresh)

    print("\n=== Half-Life ===")
    print(df_half)

    print("\n=== Decay Rates ===")
    print(df_decay)

    print("\n=== Area Under Curve (AUC) ===")
    print(df_auc)
    
    # Merge for a summary view
    df_summary = (
        df_thresh.query(f"threshold == {THRESHOLDS[0]}")[["market", "I_delta", "time_I_delta"]]
        .merge(df_half[["market", "i_half", "time_half"]], on="market")
        .merge(df_decay, on="market")
        .merge(df_auc, on="market")
    )
    print("\n=== Summary Table ===")
    print(df_summary)

    # 4. Generate plots
    print("\nGenerating plots...")
    plotting.plot_price_histories(df_prices)
    plotting.plot_rolling_belief_spread(df_long, window=ROLLING_WINDOW)
    plotting.plot_annotated_rolling_belief_spread(df_long, ROLLING_WINDOW, df_thresh, df_half, df_decay)

if __name__ == "__main__":
    # The 'trapz' function used in analysis.py is deprecated. This suppresses the warning.
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    main() 