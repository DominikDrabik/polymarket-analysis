import pandas as pd
import numpy as np
import statsmodels.api as sm

def prepare_long_dataframe(df_prices):
    """
    Converts a wide-format price DataFrame to a long-format DataFrame.

    Args:
        df_prices (pd.DataFrame): Wide-format DataFrame with a DatetimeIndex
            and columns for each market's price.

    Returns:
        pd.DataFrame: A long-format DataFrame with columns:
            - 'time': Timestamp of the observation.
            - 'market': Identifier of the market.
            - 'p_t': Price at time t.
            - 'i': Snapshot index within each market's time series.
            - 'p_prev': Price at the previous time step.
            The resulting DataFrame is ready for time-series belief analysis.
    """
    df_long = (
        df_prices
        .reset_index()
        .melt(id_vars="time", var_name="market", value_name="p_t")
        .dropna(subset=["p_t"])
    )
    df_long = df_long.sort_values(["market", "time"]).reset_index(drop=True)
    df_long["p_prev"] = df_long.groupby("market")["p_t"].shift(1)
    df_long = df_long.dropna(subset=["p_prev"])
    
    # Re-calculate snapshot index `i` after dropping rows
    df_long["i"] = df_long.groupby("market").cumcount() + 1
    return df_long

def compute_belief_update(p0, p1):
    """
    Computes the belief update between two price points.

    The update is defined as the fraction of remaining belief resolved (if increasing),
    or the fraction of prior belief lost (if decreasing). This normalization makes
    belief changes comparable across time and markets, and reflects the conditional
    probability of an event occurring in the current step.

    Args:
        p0 (float): The previous price (0 to 1).
        p1 (float): The current price (0 to 1).

    Returns:
        float: The belief update b_i, a value in [0, 1].
    """
    if p1 > p0:
        # Normalize by the potential gain
        return (p1 - p0) / (1 - p0) if (1 - p0) != 0 else 0.0
    else:
        # Normalize by the potential loss
        return (p0 - p1) / p0 if p0 != 0 else 0.0

# def calculate_belief_metrics(df_long):
#     """
#     Calculates belief updates and their cumulative spread for each market.
#
#     Args:
#         df_long (pd.DataFrame): A long-format DataFrame from `prepare_long_dataframe`.
#
#     Returns:
#         pd.DataFrame: The input DataFrame with added columns: 'b_i' (belief update),
#             'cum_max_b', 'cum_min_b', and 'delta_b' (cumulative spread).
#     """
#     df_long["b_i"] = df_long.apply(lambda r: compute_belief_update(r.p_prev, r.p_t), axis=1)
#     df_long["cum_max_b"] = df_long.groupby("market")["b_i"].cummax()
#     df_long["cum_min_b"] = df_long.groupby("market")["b_i"].cummin()
#     df_long["delta_b"] = df_long["cum_max_b"] - df_long["cum_min_b"]
#     return df_long

def calculate_rolling_belief_spread(df_long, window_size_str):
    """
    Calculates the belief update `b_i` and its spread over a rolling time window.

    This first computes the belief update `b_i` for each snapshot. Then, it
    captures recent volatility by finding the difference between the min
    and max belief updates within a time-based rolling window specified by
    `window_size_str`.

    Args:
        df_long (pd.DataFrame): DataFrame from `prepare_long_dataframe`.
        window_size_str (str): The size of the rolling window as a pandas
            time offset string (e.g., '1D', '12H').

    Returns:
        pd.DataFrame: The input DataFrame with added columns: 'b_i', 'b_roll_max',
            'b_roll_min', and 'delta_b_roll' (the rolling spread).
    """
    df_long["b_i"] = df_long.apply(lambda r: compute_belief_update(r.p_prev, r.p_t), axis=1)

    # For time-based rolling, we need a DatetimeIndex.
    df_with_time_index = df_long.set_index('time')

    # Use transform with a lambda to apply rolling on a time window for each market.
    # The lambda function receives a Series with a DatetimeIndex.
    b_roll_max = (
        df_with_time_index
        .groupby("market")["b_i"]
        .transform(lambda x: x.rolling(window_size_str, min_periods=1).max())
    )
    b_roll_min = (
        df_with_time_index
        .groupby("market")["b_i"]
        .transform(lambda x: x.rolling(window_size_str, min_periods=1).min())
    )

    # Assign the results back. Using .values is safe because transform preserves order.
    df_long["b_roll_max"] = b_roll_max.values
    df_long["b_roll_min"] = b_roll_min.values

    df_long["delta_b_roll"] = df_long["b_roll_max"] - df_long["b_roll_min"]
    return df_long

def get_first_nonzero_spread(df_long):
    """
    Finds the first snapshot index `i` where market volatility begins.

    For each market, this identifies the first moment the rolling belief spread
    becomes greater than zero.

    Args:
        df_long (pd.DataFrame): DataFrame with 'market', 'i', and 'delta_b_roll'.

    Returns:
        dict: A map of {market_name: first_nonzero_index}.
    """
    return {
        market: int(grp.loc[grp["delta_b_roll"] > 0, "i"].min())
        if (grp["delta_b_roll"] > 0).any()
        else None
        for market, grp in df_long.groupby("market")
    }

def calculate_threshold_hits(df_long, first_nonzero, thresholds):
    """
    Calculates when the rolling spread first drops below given thresholds.

    For each market, this finds the snapshot index `I_delta` at which the
    rolling belief spread falls to or below each specified `threshold`,
    signaling a return to consensus.

    Args:
        df_long (pd.DataFrame): DataFrame with convergence data.
        first_nonzero (dict): A map from `get_first_nonzero_spread`.
        thresholds (list[float]): A list of thresholds to check (e.g., [0.02, 0.01]).

    Returns:
        pd.DataFrame: A DataFrame detailing the `I_delta` and `time_I_delta`
            for each market-threshold pair.
    """
    thresh_rows = []
    for market, grp in df_long.groupby("market"):
        i0 = first_nonzero.get(market)
        for δ in thresholds:
            if i0 is None:
                i_hit, t_hit = None, None
            else:
                mask = (grp["i"] >= i0) & (grp["delta_b_roll"] <= δ)
                if mask.any():
                    hit_row = grp.loc[mask].iloc[0]
                    i_hit = int(hit_row["i"])
                    t_hit = hit_row["time"]
                else:
                    i_hit, t_hit = None, None
            thresh_rows.append({
                "market": market, "threshold": δ,
                "I_delta": i_hit, "time_I_delta": t_hit
            })
    return pd.DataFrame(thresh_rows)

def calculate_half_life(df_long, first_nonzero):
    """
    Calculates the half-life of convergence for the belief spread.

    For each market, it measures the initial spread value when volatility begins
    and finds the snapshot index `i_half` where the spread first drops
    to 50% of that initial value.

    Args:
        df_long (pd.DataFrame): DataFrame with convergence data.
        first_nonzero (dict): A map from `get_first_nonzero_spread`.

    Returns:
        pd.DataFrame: A DataFrame detailing the half-life metrics for each market.
    """
    half_rows = []
    for market, grp in df_long.groupby("market"):
        i0 = first_nonzero.get(market)
        if i0 is None:
            half_rows.append({"market": market, "initial_delta": None, "half_value": None, "i_half": None, "time_half": None})
            continue

        initial_spread = grp.loc[grp["i"] == i0, "delta_b_roll"].iloc[0]
        half_val = 0.5 * initial_spread
        mask = (grp["i"] >= i0) & (grp["delta_b_roll"] <= half_val)

        if mask.any():
            hit_row = grp.loc[mask].iloc[0]
            i_half = int(hit_row["i"])
            t_half = hit_row["time"]
        else:
            i_half, t_half = None, None

        half_rows.append({
            "market": market,
            "initial_delta": initial_spread,
            "half_value": half_val,
            "i_half": i_half,
            "time_half": t_half
        })
    return pd.DataFrame(half_rows)

def calculate_decay_rate(df_long, first_nonzero):
    """
    Estimates the exponential decay rate (lambda) of the belief spread.

    For each market, this performs a log-linear regression of the rolling
    belief spread against the snapshot index `i` to find how quickly the
    spread converges exponentially.

    Args:
        df_long (pd.DataFrame): DataFrame with convergence data.
        first_nonzero (dict): A map from `get_first_nonzero_spread`.

    Returns:
        pd.DataFrame: A DataFrame with the market, decay rate `lambda`, and `r2`
            of the regression fit.
    """
    decay_rows = []
    for market, grp in df_long.groupby("market"):
        i0 = first_nonzero.get(market)
        if i0 is None:
            decay_rows.append({"market": market, "lambda": None, "r2": None})
            continue

        sub = grp.loc[(grp["i"] >= i0) & (grp["delta_b_roll"] > 0)].copy()
        if len(sub) < 5:  # Need sufficient data for a meaningful regression
            decay_rows.append({"market": market, "lambda": None, "r2": None})
            continue

        sub["i_shifted"] = sub["i"] - i0
        X = sm.add_constant(sub["i_shifted"])
        Y = np.log(sub["delta_b_roll"])
        model = sm.OLS(Y, X).fit()
        
        λ = -model.params.get("i_shifted", np.nan)
        decay_rows.append({"market": market, "lambda": λ, "r2": model.rsquared})
    return pd.DataFrame(decay_rows)

def calculate_auc(df_long, first_nonzero):
    """
    Calculates the Area Under the Curve (AUC) for the rolling belief spread.

    This integrates the `delta_b_roll` with respect to the snapshot index `i`
    to quantify the total amount of disagreement/volatility over time for
    each market. A smaller AUC implies quicker convergence.

    Args:
        df_long (pd.DataFrame): DataFrame with convergence data.
        first_nonzero (dict): A map from `get_first_nonzero_spread`.

    Returns:
        pd.DataFrame: A DataFrame with the market and its `AUC`, sorted by AUC.
    """
    auc_rows = []
    for market, grp in df_long.groupby("market"):
        i0 = first_nonzero.get(market)
        sub = grp[grp["i"] >= i0] if i0 is not None else pd.DataFrame()
        
        if len(sub) < 2:
            auc = None
        else:
            # Use np.trapz for numerical integration.
            auc = np.trapz(y=sub["delta_b_roll"], x=sub["i"])
            
        auc_rows.append({"market": market, "AUC": auc})
    return pd.DataFrame(auc_rows).sort_values("AUC")
