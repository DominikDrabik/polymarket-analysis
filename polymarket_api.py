import requests
import json
from urllib.parse import urlparse
import pandas as pd

# --------------------------------------------------------------------------
# Low-Level API Wrappers & Helpers
# --------------------------------------------------------------------------

def extract_slug_from_event_url(event_url: str) -> str:
    """
    Given a Polymarket "/event/..." URL (possibly with query parameters), return just the slug part.
    Example: "https://polymarket.com/event/presidential-election-winner-2024?tid=abc123"
             → "presidential-election-winner-2024"
    Raises ValueError if the URL path does not look like /event/<slug>.
    """
    parsed = urlparse(event_url)
    parts = parsed.path.strip("/").split("/")
    if len(parts) < 2 or parts[0] != "event":
        raise ValueError(f"URL does not look like a Polymarket /event/ URL: {event_url!r}")
    return parts[1]

def fetch_markets_from_slug(slug: str) -> list:
    """
    Fetch the event data from Polymarket's Gamma API using the slug, then return the list of markets.
    """
    api_url = f"https://gamma-api.polymarket.com/events?slug={slug}"
    resp = requests.get(api_url, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if not data:
        raise RuntimeError(f"No event found for slug '{slug}'")

    first_event = data[0]
    return first_event.get("markets", [])

def get_prices(market_id, fidelity):
    """
    Fetches price history for a given market token ID.
    """
    url = "https://clob.polymarket.com/prices-history"
    params = {
      "interval": "all",
      "market": market_id,
      "fidelity": fidelity
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data

def get_trades(market_id):
    """
    NOTE: This endpoint (data-api.polymarket.com) appears to return the latest trades
    across ALL markets, not for a specific market_id. The official, supported endpoint for
    fetching trades by market is on clob.polymarket.com and requires authentication.
    See: https://docs.polymarket.com/developers/CLOB/trades/trades

    Returns a list of recent trades.
    Each trade is a dict with keys like: ['proxyWallet', 'side', 'asset', 'conditionId', 'size', 'price', 'timestamp', 'title', ...].
    """
    # The 'market' parameter in this URL doesn't seem to filter by market.
    url = f"https://data-api.polymarket.com/trades?market={market_id}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    payload = resp.json()
    return payload

def fetch_market_details(event_url: str) -> dict:
    """
    Given a full Polymarket event URL, returns a dict mapping the market question to its full market data object.

    The volume can be accessed via the 'volume' or 'volumeNum' key in the market data dict.
    Example:
    details = fetch_market_details(event_url)
    for question, market_data in details.items():
        print(f"{question}: {market_data['volumeNum']}")
    """
    slug = extract_slug_from_event_url(event_url)
    markets = fetch_markets_from_slug(slug)

    mapping = {}
    for market in markets:
        question = market.get("question", "<no question>")
        mapping[question] = market

    return mapping

# --------------------------------------------------------------------------
# Higher-Level Functions
# --------------------------------------------------------------------------

def fetch_market_token_mapping(event_url: str) -> dict:
    """
    Given a full Polymarket event URL, returns a dict mapping the market question to its first clobTokenId.
    """
    slug = extract_slug_from_event_url(event_url)
    markets = fetch_markets_from_slug(slug)

    mapping = {}
    for market in markets:
        question = market.get("question", "<no question>")
        raw_clob_ids = market.get("clobTokenIds", "[]")
        try:
            clob_list = json.loads(raw_clob_ids)
        except json.JSONDecodeError:
            clob_list = []

        first_token = clob_list[0] if clob_list else None
        if first_token:
            mapping[question] = first_token
        else:
            print(f"Warning: no valid clobTokenIds for market question: {question!r}. Skipping.")

    return mapping

def fetch_and_merge_price_histories(token_mapping: dict, fidelity: int = 720):
    """
    Fetches and merges price histories for multiple markets into a single DataFrame, 
    aligning timestamps with an inner join to include only overlapping data points.

    Args:
        token_mapping (dict): Maps market labels (str) to clobTokenIds (str).
            e.g., {'Biden to win': '0x...', 'Trump to win': '0x...'}
        fidelity (int, optional): Fidelity for the price history API. Defaults to 720.

    Returns:
        pd.DataFrame: A DataFrame with a DatetimeIndex, and columns for each
            market's price, labeled from the token_mapping.
    """
    dfs = []
    for label, token_id in token_mapping.items():
        price_data = get_prices(token_id, fidelity)
        hist = price_data.get("history", [])
        if not hist:
            print(f"Warning: no history for {label!r}, skipping.")
            continue

        df = pd.DataFrame(hist)
        df["time"] = pd.to_datetime(df["t"], unit="s")
        df = df.set_index("time").rename(columns={"p": label}).drop(columns=["t"])
        dfs.append(df)

    if not dfs:
        raise RuntimeError("No valid market DataFrames were created.")

    df_merged = dfs[0]
    for df in dfs[1:]:
        df_merged = df_merged.join(df, how="inner")

    if df_merged.empty:
        raise RuntimeError("No overlapping timestamps between markets.")
    return df_merged
