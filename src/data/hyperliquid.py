from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd

@dataclass
class OrderBookSnapshot: 
    timestamp: pd.Timestamp
    bids: List[Tuple[float, float]]  # List of (price, size
    asks: List[Tuple[float, float]]  # List of (price, size)
    mid_price: float


def compute_orderbook_features(
    snapshots: List[OrderBookSnapshot],
) -> pd.DataFrame:
    """
    Given a list of OrderBookSnapshot, compute simple orderbook features per tick.

    For now:
      - total_bid_volume = sum of sizes on bid side
      - total_ask_volume = sum of sizes on ask side
      - imbalance = (bid - ask) / (bid + ask), safe when denom = 0

    Returns a DataFrame indexed by timestamp with columns:
      ['bid_volume', 'ask_volume', 'imbalance']
    """
    records = []

    for snap in snapshots:
        bid_vol = sum(size for _, size in snap.bids)
        ask_vol = sum(size for _, size in snap.asks)

        denom = bid_vol + ask_vol
        if denom > 0:
            imbalance = (bid_vol - ask_vol) / denom
        else:
            imbalance = 0.0  

        records.append(
            {
                "timestamp": snap.timestamp,
                "bid_volume": bid_vol,
                "ask_volume": ask_vol,
                "imbalance": imbalance,
            }
        )

    df = pd.DataFrame(records).set_index("timestamp").sort_index()
    return df


def parse_orderbook_line(line: str) -> OrderBookSnapshot: 
    """
    Parse one Hyperliquid l2Book JSON line into an OrderBookSnapshot.
    """
    obj = json.loads(line)
    data = obj["raw"]["data"]

    ts_ms = data["time"]
    timestamp = pd.to_datetime(ts_ms, unit="ms", utc=True)

    raw_bids = data["levels"][0]
    raw_asks = data["levels"][1]
    bids = [(float(level["px"]), float(level["sz"])) for level in raw_bids]
    asks = [(float(level["px"]), float(level["sz"])) for level in raw_asks]

    # best bid/ask for mid
    best_bid = max(p for p, _ in bids)
    best_ask = min(p for p, _ in asks)
    mid = 0.5 * (best_bid + best_ask)

    return OrderBookSnapshot(timestamp=timestamp, bids=bids, asks=asks, mid_price=mid)

def load_orderbook_and_mid_series(file_path: str | Path, 
                                  include_features: bool = False
) -> tuple[pd.DataFrame, List[OrderBookSnapshot]] | tuple[pd.DataFrame, List[OrderBookSnapshot], pd.DataFrame]:
    """
    Read a JSONL file of Hyperliquid l2Book snapshots and return:
    If include_features == False (default):
        - mid_df:    DataFrame indexed by timestamp with 'mid'
        - snapshots: list[OrderBookSnapshot]

    If include_features == True:
        - full_df:   DataFrame with ['mid', 'bid_volume', 'ask_volume', 'imbalance']
        - snapshots: list[OrderBookSnapshot]
        - features:  DataFrame with ['bid_volume', 'ask_volume', 'imbalance']
    """

    jsonl_path = Path(file_path)
    timestamps: List[pd.Timestamp] = []
    mids: List[float] = []
    snapshots: List[OrderBookSnapshot] = []

    with jsonl_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            snap = parse_orderbook_line(line)
            snapshots.append(snap)
            timestamps.append(snap.timestamp)
            mids.append(snap.mid_price)

    mid_df = pd.DataFrame({"timestamp": timestamps, "mid": mids})
    mid_df = mid_df.set_index("timestamp").sort_index()

    # keep snapshots in the same sorted order
    snapshots_sorted = sorted(snapshots, key=lambda s: s.timestamp)
    if not include_features:
        return mid_df, snapshots_sorted

    features_df = compute_orderbook_features(snapshots_sorted)
    full_df = mid_df.join(features_df, how="inner")
    return full_df, snapshots_sorted, features_df