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

def load_orderbook_and_mid_series(file_path: str | Path) -> tuple[pd.DataFrame, List[OrderBookSnapshot]]:
    """
    Read a JSONL file of Hyperliquid l2Book snapshots and return:
      - mid_df: DataFrame indexed by timestamp with a single 'mid' column
      - snapshots: list[OrderBookSnapshot]
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

    return mid_df, snapshots_sorted

