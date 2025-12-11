from __future__ import annotations
import plotly.graph_objects as go
from src.data.hyperliquid import OrderBookSnapshot
from typing import List, Tuple
import numpy as np

def _aggregate_levels_to_bins(
    levels: List[Tuple[float, float]],
    bin_edges: np.ndarray,
) -> np.ndarray:
    """
    Given (price, size) levels and fixed bin edges,
    return an array of total size per bin.
    """
    sizes = np.zeros(len(bin_edges) - 1, dtype=float)
    if not levels:
        return sizes

    prices = np.array([p for p, _ in levels], dtype=float)
    qtys = np.array([s for _, s in levels], dtype=float)

    # digitize gives indices of bins each price falls into
    idxs = np.digitize(prices, bin_edges) - 1  # bins are [edge_i, edge_{i+1})

    for i, q in zip(idxs, qtys):
        if 0 <= i < len(sizes):
            sizes[i] += q

    return sizes


def vertical_orderbook_rows(
    snapshot: OrderBookSnapshot,
    top_n: int = 10,
) -> Tuple[list[str], list[str], list[str]]:
    """
    For a single snapshot, build:

        - price_col: list of str
        - size_col: list of str
        - fill_colors: list of hex colors (one per row)

    Layout:
        Asks at top (highest ask at the very top),
        middle row = spread,
        bids at bottom.
    """
    # Sort asks/bids
    asks_sorted = sorted(snapshot.asks, key=lambda x: x[0])[:top_n]               # best ask first
    bids_sorted = sorted(snapshot.bids, key=lambda x: x[0], reverse=True)[:top_n] # best bid first

    # Compute spread + mid
    best_ask = asks_sorted[0][0] if asks_sorted else None
    best_bid = bids_sorted[0][0] if bids_sorted else None

    if best_ask is not None and best_bid is not None:
        spread = best_ask - best_bid
        mid = 0.5 * (best_ask + best_bid)
        spread_label = f"Spread: {spread:.4f} (Mid: {mid:.4f})"
    else:
        spread_label = ""

    # Asks: reverse so highest ask at very top, best ask closest to spread
    ask_prices = [f"{p:.4f}" for p, _ in reversed(asks_sorted)]
    ask_sizes  = [f"{s:.2f}"  for _, s in reversed(asks_sorted)]
    ask_rows   = list(zip(ask_prices, ask_sizes))

    # Spread row
    spread_row = [spread_label, ""]

    # Bids: best bid closest to spread, deeper bids lower
    bid_prices = [f"{p:.4f}" for p, _ in bids_sorted]
    bid_sizes  = [f"{s:.2f}"  for _, s in bids_sorted]
    bid_rows   = list(zip(bid_prices, bid_sizes))

    # Combine: asks • spread • bids
    final_rows = ask_rows + [spread_row] + bid_rows

    price_col = [r[0] for r in final_rows]
    size_col  = [r[1] for r in final_rows]

    n_asks = len(ask_rows)
    n_bids = len(bid_rows)

    ask_color    = "#330000"  # red-ish
    bid_color    = "#003300"  # green-ish
    spread_color = "#222222"

    fill_colors = (
        [ask_color]    * n_asks +
        [spread_color] * 1 +
        [bid_color]    * n_bids
    )

    return price_col, size_col, fill_colors


def make_orderbook_histogram(
    snapshots: List[OrderBookSnapshot],
    top_n: int = 20,
    max_snapshots: int = 200,
    downsample: bool = True,
    bin_size: float = 0.001,
) -> go.Figure:
    """
    Build an interactive order book depth viewer with a slider.

    - Displays bids vs asks as mirrored horizontal bars around x=0
      (bids negative, asks positive), with price on the y-axis.
    - Optionally downsamples snapshots for performance.
    """

    if not snapshots:
        raise ValueError("No snapshots provided")

    # --- downsample for performance ---
    if downsample and len(snapshots) > max_snapshots:
        step = max(1, len(snapshots) // max_snapshots)
        snapshots = snapshots[::step]

    all_prices = []
    all_sizes_raw = []

    for snap in snapshots:
        for p, s in snap.bids + snap.asks:
            all_prices.append(p)
            all_sizes_raw.append(abs(s))

    price_min = min(all_prices)
    price_max = max(all_prices)

    # Fixed price bins (global)
    bin_edges = np.arange(price_min, price_max + bin_size, bin_size)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    binned_data: List[Tuple[np.ndarray, np.ndarray]] = []
    global_max_abs_depth = 0.0

    for snap in snapshots:
        bid_sizes_raw = _aggregate_levels_to_bins(snap.bids, bin_edges)
        ask_sizes_raw = _aggregate_levels_to_bins(snap.asks, bin_edges)

        # Mirrored: bids negative, asks positive
        bid_sizes = -bid_sizes_raw
        ask_sizes = ask_sizes_raw

        binned_data.append((bid_sizes, ask_sizes))

        max_abs_for_snapshot = max(
            np.max(np.abs(bid_sizes)) if bid_sizes.size else 0.0,
            np.max(np.abs(ask_sizes)) if ask_sizes.size else 0.0,
        )
        if max_abs_for_snapshot > global_max_abs_depth:
            global_max_abs_depth = max_abs_for_snapshot

    price_padding = (price_max - price_min) * 0.05
    price_range = [price_min - price_padding, price_max + price_padding]

    x_range = [
        -global_max_abs_depth * 1.1,
        global_max_abs_depth * 1.1,
    ]

    bar_width = bin_size * 0.9  # so bins almost touch

    # Frames for animation
    frames = []

    for i, snap in enumerate(snapshots):
        # sort bids descending (best at top), asks ascending
        bids_sorted = sorted(snap.bids, key=lambda x: x[0], reverse=True)[:top_n]
        asks_sorted = sorted(snap.asks, key=lambda x: x[0])[:top_n]

        bid_prices = [p for p, _ in bids_sorted]
        bid_sizes = [-s for _, s in bids_sorted]  # negative for left side

        ask_prices = [p for p, _ in asks_sorted]
        ask_sizes = [s for _, s in asks_sorted]   # positive for right side

        frame = go.Frame(
            data=[
                go.Bar(
                    x=bid_sizes,
                    y=bid_prices,
                    orientation="h",
                    name="Bids",
                    opacity=1.0,
                    width=bar_width, 
                    marker=dict(line=dict(width=0))
                ),
                go.Bar(
                    x=ask_sizes,
                    y=ask_prices,
                    orientation="h",
                    name="Asks",
                    opacity=1.0,
                    width=bar_width,
                    marker=dict(line=dict(width=0))
                ),
            ],
            name=str(i),
            layout=go.Layout(
                title_text=f"Order book @ {snap.timestamp.isoformat()} (mid={snap.mid_price:.4f})"
            ),
        )
        frames.append(frame)

    # Initial frame
    init_frame = frames[0]

    # Slider steps
    slider_steps = []
    for i, snap in enumerate(snapshots):
        slider_steps.append(
            {
                "args": [[str(i)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
                "label": snap.timestamp.strftime("%H:%M:%S.%f")[:-3],  # show ms
                "method": "animate",
            }
        )

    sliders = [
        {
            "active": 0,
            "currentvalue": {"prefix": "Snapshot: "},
            "steps": slider_steps,
        }
    ]

    fig = go.Figure(
        data=init_frame.data,
        frames=frames,
        layout=go.Layout(
            barmode="relative",  # we rely on signed values, not stacking
            xaxis_title="Size (bids negative, asks positive)",
            yaxis_title="Price",
            sliders=sliders,
            updatemenus=[
                {
                    "type": "buttons",
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 80, "redraw": True},
                                    "fromcurrent": True,
                                },
                            ],
                        },
                        {
                            "label": "Pause",
                            "method": "animate",
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                },
                            ],
                        },
                    ],
                }
            ],
        ),
    )
    fig.update_xaxes(range=x_range)
    fig.update_yaxes(range=price_range)

    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="white")

    # Add a vertical line at x=0 like a spine
    fig.add_vline(x=0, line_width=1, line_dash="dash")

    return fig


def make_orderbook(
    snapshots: List[OrderBookSnapshot],
    top_n: int = 10,
    max_snapshots: int = 300,
    downsample: bool = True,
) -> go.Figure:
    """
    Build an interactive slider-based VERTICAL orderbook viewer.

    Layout per frame:
        - Asks at top (highest ask at very top)
        - Middle row shows spread + mid
        - Bids at bottom

    The slider steps through snapshots over time.
    """

    if not snapshots:
        raise ValueError("No snapshots provided")

    # Optional downsampling for performance
    if downsample and len(snapshots) > max_snapshots:
        step = max(1, len(snapshots) // max_snapshots)
        snapshots = snapshots[::step]

    frames = []

    for i, snap in enumerate(snapshots):
        price_col, size_col, fill_colors = vertical_orderbook_rows(snap, top_n=top_n)

        frames.append(
            go.Frame(
                data=[
                    go.Table(
                        header=dict(
                            values=["Price", "Size"],
                            fill_color="#111111",
                            font=dict(color="white", size=12),
                            align="center",
                        ),
                        cells=dict(
                            values=[price_col, size_col],
                            fill_color=[fill_colors, fill_colors],
                            align="center",
                            font=dict(color="white", size=11),
                        ),
                    )
                ],
                name=str(i),
                layout=go.Layout(
                    title_text=f"Vertical Order Book @ {snap.timestamp.isoformat()}"
                ),
            )
        )

    # Initial frame
    init_frame = frames[0]

    # Slider steps
    slider_steps = []
    for i, snap in enumerate(snapshots):
        slider_steps.append(
            {
                "args": [[str(i)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
                "label": snap.timestamp.strftime("%H:%M:%S"),
                "method": "animate",
            }
        )

    sliders = [
        {
            "active": 0,
            "currentvalue": {"prefix": "Snapshot: "},
            "steps": slider_steps,
        }
    ]

    fig = go.Figure(
        data=init_frame.data,
        frames=frames,
        layout=go.Layout(
            template="plotly_dark",
            sliders=sliders,
            updatemenus=[
                {
                    "type": "buttons",
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 80, "redraw": True},
                                    "fromcurrent": True,
                                },
                            ],
                        },
                        {
                            "label": "Pause",
                            "method": "animate",
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                },
                            ],
                        },
                    ],
                }
            ],
            margin=dict(l=10, r=10, t=40, b=10),
        ),
    )

    return fig
