from __future__ import annotations
from typing import List
import plotly.graph_objects as go
from src.data.hyperliquid import OrderBookSnapshot


def make_orderbook_slider(
    snapshots: List[OrderBookSnapshot],
    top_n: int = 20,
    max_snapshots: int = 200,
    downsample: bool = True,
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
                    width=0.01, 
                    marker=dict(line=dict(width=0))
                ),
                go.Bar(
                    x=ask_sizes,
                    y=ask_prices,
                    orientation="h",
                    name="Asks",
                    opacity=1.0,
                    width=0.01,
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

    # Add a vertical line at x=0 like a spine
    fig.add_vline(x=0, line_width=1, line_dash="dash")

    return fig
