from dataclasses import dataclass
from typing import Dict, Any

import pandas as pd

from src.backtest.strategy import BaseStrategy

@dataclass
class BacktestConfig:
    """
    Configuration for the backtest.
    """
    initial_capital: float = 1_000.0

@dataclass
class BacktestResult: 
    """
    Container for backtest results.
    """
    equity_curve: pd.DataFrame
    summary: Dict[str, Any]

class BacktestEngine: 
    def __init__(self, 
                prices: pd.DataFrame, 
                strategy: BaseStrategy,
                config: BacktestConfig | None = None, 
        ):
        if "mid" not in prices.columns:
            raise ValueError("Prices DataFrame must contain a 'mid' column.")
        if prices.empty:
            raise ValueError("Prices DataFrame is empty.")
            
        self.data = prices.sort_index().copy()
        self.strategy = strategy
        self.config = config or BacktestConfig()
    
    def run(self) -> BacktestResult: 
        # Generate weights from strategy 
        weights = self.strategy.generate_weights(self.data)
        if not weights.index.equals(self.data.index):
            raise ValueError("Weights index must match prices index.")
        
        # Iterate through ticks and simulate portfolio 
        equity_curve = self.data[["mid"]].copy()
        equity_curve["target_weight"] = weights
        equity_curve["equity"] = 0.0

        # Initialize portfolio
        equity = self.config.initial_capital
        cash = self.config.initial_capital
        position = 0.0 

        equity_list = []
        cash_list = []
        qty_list = []
        position_value_list = []
        returns_list = [] 

        prev_equity = equity

        for ts, row in equity_curve.iterrows(): 
            price = float(row["mid"])
            target_weight = float(row["target_weight"])

            # Current position market value 
            position_value = position * price
            equity = cash + position_value

            # Target position value and quantity 
            target_position_value = equity * target_weight
            target_qty = target_position_value / price if price > 0 else 0.0

            # Trade quantity is the difference 
            trade_qty = target_qty - position

            # Update cash after trading (assume 0 fees for now)
            cash -= trade_qty * price

            # New Position quantity 
            position = target_qty

            # Recompute position value and equity 
            position_value = position * price
            equity = cash + position_value

            # Simple return 
            ret = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0.0
            prev_equity = equity

            # Store time series metrics 
            equity_list.append(equity)
            cash_list.append(cash)
            qty_list.append(position)
            position_value_list.append(position_value)
            returns_list.append(ret)

        # write computed series back into equity_curve
        equity_curve["equity"] = equity_list
        equity_curve["cash"] = cash_list
        equity_curve["position_qty"] = qty_list
        equity_curve["position_value"] = position_value_list
        equity_curve["returns"] = returns_list

        # Simple summary statistics
        total_pnl = equity_curve["equity"].iloc[-1] - self.config.initial_capital
        total_return = equity_curve["equity"].iloc[-1] / self.config.initial_capital - 1.0
        max_equity = equity_curve["equity"].cummax()
        drawdown = equity_curve["equity"] / max_equity - 1.0
        max_drawdown = drawdown.min()

        summary = {
        "initial_capital": self.config.initial_capital,
        "final_equity": equity_curve["equity"].iloc[-1],
        "total_pnl": total_pnl,
        "total_return": total_return,
        "max_drawdown": float(max_drawdown),
        "strategy_name": self.strategy.config.name,
        }

        return BacktestResult(equity_curve=equity_curve, summary=summary)
    
