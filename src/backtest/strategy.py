from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd

@dataclass 
class StrategyConfig: 
    name: str 
    params: Dict[str, Any] | None = None

class BaseStrategy(ABC): 
    """
    Given a price DataFrame, outputs target portfolio **weights** in the 
    asset between -1 and 1. 
    The backtest engine will then use these weights to simulate trades.
    """
    def __init__(self, config:StrategyConfig): 
        self.config = config

    @abstractmethod
    def generate_weights(self, data: pd.DataFrame) -> pd.Series: 
        """
        Given a DataFrame of market data (at least a 'mid' column),
        return a pd.Series of target weights (e.g., 0.0 to 1.0) indexed by timestamp.

        For a single-asset strategy:
        - weight = 1.0  -> 100% of equity in the asset (long-only)
        - weight = 0.0  -> all in cash
        - weight = -1.0 -> 100% short (if you later support shorts)
        """
        raise NotImplementedError("generate_weights must be implemented by subclasses")
    
class BuyAndHoldStrategy(BaseStrategy):
    """
    Simplest possible strategy:
    - Always target weight = 1.0 (fully invested)
    - The backtester will invest at the first timestamp and then hold.
    """
    def __init__(self): 
        super().__init__(StrategyConfig(name="BuyAndHold", params={}))

    def generate_weights(self, data: pd.DataFrame) -> pd.Series:
        # Just return a Series of 1.0 weights for the entire period (fully invested)
        weights = pd.Series(1.0, index=data.index, name="target_weight")
        return weights
    
