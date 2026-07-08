from abc import ABC, abstractmethod
import pandas as pd

class BaseStrategy(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> dict:
        """
        Analyze the provided OHLCV data and generate a trading signal.

        Args:
            data (pd.DataFrame): OHLCV DataFrame indexed by timestamp.

        Returns:
            dict with keys:
                'action'            : 'BUY', 'SELL', or 'HOLD'
                'price'             : float, current price
                'predicted_price'   : float, estimated price at next interval
                'predicted_change_pct': float, expected % change
                'reason'            : str, human-readable explanation
                'metadata'          : dict, strategy-specific extra data
        """
        pass
