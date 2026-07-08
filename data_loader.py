import ccxt
import pandas as pd
import time

class DataLoader:
    def __init__(self, exchange_id='binance', symbol='BTC/USDT', timeframe='1h'):
        """
        Initialize the DataLoader with exchange and symbol details.
        
        Args:
            exchange_id (str): The ID of the exchange (e.g., 'binance', 'coinbase').
            symbol (str): The trading pair symbol (e.g., 'BTC/USDT').
            timeframe (str): The timeframe for OHLCV data (e.g., '1h', '1d').
        """
        try:
            self.exchange = getattr(ccxt, exchange_id)()
        except AttributeError:
            raise ValueError(f"Exchange '{exchange_id}' not found in ccxt.")
            
        self.symbol = symbol
        self.timeframe = timeframe

    def fetch_ohlcv(self, limit=100):
        """
        Fetch the most recent N OHLCV candles.
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=limit)
            if not ohlcv:
                return pd.DataFrame()
            return self._to_df(ohlcv)
        except Exception as e:
            print(f"Error fetching data for {self.symbol}: {e}")
            return pd.DataFrame()

    def fetch_ohlcv_range(self, start: str, end: str, batch_size=1000):
        """
        Fetch historical OHLCV data between two dates by paginating in batches.
        
        Args:
            start (str): Start date string, e.g. '2023-10-01'.
            end   (str): End date string, e.g.   '2024-03-01'.
            batch_size (int): Candles per API request (max 1000 on Binance).
            
        Returns:
            pd.DataFrame: Full OHLCV data for the date range.
        """
        since_ms = self.exchange.parse8601(f"{start}T00:00:00Z")
        end_ms   = self.exchange.parse8601(f"{end}T00:00:00Z")

        all_ohlcv = []
        print(f"  Fetching {self.symbol} [{start} → {end}] in batches of {batch_size}...")

        while since_ms < end_ms:
            try:
                batch = self.exchange.fetch_ohlcv(
                    self.symbol, self.timeframe,
                    since=since_ms, limit=batch_size
                )
                if not batch:
                    break

                all_ohlcv += batch
                since_ms = batch[-1][0] + 1   # advance past last candle
                print(f"  ...fetched {len(all_ohlcv)} candles so far", end='\r')
                time.sleep(self.exchange.rateLimit / 1000)  # respect rate limits

            except Exception as e:
                print(f"\nError during range fetch: {e}")
                break

        print(f"\n  ✅ Total candles fetched: {len(all_ohlcv)}")
        if not all_ohlcv:
            return pd.DataFrame()

        df = self._to_df(all_ohlcv)
        # Trim to exact requested end date
        df = df[df.index <= pd.Timestamp(end)]
        return df

    def _to_df(self, ohlcv: list) -> pd.DataFrame:
        """Convert raw list to a clean DataFrame."""
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[~df.index.duplicated(keep='first')]
        return df
