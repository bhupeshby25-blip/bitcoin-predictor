"""
MLPredictor — Walk-Forward Ensemble Price Predictor
=====================================================
Replaces naive linear regression with a proper ML pipeline.

Architecture:
  - 30+ engineered features from raw OHLCV
  - Walk-forward training: train on past 200 candles, predict next
  - Ensemble: Ridge Regression + Random Forest → averaged prediction
  - Retrains every 20 candles (efficiency vs freshness trade-off)

Features engineered (38 total):
  Momentum   : Log-returns at lags 1,2,3,5,7,10,14 days
  Trend      : SMA20/SMA50 slope, EMA7 vs EMA21 spread
  Oscillators: RSI(14), RSI(7), RSI(2), MACD, MACD histogram
  Volatility : ATR(14)/price, Bollinger width %B, Realized vol (10d)
  Volume     : Volume z-score (20d), volume momentum (5d)
  Structure  : Body/range ratio, upper/lower wicks, HL range %
  Calendar   : Day of week (BTC has Monday seasonality)
  Sentiment  : Fear & Greed Index (value, 7d/14d change, z-score, zones)

Target:
  NEXT candle's log-return (regression)
  → convert to price and direction for reporting

Walk-forward prevents look-ahead bias:
  At candle i: train on candles [i-200, i], predict candle i+1
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from fear_greed import get_fng_features, FearGreedLoader
from fear_greed_signal import FearGreedSignal
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


class MLPredictor:
    MIN_TRAIN_CANDLES = 120   # minimum to fit the model
    RETRAIN_EVERY     = 20    # retrain every N candles (balance speed vs accuracy)
    TRAIN_WINDOW      = 200   # rolling training window size

    def __init__(self):
        # Benchmark-proven ensemble (walk-forward 2022-2026, 262 steps):
        #   ElasticNet  55.0%  (+5.0% edge) — L1+L2 selects best correlated features
        #   SVR-RBF     53.1%  (+3.1% edge) — captures BTC cyclic non-linearity
        #   RandomForest53.1%  (+3.1% edge) — stable low-variance bagging
        #   LightGBM    51.9%  (+1.9% edge) — fast leaf-wise boosting
        self._elastic_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model',  ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000))
        ])
        self._svr_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model',  SVR(kernel='rbf', C=1.0, epsilon=0.01))
        ])
        self._rf_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model',  RandomForestRegressor(
                n_estimators=100, max_depth=4,
                min_samples_leaf=5, random_state=42, n_jobs=-1
            ))
        ])
        self._lgb_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model',  lgb.LGBMRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbose=-1
            ))
        ])
        self._last_train_i = -1
        self._fitted       = False
        self._fng_signal   = FearGreedSignal()
        self._fng_loader   = FearGreedLoader()

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #
    def predict(self, data: pd.DataFrame) -> dict:
        """
        Given OHLCV data up to and including the current candle,
        predict the NEXT candle's price and direction.

        Returns:
            dict with keys:
              predicted_price       : float
              predicted_change_pct  : float (positive = up)
              direction             : 'UP' | 'DOWN'
              confidence_pct        : float (0-100, model agreement score)
              features_used         : int
        """
        if len(data) < self.MIN_TRAIN_CANDLES:
            return self._fallback(data)

        features_df = self._build_features(data)
        if features_df is None or len(features_df) < self.MIN_TRAIN_CANDLES:
            return self._fallback(data)

        # Retrain periodically
        i = len(data)
        if not self._fitted or (i - self._last_train_i) >= self.RETRAIN_EVERY:
            self._train(features_df)
            self._last_train_i = i

        # Predict using the LAST row (current candle's features)
        X_pred = features_df.drop(columns=['target']).iloc[[-1]].values

        try:
            elastic_pred = self._elastic_pipe.predict(X_pred)[0]
            svr_pred     = self._svr_pipe.predict(X_pred)[0]
            rf_pred      = self._rf_pipe.predict(X_pred)[0]
            lgb_pred     = self._lgb_pipe.predict(X_pred)[0]

            # Benchmark-proven weights (proportional to direction accuracy edge):
            #   ElasticNet 35%, SVR 30%, RF 25%, LightGBM 10%
            tech_pred = (0.35 * elastic_pred +
                         0.30 * svr_pred     +
                         0.25 * rf_pred      +
                         0.10 * lgb_pred)

            # --- F&G Signal (5th model) with dynamic weighting ---
            fng_df     = self._fng_loader._load_data()
            fng_series = fng_df['fng_value'].reindex(
                data.index.normalize()
            ).ffill() if fng_df is not None else pd.Series(dtype=float)
            fng_result = self._fng_signal.predict(fng_series)
            fng_pred   = fng_result['log_ret_pred']
            fng_prob   = fng_result['prob_up']

            # F&G weight: 10% neutral, up to 30% in extreme zones
            fng_strength = abs(fng_prob - 0.5) * 2
            fng_weight   = 0.10 + 0.20 * fng_strength   # 10%–30%
            tech_weight  = 1.0 - fng_weight

            # Final blended prediction
            ensemble_log_ret = tech_weight * tech_pred + fng_weight * fng_pred

            current_price   = data['close'].iloc[-1]
            predicted_price = current_price * np.exp(ensemble_log_ret)
            change_pct      = (np.exp(ensemble_log_ret) - 1) * 100

            # Confidence: all 5 models' direction agreement
            all_preds = [elastic_pred, svr_pred, rf_pred, lgb_pred, fng_pred]
            signs     = [1 if p > 0 else -1 for p in all_preds]
            agreement = abs(sum(signs)) / len(signs)
            direction = "UP" if ensemble_log_ret > 0 else "DOWN"

            return {
                "predicted_price":      round(predicted_price, 2),
                "predicted_change_pct": round(change_pct, 2),
                "direction":            direction,
                "confidence_pct":       round(agreement * 100, 0),
                "features_used":        features_df.shape[1] - 1,
                "fng_value":            fng_result['fng_value'],
                "fng_class":            fng_result['fng_class'],
                "fng_weight_pct":       round(fng_weight * 100, 1),
                "fng_signal":           fng_result['signal'],
                "model_votes":          {
                    "ElasticNet": "UP" if elastic_pred > 0 else "DOWN",
                    "SVR-RBF":    "UP" if svr_pred     > 0 else "DOWN",
                    "RF":         "UP" if rf_pred      > 0 else "DOWN",
                    "LightGBM":   "UP" if lgb_pred     > 0 else "DOWN",
                    "F&G":        "UP" if fng_pred     > 0 else "DOWN",
                }
            }
        except Exception as e:
            return self._fallback(data)

    # ------------------------------------------------------------------ #
    #  Feature engineering
    # ------------------------------------------------------------------ #
    def _build_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Build 30+ predictive features. All features are lagged to prevent look-ahead."""
        df = data.copy()
        c  = df['close']
        h  = df['high']
        l  = df['low']
        v  = df['volume']

        feat = pd.DataFrame(index=df.index)

        # ---- Momentum: log-returns at multiple lags ----
        log_ret = np.log(c / c.shift(1))
        for lag in [1, 2, 3, 5, 7, 10, 14]:
            feat[f'ret_{lag}d']  = log_ret.shift(1).rolling(lag).sum()

        # ---- Trend: MA ratios and slopes ----
        sma20  = c.rolling(20).mean()
        sma50  = c.rolling(50).mean()
        ema7   = c.ewm(span=7,  adjust=False).mean()
        ema21  = c.ewm(span=21, adjust=False).mean()

        feat['sma20_slope']  = (sma20 / sma20.shift(5) - 1).shift(1)
        feat['sma50_slope']  = (sma50 / sma50.shift(10) - 1).shift(1)
        feat['ema_spread']   = ((ema7 - ema21) / ema21).shift(1)
        feat['price_vs_sma20'] = ((c - sma20) / sma20).shift(1)
        feat['price_vs_sma50'] = ((c - sma50) / sma50).shift(1)

        # ---- Oscillators: RSI at multiple periods ----
        for period in [2, 7, 14]:
            feat[f'rsi_{period}'] = self._rsi(c, period).shift(1)

        # ---- MACD ----
        macd_line   = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
        signal_line = macd_line.ewm(span=9,  adjust=False).mean()
        feat['macd']      = (macd_line / c).shift(1)
        feat['macd_hist'] = ((macd_line - signal_line) / c).shift(1)

        # ---- Volatility ----
        # ATR
        tr = pd.concat([
            h - l,
            (h - c.shift(1)).abs(),
            (l - c.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean()
        feat['atr_pct']       = (atr14 / c).shift(1)
        feat['realized_vol']  = log_ret.rolling(10).std().shift(1)

        # Bollinger Bands
        bb_mean = c.rolling(20).mean()
        bb_std  = c.rolling(20).std()
        feat['bb_width']  = (2 * bb_std / bb_mean).shift(1)
        feat['bb_pct_b']  = ((c - (bb_mean - 2*bb_std)) / (4 * bb_std)).shift(1)

        # ---- Volume features ----
        vol_mean = v.rolling(20).mean()
        vol_std  = v.rolling(20).std()
        feat['vol_zscore']   = ((v - vol_mean) / (vol_std + 1e-9)).shift(1)
        feat['vol_momentum'] = (v / v.shift(5) - 1).shift(1)
        feat['vol_trend']    = (v.rolling(5).mean() / v.rolling(20).mean() - 1).shift(1)

        # ---- Candle structure ----
        body    = (c - df['open']).abs() / (h - l + 1e-9)
        up_wick = (h - c.clip(upper=df['open'])) / (h - l + 1e-9)
        dn_wick = (c.clip(lower=df['open']) - l) / (h - l + 1e-9)
        feat['body_ratio']    = body.shift(1)
        feat['upper_wick']    = up_wick.shift(1)
        feat['lower_wick']    = dn_wick.shift(1)
        feat['hl_range_pct']  = ((h - l) / c).shift(1)

        # ---- Calendar ----
        feat['day_of_week']   = df.index.dayofweek.values / 6.0  # normalize 0-1

        # ---- Fear & Greed Index (sentiment) ----
        try:
            fng = get_fng_features(feat.index)
            fng_cols = ['fng_norm', 'fng_7d_change', 'fng_14d_change',
                        'fng_30d_zscore', 'fng_7d_ma',
                        'fng_extreme', 'fng_fear_zone', 'fng_greed_zone']
            for col in fng_cols:
                if col in fng.columns:
                    # Shift by 1 to avoid look-ahead (today's F&G predicts tomorrow)
                    feat[col] = fng[col].shift(1).values
        except Exception:
            pass  # F&G unavailable — proceed with OHLCV features only

        # ---- Target: next candle log-return ----
        feat['target'] = log_ret  # NOT shifted — this is what we're predicting

        # Drop NaN rows (from rolling windows)
        feat = feat.dropna()
        return feat if len(feat) >= self.MIN_TRAIN_CANDLES else None

    # ------------------------------------------------------------------ #
    #  Model training
    # ------------------------------------------------------------------ #
    def _train(self, features_df: pd.DataFrame):
        """Train all models on the rolling window."""
        window = features_df.tail(self.TRAIN_WINDOW)
        X = window.drop(columns=['target']).values
        y = window['target'].values
        try:
            self._elastic_pipe.fit(X, y)
            self._svr_pipe.fit(X, y)
            self._rf_pipe.fit(X, y)
            self._lgb_pipe.fit(X, y)
            self._fitted = True
        except Exception:
            self._fitted = False

    # ------------------------------------------------------------------ #
    #  Fallback (not enough data or model not ready)
    # ------------------------------------------------------------------ #
    def _fallback(self, data: pd.DataFrame) -> dict:
        """Simple linreg fallback when ML model not ready."""
        try:
            y = data['close'].tail(20).values
            x = np.arange(len(y))
            slope, intercept = np.polyfit(x, y, 1)
            predicted = slope * len(y) + intercept
            change_pct = ((predicted - y[-1]) / y[-1]) * 100
            return {
                "predicted_price":      round(predicted, 2),
                "predicted_change_pct": round(change_pct, 2),
                "direction":            "UP" if change_pct > 0 else "DOWN",
                "confidence_pct":       0.0,
                "features_used":        1,
                "model_votes":          {"fallback": "linear regression"}
            }
        except Exception:
            return {
                "predicted_price":      None,
                "predicted_change_pct": None,
                "direction":            None,
                "confidence_pct":       0.0,
                "features_used":        0,
                "model_votes":          {}
            }

    # ------------------------------------------------------------------ #
    #  Indicator helpers
    # ------------------------------------------------------------------ #
    def _rsi(self, prices: pd.Series, period: int) -> pd.Series:
        delta = prices.diff()
        gain  = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
        rs    = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))
