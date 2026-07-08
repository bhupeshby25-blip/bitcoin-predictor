"""
RegimeConditionedPredictor
===========================
Academic research shows regime-specific ML models significantly outperform
single global models for crypto price prediction. This implements that approach.

Architecture:
  Step 1: Detect current market regime (reuses our RegimeDetector)
  Step 2: Route to the regime's dedicated ML model
  Step 3: Each model is trained ONLY on data from that regime
          (so it learns patterns specific to that market condition)

Why regime-specific models are better:
  In a BULL market, the most predictive features are:
    → EMA crossovers, MACD momentum, RSI < 70 (not overbought yet)
    → F&G greed levels (momentum continuation)
  In a BEAR market, the most predictive features are:
    → RSI (2) extreme oversold bounces
    → Volume spikes (capitulation detection)
    → F&G extreme fear (contrarian signal stronger in bears)
  In RANGING markets, the best predictors are:
    → Bollinger %B (mean reversion signals)
    → Z-score from mean (statistical extremes)
  In HIGH_VOLATILITY, the best predictors are:
    → ATR-relative moves (volatility-normalized signals)
    → Recent momentum direction (breakout continuation)

Reference: "Regime-switching ML models for Bitcoin" (MDPI 2024),
           "HMM with covariate selection for crypto" (arxiv 2024)
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb

from strategies.regime_detector import RegimeDetector
from fear_greed import FearGreedLoader
from fear_greed_signal import FearGreedSignal


# ── Regime-specific feature sets ──────────────────────────────────────────── #
# Each regime emphasizes different features for prediction
REGIME_FEATURES = {
    "BULL_TREND": [
        # Trend continuation features dominate in bull markets
        'ret_1d', 'ret_3d', 'ret_5d', 'ret_7d',
        'ema_spread', 'sma20_slope', 'sma50_slope',
        'macd', 'macd_hist',
        'rsi_14', 'rsi_7',
        'vol_momentum', 'vol_trend',
        'fng_norm', 'fng_7d_change', 'fng_greed_zone',  # greed = momentum
        'day_of_week',
    ],
    "BEAR_TREND": [
        # Mean reversion + capitulation features dominate in bear markets
        'ret_1d', 'ret_2d', 'ret_3d',
        'rsi_2', 'rsi_7',                      # oversold bounces
        'vol_zscore', 'vol_momentum',           # volume spikes = capitulation
        'bb_pct_b', 'atr_pct',
        'lower_wick', 'body_ratio',             # hammer candle patterns
        'fng_norm', 'fng_7d_change', 'fng_30d_zscore', 'fng_fear_zone', 'fng_extreme',
        'price_vs_sma20', 'price_vs_sma50',
    ],
    "RANGING": [
        # Mean reversion from extremes dominates in ranging markets
        'ret_1d', 'ret_2d', 'ret_3d',
        'bb_pct_b', 'bb_width',                # Bollinger band position
        'rsi_14', 'rsi_7', 'rsi_2',
        'price_vs_sma20',
        'atr_pct', 'realized_vol',
        'vol_zscore',
        'fng_norm', 'fng_30d_zscore',
        'upper_wick', 'lower_wick',             # rejection candles
        'day_of_week',
    ],
    "HIGH_VOLATILITY": [
        # Volatility-normalized momentum dominates in high vol regimes
        'ret_1d', 'ret_2d', 'ret_3d', 'ret_5d',
        'atr_pct', 'realized_vol', 'bb_width',  # volatility state
        'hl_range_pct', 'body_ratio',           # candle structure in volatile periods
        'macd', 'macd_hist',
        'rsi_14', 'rsi_7',
        'vol_zscore', 'vol_momentum',
        'fng_norm', 'fng_7d_change', 'fng_extreme',  # sentiment in high vol
        'sma20_slope',
    ],
}

# Fallback: use all features if regime not recognized
ALL_FEATURES_FALLBACK = list(set(
    f for features in REGIME_FEATURES.values() for f in features
))


class RegimeConditionedPredictor:
    """
    Separate ML model per market regime, each trained on regime-specific data.
    Combines predictions from the regime-specific model with the F&G signal.
    """

    MIN_REGIME_SAMPLES = 40   # minimum samples needed to train a regime model
    TRAIN_WINDOW       = 300  # longer window to collect enough regime samples
    RETRAIN_EVERY      = 15   # candles between retrains

    def __init__(self):
        self.detector    = RegimeDetector()
        self.fng_loader  = FearGreedLoader()
        self.fng_signal  = FearGreedSignal()

        # One pipeline per regime
        self._models = {
            regime: self._make_pipeline()
            for regime in REGIME_FEATURES
        }
        self._fitted_regimes = set()
        self._last_train_i   = -1
        self._full_feat_df   = None   # cached feature dataframe

    def predict(self, data: pd.DataFrame) -> dict:
        """
        Detect regime → route to regime-specific model → blend with F&G.
        """
        if len(data) < 120:
            return self._fallback(data)

        # Detect current regime
        regime_info = self.detector.detect(data)
        regime      = regime_info.get("regime", "HIGH_VOLATILITY")

        # Build features
        feat_df = self._build_all_features(data)
        if feat_df is None or len(feat_df) < 80:
            return self._fallback(data)

        # Retrain periodically
        if (self._last_train_i < 0 or
                len(data) - self._last_train_i >= self.RETRAIN_EVERY):
            self._train_all_regimes(feat_df, data)
            self._last_train_i = len(data)

        # Get regime-specific features
        feat_cols = [c for c in REGIME_FEATURES.get(regime, ALL_FEATURES_FALLBACK)
                     if c in feat_df.columns]
        if not feat_cols or regime not in self._fitted_regimes:
            # Fallback to global features
            return self._global_predict(feat_df, data, regime)

        X_pred = feat_df[feat_cols].iloc[[-1]].values

        try:
            pred = self._models[regime].predict(X_pred)[0]
        except Exception:
            return self._fallback(data)

        # F&G signal blend
        fng_df     = self.fng_loader._load_data()
        fng_series = fng_df['fng_value'].reindex(
            data.index.normalize()
        ).ffill() if fng_df is not None else pd.Series(dtype=float)
        fng_result = self.fng_signal.predict(fng_series)
        fng_pred   = fng_result['log_ret_pred']
        fng_prob   = fng_result['prob_up']

        # Dynamic F&G weight based on sentiment extremity
        fng_strength = abs(fng_prob - 0.5) * 2
        fng_weight   = 0.10 + 0.20 * fng_strength   # 10–30%

        ensemble_log_ret = (1 - fng_weight) * pred + fng_weight * fng_pred
        
        # Calculate recent volatility matching the implementation protocol
        from analysis_utils import get_historical_volatility, generate_trade_analysis
        historical_volatility = get_historical_volatility(data['close'].pct_change().dropna().values)
        
        # Generate new user-friendly insights rather than pure data
        analysis_data = generate_trade_analysis(ensemble_log_ret, regime, historical_volatility)

        current_price   = data['close'].iloc[-1]
        predicted_price = current_price * np.exp(ensemble_log_ret)
        change_pct      = (np.exp(ensemble_log_ret) - 1) * 100

        # Confidence: agreement between regime model and F&G
        tech_dir = "UP" if pred > 0 else "DOWN"
        fng_dir  = "UP" if fng_pred > 0 else "DOWN"
        
        # We blend the internal technical consensus with the new variance based conviction score
        base_agreement = 100.0 if tech_dir == fng_dir else 50.0
        final_conviction = (analysis_data['conviction_score'] + base_agreement) / 2

        return {
            "predicted_price":        round(predicted_price, 2),
            "predicted_change_pct":   round(change_pct, 2),
            "action":                 analysis_data['trade_signal'], # Changed from 'direction'
            "analysis_summary":       analysis_data['analysis_summary'],
            "conviction_score":       round(final_conviction, 1), # Changed from 'confidence_pct'
            "volatility_context":     analysis_data['volatility_context'],
            "regime":                 regime,
            "features_used":          len(feat_cols),
            "fng_value":              fng_result["fng_value"],
            "fng_class":              fng_result["fng_class"],
            "fng_weight_pct":         round(fng_weight * 100, 1),
            "fng_signal":             fng_result["signal"],
            "model_votes": {
                f"Regime-{regime[:4]}": tech_dir,
                "F&G":                  fng_dir,
            }
        }

    # ── Training ──────────────────────────────────────────────────────────── #
    def _train_all_regimes(self, feat_df: pd.DataFrame, raw_data: pd.DataFrame):
        """
        Label each candle with its historical regime, then train each model
        only on candles from that regime.
        """
        regimes = []
        for i in range(len(raw_data)):
            if i < 60:
                regimes.append("HIGH_VOLATILITY")
                continue
            w = raw_data.iloc[:i+1]
            try:
                r = self.detector.detect(w)
                regimes.append(r.get("regime", "HIGH_VOLATILITY"))
            except Exception:
                regimes.append("HIGH_VOLATILITY")

        regime_series = pd.Series(regimes, index=raw_data.index)

        for regime, feat_cols_all in REGIME_FEATURES.items():
            feat_cols = [c for c in feat_cols_all if c in feat_df.columns]
            if not feat_cols:
                continue

            # Align regime labels to feat_df index
            aligned_regimes = regime_series.reindex(feat_df.index)
            mask = aligned_regimes == regime

            regime_data = feat_df[mask].tail(self.TRAIN_WINDOW)
            if len(regime_data) < self.MIN_REGIME_SAMPLES:
                continue  # not enough data for this regime

            X = regime_data[feat_cols].values
            y = regime_data['target'].values

            if np.any(np.isnan(X)) or np.any(np.isnan(y)):
                X = np.nan_to_num(X)
                y = np.nan_to_num(y)

            try:
                self._models[regime].fit(X, y)
                self._fitted_regimes.add(regime)
            except Exception:
                pass

    # ── Feature engineering (full set, regime slices selected later) ─────── #
    def _build_all_features(self, data: pd.DataFrame) -> pd.DataFrame | None:
        """Build the complete feature matrix; regime-specific columns selected at predict time."""
        df = data.copy()
        c, h, l, v = df['close'], df['high'], df['low'], df['volume']

        feat = pd.DataFrame(index=df.index)
        log_ret = np.log(c / c.shift(1))

        # Returns
        for lag in [1, 2, 3, 5, 7, 10, 14]:
            feat[f'ret_{lag}d'] = log_ret.shift(1).rolling(lag).sum()

        # Trend
        sma20 = c.rolling(20).mean()
        sma50 = c.rolling(50).mean()
        ema7  = c.ewm(span=7, adjust=False).mean()
        ema21 = c.ewm(span=21, adjust=False).mean()
        feat['sma20_slope']    = (sma20 / sma20.shift(5) - 1).shift(1)
        feat['sma50_slope']    = (sma50 / sma50.shift(10) - 1).shift(1)
        feat['ema_spread']     = ((ema7 - ema21) / ema21).shift(1)
        feat['price_vs_sma20'] = ((c - sma20) / sma20).shift(1)
        feat['price_vs_sma50'] = ((c - sma50) / sma50).shift(1)

        # Oscillators
        for period in [2, 7, 14]:
            feat[f'rsi_{period}'] = self._rsi(c, period).shift(1)

        # MACD
        macd_line   = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        feat['macd']      = (macd_line / c).shift(1)
        feat['macd_hist'] = ((macd_line - signal_line) / c).shift(1)

        # Volatility
        tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean()
        feat['atr_pct']       = (atr14 / c).shift(1)
        feat['realized_vol']  = log_ret.rolling(10).std().shift(1)

        bb_std  = c.rolling(20).std()
        feat['bb_width']  = (2 * bb_std / sma20).shift(1)
        feat['bb_pct_b']  = ((c - (sma20 - 2*bb_std)) / (4*bb_std + 1e-9)).shift(1)

        # Volume
        vol_mean = v.rolling(20).mean()
        vol_std  = v.rolling(20).std()
        feat['vol_zscore']   = ((v - vol_mean) / (vol_std + 1e-9)).shift(1)
        feat['vol_momentum'] = (v / v.shift(5) - 1).shift(1)
        feat['vol_trend']    = (v.rolling(5).mean() / v.rolling(20).mean() - 1).shift(1)

        # Candle structure
        body    = (c - df['open']).abs() / (h - l + 1e-9)
        up_wick = (h - c.clip(upper=df['open'])) / (h - l + 1e-9)
        dn_wick = (c.clip(lower=df['open']) - l) / (h - l + 1e-9)
        feat['body_ratio']   = body.shift(1)
        feat['upper_wick']   = up_wick.shift(1)
        feat['lower_wick']   = dn_wick.shift(1)
        feat['hl_range_pct'] = ((h - l) / c).shift(1)

        # Calendar
        feat['day_of_week'] = df.index.dayofweek.values / 6.0

        # Fear & Greed
        try:
            fng_df = self.fng_loader._load_data()
            if fng_df is not None:
                fng_aligned = fng_df.reindex(feat.index).ffill()
                for col in ['fng_norm', 'fng_7d_change', 'fng_14d_change',
                            'fng_30d_zscore', 'fng_7d_ma',
                            'fng_extreme', 'fng_fear_zone', 'fng_greed_zone']:
                    if col in fng_aligned.columns:
                        feat[col] = fng_aligned[col].shift(1).values
        except Exception:
            pass

        # Target
        feat['target'] = log_ret

        feat = feat.replace([np.inf, -np.inf], np.nan).dropna()
        return feat if len(feat) >= 80 else None

    # ── Helpers ────────────────────────────────────────────────────────────── #
    def _make_pipeline(self):
        """Create the per-regime model: ElasticNet won the benchmark at 55%."""
        return Pipeline([
            ('scaler', StandardScaler()),
            ('model',  ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000))
        ])

    def _global_predict(self, feat_df, data, regime):
        """Fallback global prediction when regime model not ready."""
        from predictor import MLPredictor
        _tmp = MLPredictor()
        return _tmp._fallback(data)

    def _rsi(self, prices: pd.Series, period: int) -> pd.Series:
        delta = prices.diff()
        gain  = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
        rs    = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def _fallback(self, data: pd.DataFrame) -> dict:
        try:
            y = data['close'].tail(20).values
            x = np.arange(len(y))
            slope, intercept = np.polyfit(x, y, 1)
            predicted  = slope * len(y) + intercept
            change_pct = ((predicted - y[-1]) / y[-1]) * 100
            return {
                "predicted_price":      round(predicted, 2),
                "predicted_change_pct": round(change_pct, 2),
                "direction":            "UP" if change_pct > 0 else "DOWN",
                "confidence_pct":       0.0,
                "regime":               "UNKNOWN",
                "features_used":        1,
                "fng_value":            50,
                "fng_class":            "Neutral",
                "fng_weight_pct":       10.0,
                "fng_signal":           "fallback",
                "model_votes":          {"fallback": "linear regression"},
            }
        except Exception:
            return {"predicted_price": None, "predicted_change_pct": None,
                    "direction": None, "confidence_pct": 0.0,
                    "regime": "UNKNOWN", "features_used": 0,
                    "fng_value": 50, "fng_class": "Neutral",
                    "fng_weight_pct": 10.0, "fng_signal": "N/A", "model_votes": {}}
