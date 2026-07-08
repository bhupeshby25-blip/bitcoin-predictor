import numpy as np
import logging

def get_historical_volatility(recent_log_returns: np.ndarray) -> float:
    """Calculate the baseline expected movement magnitude (volatility) in the current window."""
    if len(recent_log_returns) < 5:
        return 0.02 # Fallback volatility 
    
    # Use 14-day rolling standard dev representing normal market noise 
    baseline_vol = np.std(recent_log_returns[-14:])
    
    # Cap and floor to prevent pathological edge cases
    return max(0.005, min(baseline_vol, 0.15))
    

def generate_trade_analysis(ensemble_log_ret: float, regime: str, historical_volatility: float) -> dict:
    """
    Translates the raw ensemble prediction into a user-friendly trade signal
    by comparing the magnitude of the prediction against historical volatility.
    """
    
    vol = max(historical_volatility, 1e-9)
    strength_ratio = ensemble_log_ret / vol
    abs_strength = abs(strength_ratio)
    
    # Use thresholds to construct non-noisy actionable feedback
    if strength_ratio > 0.6:  # High magnitude UP
        signal = "STRONG_BUY"
        score  = min(99, 60 + (abs_strength * 20)) 
    elif strength_ratio > 0.15: # Weak UP
        signal = "WEAK_BUY"
        score  = min(59, 40 + (abs_strength * 20))
    elif strength_ratio < -0.6: # High magnitude DOWN
        signal = "STRONG_SELL"
        score  = min(99, 60 + (abs_strength * 20)) 
    elif strength_ratio < -0.15: # Weak DOWN 
        signal = "WEAK_SELL"
        score  = min(59, 40 + (abs_strength * 20))
    else: # Within noise threshold (~ +/- 0.15 deviation)
        signal = "NEUTRAL"
        score  = 50

    # Human readable insights
    if "STRONG" in signal:
        narrative = f"High conviction trend alignment detected. Outsized momentum likely in current {regime.replace('_', ' ').lower()}"
    elif "NEUTRAL" in signal:
        narrative = f"Prediction magnitude falls within normal background noise. No statistically significant edge available."
    else:
        narrative = f"Weak alignment. Marginal edge vs normal variance."

    return {
        "trade_signal": signal,
        "conviction_score": round(score, 1), 
        "analysis_summary": narrative,
        "volatility_context": round(vol * 100, 2)
    }
