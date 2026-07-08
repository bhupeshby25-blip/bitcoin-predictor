import numpy as np

def generate_trade_analysis(ensemble_log_ret: float, regime: str, historical_volatility: float) -> dict:
    """
    Translates the raw ensemble prediction into a user-friendly trade signal
    by comparing the magnitude of the prediction against historical volatility.
    """
    
    # Avoid zero division
    vol = max(historical_volatility, 1e-9)
    
    # Calculate the strength ratio: Predictied Return vs Volatility
    strength_ratio = ensemble_log_ret / vol
    
    # Identify base action and scale
    if strength_ratio > 0.5:
        signal = "STRONG_BUY"
        score = min(99, 60 + (strength_ratio * 20)) 
    elif strength_ratio > 0:
        signal = "WEAK_BUY"
        score = min(60, 40 + (strength_ratio * 20))
    elif strength_ratio > -0.5:
        signal = "WEAK_SELL"
        score = min(60, 40 + (abs(strength_ratio) * 20))
    else:
        signal = "STRONG_SELL"
        score = min(99, 60 + (abs(strength_ratio) * 20))

    # Context analysis builder
    if "STRONG" in signal:
        narrative = f"High conviction trend alignment detected in {regime}."
    else:
        narrative = f"Wait/Hold. Actionable edge is within normal market noise."

    return {
        "trade_signal": signal,
        "conviction_score": round(score, 1), 
        "analysis_summary": narrative
    }
