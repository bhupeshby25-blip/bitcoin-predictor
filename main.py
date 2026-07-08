"""
Bitcoin Price Prediction & Recommendation Bot
----------------------------------------------
Uses python-telegram-bot's Application + JobQueue to:
  - Run analysis on a configurable, live-adjustable interval
  - Accept commands via the Telegram channel/chat

Commands:
  /start           - Start the bot
  /help            - List commands
  /setinterval Xm|Xh - Change analysis interval (e.g. /setinterval 30m or /setinterval 4h)
  /status          - Show current settings and last signal
"""

import logging
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
)

from data_loader import DataLoader
from risk_manager import RiskManager
from strategies.regime_aware_strategy import RegimeAwareStrategy
from config import (
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
    SYMBOL, TIMEFRAME, LIMIT, DEFAULT_RISK_PER_TRADE
)

# ------------------------------------------------------------------ #
#  Logging
# ------------------------------------------------------------------ #
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Shared state (stored in bot_data so all handlers share it)
# ------------------------------------------------------------------ #
DEFAULT_INTERVAL_SECONDS = 3600     # 1 hour

# ------------------------------------------------------------------ #
#  Core analysis job
# ------------------------------------------------------------------ #
async def analysis_job(context: ContextTypes.DEFAULT_TYPE):
    """Runs analysis and sends a Telegram message. Scheduled by JobQueue."""
    loader: DataLoader   = context.bot_data["loader"]
    risk_mgr: RiskManager = context.bot_data["risk_manager"]
    strategies: list     = context.bot_data["strategies"]
    chat_id: str         = context.bot_data["chat_id"]
    timeframe: str       = context.bot_data.get("timeframe", TIMEFRAME)

    logger.info("Running analysis...")
    data = loader.fetch_ohlcv(limit=LIMIT)

    if data.empty:
        await context.bot.send_message(chat_id=chat_id, text="⚠️ Could not fetch market data. Retrying next interval.")
        return

    current_price = data['close'].iloc[-1]

    # Risk classification (once per interval)
    risk_info = risk_mgr.classify_risk(data)

    for strategy in strategies:
        signal            = strategy.analyze(data)
        action            = signal.get("action", "HOLD")
        reason            = signal.get("reason", "—")
        predicted_price   = signal.get("predicted_price")
        predicted_chg_pct = signal.get("predicted_change_pct")
        confidence        = signal.get("confidence", "—")
        regime_info_sig   = signal.get("regime", {})
        sub_signals       = signal.get("sub_signals", [])

        # --- Position sizing ---
        sl_pct    = 0.05
        stop_loss = current_price * (1 - sl_pct) if action == "BUY" else current_price * (1 + sl_pct)
        pos       = risk_mgr.calculate_position_size(current_price, stop_loss)

        # --- Prediction line ---
        if predicted_price and predicted_chg_pct is not None:
            direction = "📈" if predicted_chg_pct >= 0 else "📉"
            pred_line = f"{direction} *Predicted ({timeframe})*: ${predicted_price:,.2f} ({predicted_chg_pct:+.2f}%)"
        else:
            pred_line = "Prediction unavailable"

        # --- Action emoji ---
        action_emoji = {"BUY": "🟢", "SELL": "🔴"}.get(action, "⚪")

        # --- Regime block (only for RegimeAwareStrategy) ---
        if regime_info_sig:
            reg = regime_info_sig
            regime_block = (
                f"🌍 *Market Regime*: {reg.get('emoji_label', reg.get('regime', '—'))}\n"
                f"  ADX: {reg.get('adx', '—')}  |  ATR: {reg.get('atr_pct', '—')}%\n"
                f"  {reg.get('description', '')}\n\n"
            )
        else:
            regime_block = ""

        # --- Sub-signals breakdown ---
        if sub_signals:
            sub_lines = "\n".join(
                f"  {'🟢' if s['action']=='BUY' else '🔴' if s['action']=='SELL' else '⚪'} "
                f"{s['strategy']}: {s['action']}"
                for s in sub_signals
            )
            sub_block = f"📊 *Sub-Strategies*:\n{sub_lines}\n\n"
        else:
            sub_block = ""

        msg = (
            f"{action_emoji} *{action} — {SYMBOL}*  |  Confidence: {confidence}\n\n"
            f"💰 *Price Now*: ${current_price:,.2f}\n"
            f"{pred_line}\n\n"
            f"{regime_block}"
            f"{sub_block}"
            f"📝 *Reason*: {reason}\n\n"
            f"🛡 *Risk*: {risk_info['emoji']} *{risk_info['label']}*  |  ATR: {risk_info['atr_pct']}%\n"
            f"  {risk_info['description']}\n\n"
            f"📐 *Position Sizing* (2% risk rule):\n"
            f"  Stop Loss: ${pos.get('stop_loss', '—'):,}\n"
            f"  Recommended: {pos.get('position_size_units', '—')} BTC\n"
            f"  Max Risk: ${pos.get('risk_amount_usd', '—')}"
        )

        await context.bot.send_message(
            chat_id=chat_id, text=msg, parse_mode="Markdown"
        )
        logger.info(f"Sent {action} signal | Regime: {regime_info_sig.get('regime','—')} | Confidence: {confidence}")

    # Store last signal for /status
    context.bot_data["last_signal"] = {
        "action": action,
        "price":  current_price,
        "risk":   risk_info["label"],
        "regime": regime_info_sig.get("regime", "—") if regime_info_sig else "—"
    }


# ------------------------------------------------------------------ #
#  Command Handlers
# ------------------------------------------------------------------ #
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    interval_sec = context.bot_data.get("interval_seconds", DEFAULT_INTERVAL_SECONDS)
    interval_str = _fmt_interval(interval_sec)
    await update.message.reply_text(
        f"🚀 *Bitcoin Prediction Bot is running!*\n\n"
        f"Current symbol   : `{SYMBOL}`\n"
        f"Analysis interval: `{interval_str}`\n\n"
        f"Use /help to see available commands.",
        parse_mode="Markdown"
    )

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📖 *Available Commands*\n\n"
        "/start — Show bot status\n"
        "/status — Last signal + current settings\n"
        "/setinterval <value> — Change analysis interval\n"
        "   Examples: `/setinterval 15m`  `/setinterval 1h`  `/setinterval 4h`\n"
        "/help — This message",
        parse_mode="Markdown"
    )

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    interval_sec = context.bot_data.get("interval_seconds", DEFAULT_INTERVAL_SECONDS)
    last = context.bot_data.get("last_signal", {})
    if last:
        last_txt = (
            f"Last action : *{last['action']}* at ${last['price']:,.2f}\n"
            f"Risk level  : {last['risk']}\n"
            f"Regime      : {last.get('regime', '—')}"
        )
    else:
        last_txt = "No analysis run yet."

    await update.message.reply_text(
        f"📊 *Bot Status*\n\n"
        f"Symbol   : `{SYMBOL}`\n"
        f"Interval : `{_fmt_interval(interval_sec)}`\n\n"
        f"{last_txt}",
        parse_mode="Markdown"
    )

async def cmd_set_interval(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /setinterval 30m  or  /setinterval 2h
    """
    if not context.args:
        await update.message.reply_text(
            "⚠️ Usage: `/setinterval <value>`\nExamples: `15m` `1h` `4h`",
            parse_mode="Markdown"
        )
        return

    raw = context.args[0].strip().lower()
    seconds = _parse_interval(raw)

    if seconds is None:
        await update.message.reply_text(
            "❌ Invalid format. Use `15m`, `1h`, `4h`, etc.",
            parse_mode="Markdown"
        )
        return

    # Minimum 5 minutes to avoid rate limit issues
    if seconds < 300:
        await update.message.reply_text("⚠️ Minimum interval is 5 minutes (5m).")
        return

    # Remove old job and add new one
    current_jobs = context.job_queue.get_jobs_by_name("analysis")
    for job in current_jobs:
        job.schedule_removal()

    context.job_queue.run_repeating(
        analysis_job,
        interval=seconds,
        first=10,
        name="analysis",
        chat_id=context.bot_data["chat_id"]
    )
    context.bot_data["interval_seconds"] = seconds

    await update.message.reply_text(
        f"✅ Interval updated to *{_fmt_interval(seconds)}*. "
        f"Next analysis in ~10 seconds.",
        parse_mode="Markdown"
    )
    logger.info(f"Interval changed to {seconds}s")


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #
def _parse_interval(raw: str):
    """Convert '30m' or '2h' to seconds. Returns None on failure."""
    try:
        if raw.endswith('m'):
            return int(raw[:-1]) * 60
        elif raw.endswith('h'):
            return int(raw[:-1]) * 3600
        elif raw.endswith('s'):
            return int(raw[:-1])
        else:
            return int(raw)    # assume seconds if no suffix
    except ValueError:
        return None

def _fmt_interval(seconds: int) -> str:
    if seconds >= 3600 and seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    elif seconds >= 60 and seconds % 60 == 0:
        return f"{seconds // 60}m"
    else:
        return f"{seconds}s"


# ------------------------------------------------------------------ #
#  Entry Point
# ------------------------------------------------------------------ #
def main():
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in .env")
        return

    print("---------------------------------------------------")
    print("🚀 Bitcoin Prediction Bot Starting...")
    print("---------------------------------------------------")

    # Build application
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Shared state
    app.bot_data["loader"]       = DataLoader(symbol=SYMBOL, timeframe=TIMEFRAME)
    app.bot_data["risk_manager"] = RiskManager(risk_per_trade=DEFAULT_RISK_PER_TRADE)
    app.bot_data["strategies"]   = [RegimeAwareStrategy()]
    app.bot_data["chat_id"]      = TELEGRAM_CHAT_ID
    app.bot_data["interval_seconds"] = DEFAULT_INTERVAL_SECONDS
    app.bot_data["timeframe"]    = TIMEFRAME

    # Register command handlers
    app.add_handler(CommandHandler("start",       cmd_start))
    app.add_handler(CommandHandler("help",        cmd_help))
    app.add_handler(CommandHandler("status",      cmd_status))
    app.add_handler(CommandHandler("setinterval", cmd_set_interval))

    # Schedule the recurring analysis job
    app.job_queue.run_repeating(
        analysis_job,
        interval=DEFAULT_INTERVAL_SECONDS,
        first=10,               # run first analysis 10s after startup
        name="analysis",
        chat_id=TELEGRAM_CHAT_ID
    )

    print(f"✅ Bot started. Monitoring {SYMBOL} every {_fmt_interval(DEFAULT_INTERVAL_SECONDS)}.")
    print("   Send /setinterval 30m in your Telegram chat to change it.")
    print("   Press Ctrl+C to stop.\n")

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
