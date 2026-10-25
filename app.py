import os, sqlite3, time, logging, threading, math
from datetime import datetime, timedelta
from functools import lru_cache

import requests
import pandas as pd
from flask import Flask, jsonify, render_template_string
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
TWELVE_KEY = os.getenv("e4f18046c5b74d5fa1d804b70f4bd3d2", "")
BOT_TOKEN  = os.getenv("8867619342:AAGeX1XHtf1IkZig7Rry_EqsPNvDFhYQCMc", "")
CHANNEL_ID = os.getenv("1004268221265", "")
PORT       = int(os.getenv("PORT", 5000))
DEV        = "@exodus_own3r"
DB_PATH    = "signals.db"

SYMBOLS = {
    "XAUUSD": {"td":"XAU/USD","yf":"XAUUSD=X","decimals":2,"label":"GOLD (XAU/USD)",
               "icon":"🥇","pip_size":0.1,"lot_multiplier":100.0},
    "EURUSD": {"td":"EUR/USD","yf":"EURUSD=X","decimals":5,"label":"EUR/USD",
               "icon":"💶","pip_size":0.0001,"lot_multiplier":100000.0},
}

EMA_FAST, EMA_SLOW, EMA_TREND = 9, 21, 50
RSI_LEN, RSI_OS, RSI_OB = 14, 30, 70
ATR_LEN, ATR_SL, ATR_TP = 14, 1.5, 2.5
MIN_CONF = 3

# Simple in-memory cache to avoid hammering APIs
_cache = {"signals": {}, "prices": {}, "ts": 0}
CACHE_TTL = 60  # seconds


# ─── DATABASE ─────────────────────────────────────────────────────────────────
def init_db():
    with sqlite3.connect(DB_PATH) as c:
        c.execute('''CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL, direction TEXT NOT NULL,
            entry_price REAL NOT NULL, sl REAL NOT NULL,
            tp1 REAL NOT NULL, tp2 REAL NOT NULL,
            confidence TEXT, confluence_score INTEGER,
            timeframe TEXT DEFAULT 'M15/H1', strategy TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            outcome TEXT DEFAULT 'PENDING', pnl_pips REAL
        )''')
    log.info("[DB] initialized")


def save_signal(s: dict) -> int:
    with sqlite3.connect(DB_PATH) as c:
        cur = c.execute('''INSERT INTO signals
            (symbol,direction,entry_price,sl,tp1,tp2,confidence,
             confluence_score,timeframe,strategy)
            VALUES (?,?,?,?,?,?,?,?,?,?)''',
            (s["symbol"], s["direction"], s["entry"], s["sl"], s["tp1"], s["tp2"],
             s.get("confidence",""), s.get("confluence_score",0),
             s.get("timeframe","M15/H1"), s.get("strategy","")))
        return cur.lastrowid


def get_signals(limit=30):
    with sqlite3.connect(DB_PATH) as c:
        c.row_factory = sqlite3.Row
        rows = c.execute("SELECT * FROM signals ORDER BY timestamp DESC LIMIT ?",
                         (limit,)).fetchall()
    return [dict(r) for r in rows]


def get_stats():
    with sqlite3.connect(DB_PATH) as c:
        r = c.execute('''SELECT
            COUNT(*),
            SUM(CASE WHEN outcome='WIN' THEN 1 ELSE 0 END),
            SUM(CASE WHEN outcome='LOSS' THEN 1 ELSE 0 END),
            AVG(COALESCE(pnl_pips,0))
            FROM signals WHERE outcome != 'PENDING' ''').fetchone()
    total, wins, losses, avg = r
    total = total or 0; wins = wins or 0; losses = losses or 0
    return {"total":total,"wins":wins,"losses":losses,
            "avg_pips":round(avg or 0.0,1),
            "win_rate":round(wins/total*100 if total else 0.0,1)}


def resolve_pending_signals():
    """Walk pending signals, check if TP1 or SL was hit using current price.
    This makes the win-rate actually work instead of staying at 0%."""
    try:
        with sqlite3.connect(DB_PATH) as c:
            c.row_factory = sqlite3.Row
            pend = c.execute("SELECT * FROM signals WHERE outcome='PENDING'").fetchall()
        for s in pend:
            price = get_current_price(s["symbol"], use_cache=False)
            if not price:
                continue
            hit_tp = hit_sl = False
            if s["direction"] == "BUY":
                hit_tp = price >= s["tp1"]
                hit_sl = price <= s["sl"]
            else:
                hit_tp = price <= s["tp1"]
                hit_sl = price >= s["sl"]
            if hit_tp or hit_sl:
                outcome = "WIN" if hit_tp else "LOSS"
                pip = SYMBOLS[s["symbol"]]["pip_size"]
                pnl = (abs(price - s["entry_price"]) / pip) * (1 if hit_tp else -1)
                with sqlite3.connect(DB_PATH) as c:
                    c.execute("UPDATE signals SET outcome=?, pnl_pips=? WHERE id=?",
                              (outcome, round(pnl,1), s["id"]))
                log.info(f"[RESOLVE] {s['symbol']} {s['direction']} → {outcome} ({pnl:.1f} pips)")
    except Exception as e:
        log.warning(f"[RESOLVE ERR] {e}")


# ─── DATA FETCHERS ────────────────────────────────────────────────────────────
def _twelve_data(td_symbol, interval, outputsize=200):
    if not TWELVE_KEY:
        raise ValueError("no key")
    r = requests.get("https://api.twelvedata.com/time_series",
        params={"symbol":td_symbol,"interval":interval,
                "outputsize":outputsize,"apikey":TWELVE_KEY,"format":"JSON"},
        timeout=12).json()
    if "values" not in r:
        raise ValueError(r.get("message","TD error"))
    df = pd.DataFrame(r["values"])
    for col in ("open","high","low","close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.iloc[::-1].reset_index(drop=True).dropna()


def _yf_fallback(yf_sym, period, interval):
    """Lazy-import yfinance (heavy)."""
    import yfinance as yf
    df = yf.download(yf_sym, period=period, interval=interval,
                     progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"yf empty {yf_sym}")
    df.columns = [c[0] if isinstance(c,tuple) else c for c in df.columns]
    return df.rename(columns={"Open":"open","High":"high",
                              "Low":"low","Close":"close"}).dropna()


def get_multi_tf(symbol):
    meta = SYMBOLS[symbol]
    try:
        return _twelve_data(meta["td"],"1h",200), _twelve_data(meta["td"],"15min",150)
    except Exception as e:
        log.warning(f"[TD→YF] {symbol}: {e}")
    return _yf_fallback(meta["yf"],"10d","1h"), _yf_fallback(meta["yf"],"3d","15m")


def get_current_price(symbol, use_cache=True):
    now = time.time()
    if use_cache and symbol in _cache["prices"] and now - _cache["ts"] < CACHE_TTL:
        return _cache["prices"][symbol]
    meta = SYMBOLS[symbol]
    price = 0.0
    if TWELVE_KEY:
        try:
            r = requests.get("https://api.twelvedata.com/price",
                params={"symbol":meta["td"],"apikey":TWELVE_KEY}, timeout=8).json()
            price = float(r.get("price",0) or 0)
        except Exception:
            pass
    if not price:
        try:
            df = _yf_fallback(meta["yf"],"1d","1m")
            price = float(df["close"].iloc[-1])
        except Exception:
            pass
    if price:
        _cache["prices"][symbol] = price
    return price


# ─── INDICATORS (hand-rolled — no pandas-ta dependency) ───────────────────────
def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()


def rsi(series, length=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(length).mean()
    loss  = (-delta.clip(upper=0)).rolling(length).mean()
    rs    = gain / loss.replace(0, math.nan)
    return 100 - 100/(1+rs)


def atr(high, low, close, length=14):
    tr = pd.concat([high-low,
                    (high-close.shift()).abs(),
                    (low -close.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(length).mean()


def macd_hist(series, fast=12, slow=26, sig=9):
    macd = ema(series,fast) - ema(series,slow)
    signal = ema(macd, sig)
    return macd - signal


def stoch_kd(high, low, close, k_len=14, d_len=3):
    ll = low.rolling(k_len).min()
    hh = high.rolling(k_len).max()
    k  = 100*(close-ll)/(hh-ll).replace(0, math.nan)
    d  = k.rolling(d_len).mean()
    return k, d


def add_indicators(df):
    df = df.copy()
    df["ema_fast"]  = ema(df["close"], EMA_FAST)
    df["ema_slow"]  = ema(df["close"], EMA_SLOW)
    df["ema_trend"] = ema(df["close"], EMA_TREND)
    df["rsi"]       = rsi(df["close"], RSI_LEN)
    df["atr"]       = atr(df["high"], df["low"], df["close"], ATR_LEN)
    df["macd_hist"] = macd_hist(df["close"])
    k, d = stoch_kd(df["high"], df["low"], df["close"])
    df["stoch_k"], df["stoch_d"] = k, d
    return df.dropna()


# ─── SIGNAL ENGINE ────────────────────────────────────────────────────────────
def analyze_signal(symbol):
    now = time.time()
    cached = _cache["signals"].get(symbol)
    if cached and now - cached["_ts"] < CACHE_TTL:
        return cached["data"]
    try:
        df_h1, df_m15 = get_multi_tf(symbol)
    except Exception as e:
        log.error(f"[SIGNAL ERR] {symbol}: {e}")
        return None
    df_h1, df_m15 = add_indicators(df_h1), add_indicators(df_m15)
    if len(df_h1) < 3 or len(df_m15) < 3:
        return None

    h1, h1p   = df_h1.iloc[-1],  df_h1.iloc[-2]
    m15, m15p = df_m15.iloc[-1], df_m15.iloc[-2]
    price = float(m15["close"]); atr_v = float(m15["atr"])
    bull = bear = 0; reasons = []

    # F1: H1 EMA trend
    if float(h1["ema_slow"]) > float(h1["ema_trend"]):
        bull += 1; reasons.append("H1 Trend: Bullish EMA stack")
    elif float(h1["ema_slow"]) < float(h1["ema_trend"]):
        bear += 1; reasons.append("H1 Trend: Bearish EMA stack")

    # F2: M15 EMA cross
    if (float(m15p["ema_fast"]) < float(m15p["ema_slow"]) and
        float(m15["ema_fast"])  > float(m15["ema_slow"])):
        bull += 1; reasons.append("M15 EMA Cross: Bullish")
    elif (float(m15p["ema_fast"]) > float(m15p["ema_slow"]) and
          float(m15["ema_fast"])  < float(m15["ema_slow"])):
        bear += 1; reasons.append("M15 EMA Cross: Bearish")

    # F3: RSI
    rsi_v = float(m15["rsi"])
    if   rsi_v <= RSI_OS:           bull += 1; reasons.append(f"RSI {rsi_v:.1f} — Oversold")
    elif RSI_OS < rsi_v < 55:       bull += 1; reasons.append(f"RSI {rsi_v:.1f} — Bullish momentum")
    elif 45 < rsi_v < RSI_OB:       bear += 1; reasons.append(f"RSI {rsi_v:.1f} — Bearish momentum")
    elif rsi_v >= RSI_OB:           bear += 1; reasons.append(f"RSI {rsi_v:.1f} — Overbought")

    # F4: MACD histogram turn
    hist_v = float(m15["macd_hist"]); hist_p = float(m15p["macd_hist"])
    if   hist_p < 0 and hist_v > hist_p: bull += 1; reasons.append("MACD: Histogram reversing up")
    elif hist_p > 0 and hist_v < hist_p: bear += 1; reasons.append("MACD: Histogram reversing down")

    # F5: Stochastic
    sk, sd = float(m15["stoch_k"]), float(m15["stoch_d"])
    if   sk < 25 and sk > sd: bull += 1; reasons.append(f"Stoch {sk:.0f} — Oversold cross")
    elif sk > 75 and sk < sd: bear += 1; reasons.append(f"Stoch {sk:.0f} — Overbought cross")

    direction = score = None
    if   bull >= MIN_CONF and bull > bear: direction, score = "BUY", bull
    elif bear >= MIN_CONF and bear > bull: direction, score = "SELL", bear
    if not direction:
        _cache["signals"][symbol] = {"data":None,"_ts":now}
        return None

    if direction == "BUY":
        sl  = round(price - atr_v*ATR_SL, 5)
        tp1 = round(price + atr_v*ATR_TP, 5)
        tp2 = round(price + atr_v*ATR_TP*1.8, 5)
    else:
        sl  = round(price + atr_v*ATR_SL, 5)
        tp1 = round(price - atr_v*ATR_TP, 5)
        tp2 = round(price - atr_v*ATR_TP*1.8, 5)

    risk = abs(price-sl); reward = abs(tp1-price)
    rr = round(reward/risk,2) if risk else 0
    conf = ("VERY HIGH 🔥" if score >= 5 else
            "HIGH ✅"      if score >= 4 else "MEDIUM ⚡")

    result = {
        "symbol":symbol,"direction":direction,"entry":price,
        "sl":sl,"tp1":tp1,"tp2":tp2,"rr":rr,
        "confluence_score":score,"max_factors":5,
        "confidence":conf,"reasons":reasons[:5],
        "timeframe":"M15/H1","strategy":"EMA+RSI+MACD+Stoch",
        "rsi":round(rsi_v,1),"atr":round(atr_v,5),
    }
    _cache["signals"][symbol] = {"data":result,"_ts":now}
    _cache["ts"] = now
    return result


# ─── LOT SIZE ─────────────────────────────────────────────────────────────────
def calc_lot(symbol, balance, risk_pct, sl_distance):
    meta = SYMBOLS[symbol]
    risk_amount = balance * risk_pct / 100.0
    if symbol == "XAUUSD":
        lot = risk_amount / (sl_distance * meta["lot_multiplier"])
    else:
        sl_pips = sl_distance / meta["pip_size"]
        lot = risk_amount / (sl_pips * 10.0)
    return max(0.01, round(lot, 2))


# ─── TELEGRAM BOT (background thread) ─────────────────────────────────────────
def start_telegram_bot():
    if not BOT_TOKEN or not CHANNEL_ID:
        log.warning("[BOT] BOT_TOKEN/CHANNEL_ID not set — bot disabled")
        return

    from telegram import Update
    from telegram.ext import Application, CommandHandler, ContextTypes
    from telegram.constants import ParseMode

    _last_sent = {}

    def build_msg(sig):
        meta = SYMBOLS[sig["symbol"]]; d = meta["decimals"]
        arrow = "📈" if sig["direction"]=="BUY" else "📉"
        sl_d = abs(sig["entry"]-sig["sl"])
        rows = []
        for bal in (300,500,1000,2000,5000,10000):
            l1 = calc_lot(sig["symbol"], bal, 1.0, sl_d)
            l2 = calc_lot(sig["symbol"], bal, 2.0, sl_d)
            rows.append(f"  ${bal:>6,}  │ {l1:.2f} │ {l2:.2f}")
        lot_tbl = "  Balance │  1%  │  2%\n  ─────────────────\n" + "\n".join(rows)
        rea = "\n".join(f"   › {r}" for r in sig["reasons"])
        return (
            f"{meta['icon']} *{meta['label']} SIGNAL*\n"
            f"━━━━━━━━━━━━━━━\n"
            f"*Direction:* {sig['direction']} {arrow}\n"
            f"*Entry:*  `{sig['entry']:.{d}f}`\n"
            f"*SL:*     `{sig['sl']:.{d}f}` 🔴\n"
            f"*TP1:*    `{sig['tp1']:.{d}f}` 🟢\n"
            f"*TP2:*    `{sig['tp2']:.{d}f}` 🎯\n"
            f"*R:R:*    1:{sig['rr']}\n"
            f"*Conf:*   {sig['confidence']} ({sig['confluence_score']}/5)\n"
            f"━━━━━━━━━━━━━━━\n"
            f"💰 *LOT SIZE (Exness):*\n`{lot_tbl}`\n"
            f"━━━━━━━━━━━━━━━\n"
            f"📊 *Factors:*\n{rea}\n"
            f"⚡ {DEV}"
        )

    async def scan_job(app):
        log.info("[BOT] scanning…")
        resolve_pending_signals()
        for sym in SYMBOLS:
            sig = analyze_signal(sym)
            if not sig: continue
            key = f"{sym}_{sig['direction']}"
            now_ts = time.time()
            if key in _last_sent and now_ts - _last_sent[key] < 3600: continue
            _last_sent[key] = now_ts
            save_signal(sig)
            try:
                await app.bot.send_message(CHANNEL_ID, build_msg(sig),
                                           parse_mode=ParseMode.MARKDOWN)
                log.info(f"[BOT] sent {sym} {sig['direction']}")
            except Exception as e:
                log.error(f"[BOT SEND] {e}")

    async def cmd_start(update, ctx):
        await update.message.reply_text(
            f"🟢 *ForexSignal Pro*\n\nGold & EUR/USD signals with Exness lot sizing.\n\n"
            f"Commands:\n/signal — scan now\n/stats — performance\n/lot 1000 1 — lot size\n\n⚡ {DEV}",
            parse_mode=ParseMode.MARKDOWN)

    async def cmd_signal(update, ctx):
        w = await update.message.reply_text("⏳ scanning…")
        any_sig = False
        for sym in SYMBOLS:
            sig = analyze_signal(sym)
            if sig:
                await update.message.reply_text(build_msg(sig), parse_mode=ParseMode.MARKDOWN)
                any_sig = True
        if not any_sig:
            await update.message.reply_text("No confluence on either pair right now.")
        try: await w.delete()
        except: pass

    async def cmd_stats(update, ctx):
        s = get_stats()
        await update.message.reply_text(
            f"📊 *Performance*\nTotal: *{s['total']}*\nWins: *{s['wins']}*\n"
            f"Losses: *{s['losses']}*\nWin rate: *{s['win_rate']}%*\n"
            f"Avg pips: *{s['avg_pips']}*\n\n⚡ {DEV}",
            parse_mode=ParseMode.MARKDOWN)

    async def cmd_lot(update, ctx):
        a = ctx.args
        if len(a) < 2:
            await update.message.reply_text("Usage: `/lot <balance> <risk%>`\nEx: `/lot 1000 1`",
                                            parse_mode=ParseMode.MARKDOWN); return
        try:
            bal, risk = float(a[0].replace(",","")), float(a[1])
        except ValueError:
            await update.message.reply_text("❌ Invalid input"); return
        lines = [f"💰 *Lot sizes — ${bal:,.0f} @ {risk}%*", "━━━━━━━━━━━━"]
        for sym, meta in SYMBOLS.items():
            sig = analyze_signal(sym)
            if not sig:
                lines.append(f"\n{meta['icon']} {meta['label']}: no active signal")
                continue
            sl_d = abs(sig["entry"]-sig["sl"])
            lot  = calc_lot(sym, bal, risk, sl_d)
            lines.append(f"\n{meta['icon']} *{meta['label']}* ({sig['direction']})\n"
                         f"   SL dist: {sl_d:.{meta['decimals']}f}\n"
                         f"   ✅ *{lot:.2f} lots*")
        lines.append(f"\n⚡ {DEV}")
        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)

    async def post_init(app):
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        sched = AsyncIOScheduler()
        sched.add_job(scan_job, "interval", minutes=15, args=[app])
        sched.start()
        import asyncio
        asyncio.create_task(scan_job(app))

    def run():
        app = Application.builder().token(BOT_TOKEN).post_init(post_init).build()
        app.add_handler(CommandHandler("start",  cmd_start))
        app.add_handler(CommandHandler("signal", cmd_signal))
        app.add_handler(CommandHandler("stats",  cmd_stats))
        app.add_handler(CommandHandler("lot",    cmd_lot))
        app.run_polling(drop_pending_updates=True)

    threading.Thread(target=run, daemon=True).start()
    log.info("[BOT] thread started")


# ─── FLASK APP ────────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/api/stats")
def api_stats(): return jsonify(get_stats())

@app.route("/api/signals")
def api_signals(): return jsonify(get_signals(30))

@app.route("/api/live")
def api_live():
    out = {}
    for sym in SYMBOLS:
        sig = analyze_signal(sym)
        out[sym] = sig or {"symbol":sym,"direction":"NEUTRAL","entry":0}
    return jsonify(out)

@app.route("/api/prices")
def api_prices():
    return jsonify({s: get_current_price(s) for s in SYMBOLS})

# ─── DASHBOARD HTML (inline) ──────────────────────────────────────────────────
DASHBOARD = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ForexSignal Pro</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{--bg:#080C14;--surf:#0F1623;--card:#161F30;--gold:#C9A227;--grn:#0ECB81;--red:#F6465D;--blue:#3B82F6;--t1:#EDF2FF;--t2:#8FA3C0;--t3:#4B6180;--brd:#1E2D45}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--t1);font-family:Inter,system-ui,sans-serif;font-size:14px;min-height:100vh}
header{background:var(--surf);border-bottom:1px solid var(--brd);padding:0 20px;height:58px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:10}
.logo{display:flex;align-items:center;gap:9px;font-weight:700}.logo b{color:var(--gold)}
.tickers{display:flex;gap:12px}.ticker{background:var(--card);border:1px solid var(--brd);border-radius:16px;padding:4px 12px;font-size:12px;font-family:monospace}
.ticker span:first-child{color:var(--t3);margin-right:6px}
.live{display:flex;align-items:center;gap:5px;font-size:11px;color:var(--t2)}
.dot{width:7px;height:7px;background:var(--grn);border-radius:50%;animation:p 2s infinite}
@keyframes p{50%{opacity:.3}}
main{max-width:1160px;margin:0 auto;padding:24px 18px 60px}
.lbl{font-size:10px;letter-spacing:1.2px;color:var(--t3);text-transform:uppercase;margin:0 0 12px 2px;font-weight:600}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:24px}
.sc{background:var(--card);border:1px solid var(--brd);border-radius:10px;padding:16px 18px;position:relative}
.sc::before{content:'';position:absolute;left:0;top:0;width:3px;height:100%;background:var(--gold);border-radius:10px 0 0 10px}
.sc:nth-child(2)::before{background:var(--grn)}.sc:nth-child(3)::before{background:var(--blue)}.sc:nth-child(4)::before{background:var(--red)}
.sl{font-size:11px;color:var(--t3);margin-bottom:6px}.sv{font-size:24px;font-weight:700;font-family:monospace;line-height:1}
.sig-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:24px}
.sig-card{background:var(--card);border:1px solid var(--brd);border-radius:16px;padding:20px;border-left:3px solid var(--t3)}
.sig-card.buy{border-left-color:var(--grn)}.sig-card.sell{border-left-color:var(--red)}
.sc-top{display:flex;justify-content:space-between;margin-bottom:16px}
.sc-sym{font-size:16px;font-weight:700}.sc-sym small{display:block;font-size:11px;color:var(--t3);font-weight:400;margin-top:2px}
.badge{padding:4px 11px;border-radius:18px;font-size:12px;font-weight:700}
.b-buy{background:#0ECB8120;color:var(--grn)}.b-sell{background:#F6465D20;color:var(--red)}.b-neutral{background:var(--card);color:var(--t3)}
.prices{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:14px}
.pb{background:var(--surf);border:1px solid var(--brd);border-radius:6px;padding:9px 11px}
.pb-l{font-size:10px;color:var(--t3);margin-bottom:3px}.pb-v{font-size:12px;font-weight:600;font-family:monospace}
.pb.tp .pb-v{color:var(--grn)}.pb.sl .pb-v{color:var(--red)}.pb.entry .pb-v{color:var(--gold)}
.chips{display:flex;gap:7px;flex-wrap:wrap}
.chip{background:var(--surf);border:1px solid var(--brd);border-radius:4px;padding:3px 8px;font-size:11px;color:var(--t2);font-family:monospace}
.chip.gold{border-color:#C9A22730;color:var(--gold)}
.cbar{margin-top:13px}.cbar-row{display:flex;justify-content:space-between;font-size:11px;color:var(--t3);margin-bottom:5px}
.track{height:4px;background:var(--surf);border-radius:2px;overflow:hidden}
.fill{height:100%;background:linear-gradient(90deg,var(--gold),var(--grn));transition:width .6s}
.reasons{margin-top:13px;border-top:1px solid var(--brd);padding-top:11px;display:flex;flex-direction:column;gap:4px}
.ri{font-size:11px;color:var(--t2);display:flex;gap:5px}.ri::before{content:'›';color:var(--gold);font-weight:700}
.nm{text-align:center;padding:28px;color:var(--t3);font-size:13px}.nm div:first-child{font-size:26px;margin-bottom:7px}
.bot-grid{display:grid;grid-template-columns:1fr 310px;gap:14px}
.hcard,.chart-card{background:var(--card);border:1px solid var(--brd);border-radius:16px;overflow:hidden}
.ch{padding:14px 18px;border-bottom:1px solid var(--brd);display:flex;justify-content:space-between;align-items:center}
.ct{font-size:13px;font-weight:600}
.rbtn{background:none;border:1px solid var(--brd);border-radius:5px;padding:4px 9px;font-size:11px;color:var(--t2);cursor:pointer}
.rbtn:hover{border-color:var(--gold);color:var(--gold)}
table{width:100%;border-collapse:collapse}
thead th{padding:9px 14px;text-align:left;font-size:10px;color:var(--t3);text-transform:uppercase;background:var(--surf);border-bottom:1px solid var(--brd);font-weight:600}
tbody tr{border-bottom:1px solid var(--brd)}tbody tr:last-child{border-bottom:none}
tbody td{padding:10px 14px;font-size:12px;color:var(--t2)}
.dt{padding:2px 7px;border-radius:3px;font-size:11px;font-weight:700;font-family:monospace}
.dt-buy{background:#0ECB8120;color:var(--grn)}.dt-sell{background:#F6465D20;color:var(--red)}
.ot{padding:2px 6px;border-radius:3px;font-size:10px;font-weight:600}
.oW{background:#0ECB8120;color:var(--grn)}.oL{background:#F6465D20;color:var(--red)}.oP{background:var(--card);color:var(--t3)}
.mono{font-family:monospace}
.empty{padding:36px;text-align:center;color:var(--t3);font-size:13px}
.chart-inner{padding:18px;display:flex;flex-direction:column;gap:18px}
.dw{position:relative;width:150px;height:150px;margin:0 auto}
.dc{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center}
.dc .big{font-size:22px;font-weight:700;font-family:monospace}.dc .small{font-size:10px;color:var(--t3)}
hr.div{border:none;border-top:1px solid var(--brd)}
.legend{display:flex;flex-direction:column;gap:9px}
.lr{display:flex;justify-content:space-between;align-items:center}
.ll{display:flex;align-items:center;gap:7px;font-size:12px;color:var(--t2)}
.ldot{width:8px;height:8px;border-radius:2px}
.lv{font-size:13px;font-weight:600;font-family:monospace}
.spin{width:18px;height:18px;border:2px solid var(--brd);border-top-color:var(--gold);border-radius:50%;animation:s .7s linear infinite;display:inline-block}
@keyframes s{to{transform:rotate(360deg)}}
#toast{position:fixed;bottom:20px;right:20px;background:var(--card);border:1px solid var(--brd);border-radius:10px;padding:11px 16px;font-size:12px;opacity:0;transform:translateY(10px);transition:all .25s;pointer-events:none}
#toast.show{opacity:1;transform:translateY(0)}
@media(max-width:880px){.stats{grid-template-columns:repeat(2,1fr)}.sig-grid,.bot-grid{grid-template-columns:1fr}.tickers{display:none}}
</style></head><body>
<header>
  <div class="logo">📊 Forex<b>Signal</b> Pro</div>
  <div class="tickers">
    <div class="ticker"><span>XAU</span><b id="htG">—</b></div>
    <div class="ticker"><span>EUR</span><b id="htE">—</b></div>
  </div>
  <div class="live"><div class="dot"></div>LIVE</div>
</header>
<main>
  <div class="lbl">Performance</div>
  <div class="stats">
    <div class="sc"><div class="sl">Total Signals</div><div class="sv" id="sTotal">—</div></div>
    <div class="sc"><div class="sl">Win Rate</div><div class="sv" id="sWR" style="color:var(--grn)">—</div></div>
    <div class="sc"><div class="sl">Avg Pips</div><div class="sv" id="sPips" style="color:var(--blue)">—</div></div>
    <div class="sc"><div class="sl">Losses</div><div class="sv" id="sLoss" style="color:var(--red)">—</div></div>
  </div>
  <div class="lbl">Live Signal Analysis</div>
  <div class="sig-grid" id="sigGrid">
    <div class="nm"><div><span class="spin"></span></div>Scanning markets…</div>
    <div class="nm"><div><span class="spin"></span></div>Scanning markets…</div>
  </div>
  <div class="bot-grid">
    <div class="hcard">
      <div class="ch"><span class="ct">Signal History</span><button class="rbtn" onclick="loadAll()">↻ Refresh</button></div>
      <div id="histBody"><div class="empty">Loading…</div></div>
    </div>
    <div class="chart-card">
      <div class="ch"><span class="ct">Win Rate</span></div>
      <div class="chart-inner">
        <div class="dw"><canvas id="dChart"></canvas><div class="dc"><div class="big" id="dLabel">—</div><div class="small">win rate</div></div></div>
        <hr class="div">
        <div class="legend">
          <div class="lr"><div class="ll"><div class="ldot" style="background:var(--grn)"></div>Wins</div><div class="lv" id="lW">—</div></div>
          <div class="lr"><div class="ll"><div class="ldot" style="background:var(--red)"></div>Losses</div><div class="lv" id="lL">—</div></div>
          <div class="lr"><div class="ll"><div class="ldot" style="background:var(--t3)"></div>Pending</div><div class="lv" id="lP">—</div></div>
        </div>
      </div>
    </div>
  </div>
</main>
<div id="toast"></div>
<script>
const REFRESH=30; let donut=null;
const fmt=(v,d=2)=>v==null||isNaN(v)?'—':parseFloat(v).toFixed(d);
const toast=m=>{const e=document.getElementById('toast');e.textContent=m;e.classList.add('show');setTimeout(()=>e.classList.remove('show'),2200)};
async function api(p,fb){try{const r=await fetch(p,{signal:AbortSignal.timeout(9000)});if(!r.ok)throw 0;return await r.json()}catch{return fb}}

function renderStats(s){
  document.getElementById('sTotal').textContent=s.total??0;
  document.getElementById('sWR').textContent=(s.win_rate??0)+'%';
  document.getElementById('sPips').textContent=s.avg_pips>=0?'+'+s.avg_pips:s.avg_pips;
  document.getElementById('sLoss').textContent=s.losses??0;
}
function renderDonut(s){
  const p=Math.max(0,s.total-s.wins-s.losses);
  document.getElementById('dLabel').textContent=s.win_rate+'%';
  document.getElementById('lW').textContent=s.wins;
  document.getElementById('lL').textContent=s.losses;
  document.getElementById('lP').textContent=p;
  const data={datasets:[{data:[s.wins,s.losses,p],backgroundColor:['#0ECB81','#F6465D','#243550'],borderColor:'transparent',borderWidth:0}]};
  if(donut){donut.data=data;donut.update('none')}
  else donut=new Chart(document.getElementById('dChart'),{type:'doughnut',data,options:{cutout:'72%',plugins:{legend:{display:false}}}});
}
function buildCard(sig){
  if(!sig||sig.direction==='NEUTRAL'||!sig.entry){
    const sym=sig?.symbol||'—';const lbl=sym==='XAUUSD'?'GOLD (XAU/USD)':'EUR/USD';
    return `<div class="sig-card"><div class="sc-top"><div class="sc-sym">${lbl}<small>${sym}</small></div><div class="badge b-neutral">NEUTRAL</div></div><div class="nm"><div>📊</div>No confluence — awaiting next scan</div></div>`;
  }
  const{symbol:sym,direction:dir,entry,sl,tp1,tp2,rr,confluence_score:cs,max_factors:mf,confidence:conf,reasons:rs=[]}=sig;
  const lbl=sym==='XAUUSD'?'GOLD (XAU/USD)':'EUR/USD';
  const dc=sym==='XAUUSD'?2:5;
  const cls=dir==='BUY'?'buy':'sell';
  const bcls=dir==='BUY'?'b-buy':'b-sell';
  const pct=Math.round((cs/(mf||5))*100);
  const rea=rs.map(r=>`<div class="ri">${r}</div>`).join('');
  return `<div class="sig-card ${cls}">
    <div class="sc-top"><div class="sc-sym">${lbl}<small>${sym}</small></div><div class="badge ${bcls}">${dir} ${dir==='BUY'?'↑':'↓'}</div></div>
    <div class="prices">
      <div class="pb entry"><div class="pb-l">Entry</div><div class="pb-v">${fmt(entry,dc)}</div></div>
      <div class="pb tp"><div class="pb-l">TP 1</div><div class="pb-v">${fmt(tp1,dc)}</div></div>
      <div class="pb sl"><div class="pb-l">Stop Loss</div><div class="pb-v">${fmt(sl,dc)}</div></div>
    </div>
    <div class="chips">
      <span class="chip">R:R 1:${rr}</span><span class="chip">M15/H1</span>
      <span class="chip gold">${conf}</span><span class="chip">TP2: ${fmt(tp2,dc)}</span>
    </div>
    <div class="cbar"><div class="cbar-row"><span>Confluence</span><strong>${cs}/${mf||5} factors</strong></div>
    <div class="track"><div class="fill" style="width:${pct}%"></div></div></div>
    ${rea?`<div class="reasons">${rea}</div>`:''}
  </div>`;
}
function renderLive(d){document.getElementById('sigGrid').innerHTML=['XAUUSD','EURUSD'].map(s=>buildCard(d[s])).join('')}
function renderHistory(sigs){
  const b=document.getElementById('histBody');
  if(!sigs||!sigs.length){b.innerHTML='<div class="empty">No signals yet — engine will save them automatically</div>';return}
  const rows=sigs.map(s=>{
    const lbl=s.symbol==='XAUUSD'?'🥇 GOLD':'💶 EUR/USD';
    const dc=s.symbol==='XAUUSD'?2:5;
    const dt=`<span class="dt dt-${s.direction.toLowerCase()}">${s.direction}</span>`;
    const oc=s.outcome==='WIN'?'oW':s.outcome==='LOSS'?'oL':'oP';
    const ot=`<span class="ot ${oc}">${s.outcome}</span>`;
    const ts=(s.timestamp||'').slice(0,16).replace('T',' ');
    return `<tr><td style="color:var(--t1);font-weight:500">${lbl}</td><td>${dt}</td>
      <td class="mono">${fmt(s.entry_price,dc)}</td>
      <td class="mono" style="color:var(--grn)">${fmt(s.tp1,dc)}</td>
      <td class="mono" style="color:var(--red)">${fmt(s.sl,dc)}</td>
      <td>${ot}</td><td style="color:var(--t3);font-size:11px">${ts}</td></tr>`;
  }).join('');
  b.innerHTML=`<table><thead><tr><th>Symbol</th><th>Dir</th><th>Entry</th><th>TP1</th><th>SL</th><th>Result</th><th>Time</th></tr></thead><tbody>${rows}</tbody></table>`;
}
function updateTickers(p){
  ['XAUUSD','EURUSD'].forEach(s=>{
    const v=p[s];if(!v)return;
    const el=document.getElementById(s==='XAUUSD'?'htG':'htE');
    if(el)el.textContent=fmt(v,s==='XAUUSD'?2:5);
  });
}
async function loadAll(){
  const [stats,live,hist,prices]=await Promise.all([
    api('/api/stats',{total:0,wins:0,losses:0,win_rate:0,avg_pips:0}),
    api('/api/live',{}),
    api('/api/signals',[]),
    api('/api/prices',{}),
  ]);
  renderStats(stats);renderDonut(stats);renderLive(live);renderHistory(hist);updateTickers(prices);
}
loadAll();
setInterval(loadAll,REFRESH*1000);
</script></body></html>"""

@app.route("/")
def index(): return render_template_string(DASHBOARD)


# ─── BOOT ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    resolve_pending_signals()
    start_telegram_bot()
    log.info(f"🌐 Dashboard → http://0.0.0.0:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)


