import os
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import ta
import joblib
import psycopg2
import psycopg2.extras
from apscheduler.schedulers.blocking import BlockingScheduler
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("stock_pipeline")


DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL or "localhost" in DATABASE_URL:
    print(f"[dotenv] DATABASE_URL currently resolves to: {DATABASE_URL!r}")
    raise SystemExit(
        "FATAL: DATABASE_URL is missing or invalid.\n"
        f"  - Confirm this file exists: {_env_path}\n"
        "  - Confirm it contains exactly: DATABASE_URL=postgresql://user:pass@host/db?sslmode=require\n"
        "  - No quotes, no 'export ', no trailing spaces, no BOM (save as plain UTF-8, not 'UTF-8 with BOM')."
    )

MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(BASE_DIR, "models"))
ENABLE_SENTIMENT = os.environ.get("ENABLE_SENTIMENT", "true").lower() == "true"
MAX_HEADLINES = int(os.environ.get("MAX_HEADLINES", "10"))

EXCHANGE_SUFFIX = {"NSE": ".NS", "BSE": ".BO"}
BUY_THRESHOLD, SELL_THRESHOLD = 0.70, 0.40
WEIGHT_MODEL, WEIGHT_SENTIMENT = 0.70, 0.30
LOOKBACK_PERIOD = "2y"

NIFTY_50_SYMBOLS = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BEL", "BPCL",
    "BHARTIARTL", "BRITANNIA", "CIPLA", "COALINDIA", "DIVISLAB",
    "DRREDDY", "EICHERMOT", "GRASIM", "HCLTECH", "HDFCBANK",
    "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK",
    "ITC", "INDUSINDBK", "INFY", "JSWSTEEL", "KOTAKBANK",
    "LT", "M&M", "MARUTI", "NTPC", "NESTLEIND",
    "ONGC", "POWERGRID", "RELIANCE", "SBILIFE", "SHRIRAMFIN",
    "SBIN", "SUNPHARMA", "TCS", "TATACONSUM", "TATAMOTORS",
    "TATASTEEL", "TECHM", "TITAN", "TRENT", "ULTRACEMCO", "WIPRO",
]

# ---------------------------------------------------------------- artifacts
portfolio_models = joblib.load(os.path.join(MODELS_DIR, "portfolio_models_dict.pkl"))
feature_columns = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))
stock_encoder = joblib.load(os.path.join(MODELS_DIR, "stock_encoder.pkl"))

finbert = None
if ENABLE_SENTIMENT:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from torch.nn.functional import softmax

    class FinBERTAnalyzer:
        LABELS = {0: "positive", 1: "negative", 2: "neutral"}
        SCORES = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}

        def __init__(self, model_name="ProsusAI/finbert"):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tok = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()

        def analyze_headline(self, headline):
            inputs = self.tok(headline, return_tensors="pt", truncation=True, max_length=512, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self.model(**inputs).logits
            probs = softmax(logits, dim=-1).cpu().numpy()[0]
            idx = int(np.argmax(probs))
            label = self.LABELS[idx]
            return {"headline": headline, "label": label, "score": self.SCORES[label], "confidence": float(probs[idx])}

        def analyze_batch(self, headlines):
            if not headlines:
                return {"avg_score": 0.0, "sentiment_label": "Neutral", "results": []}
            results = [self.analyze_headline(h) for h in headlines]
            avg = float(np.mean([r["score"] for r in results]))
            label = "Positive" if avg > 0.1 else "Negative" if avg < -0.1 else "Neutral"
            return {"avg_score": avg, "sentiment_label": label, "results": results}

    finbert = FinBERTAnalyzer()

def create_stationary_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Date").copy()
    close, open_, high, low, volume = df["Close"], df["Open"], df["High"], df["Low"], df["Volume"]
    prev_close = close.shift(1)

    df["Ret_1"] = close.pct_change(1)
    df["Ret_3"] = close.pct_change(3)
    df["Ret_5"] = close.pct_change(5)
    df["Ret_10"] = close.pct_change(10)
    df["Ret_20"] = close.pct_change(20)
    df["LogRet_1"] = np.log(close / prev_close)
    df["Vol_Chg_1"] = volume.pct_change(1)
    df["Vol_Chg_5"] = volume.pct_change(5)
    df["Gap_Return"] = open_ / prev_close - 1.0
    df["Intraday_Return"] = close / open_ - 1.0
    df["Range_Pct"] = (high - low) / close
    df["Body_Pct"] = (close - open_) / open_
    df["Upper_Shadow_Pct"] = (high - np.maximum(open_, close)) / close
    df["Lower_Shadow_Pct"] = (np.minimum(open_, close) - low) / close

    for i in range(1, 11):
        df[f"Ret_Lag_{i}"] = df["Ret_1"].shift(i)
        df[f"Vol_Chg_Lag_{i}"] = df["Vol_Chg_1"].shift(i)
        df[f"Range_Lag_{i}"] = df["Range_Pct"].shift(i)

    for w in [5, 10, 20, 50]:
        price_mean, price_std = close.rolling(w).mean(), close.rolling(w).std()
        vol_mean = volume.rolling(w).mean()
        ret_mean, ret_std = df["Ret_1"].rolling(w).mean(), df["Ret_1"].rolling(w).std()
        rolling_high, rolling_low = high.rolling(w).max(), low.rolling(w).min()

        df[f"Mom_{w}"] = close.pct_change(w)
        df[f"Ret_Mean_{w}"] = ret_mean
        df[f"Ret_Std_{w}"] = ret_std
        df[f"Price_Z_{w}"] = (close - price_mean) / price_std
        df[f"Dist_SMA_{w}"] = close / price_mean - 1.0
        df[f"Dist_High_{w}"] = close / rolling_high - 1.0
        df[f"Dist_Low_{w}"] = close / rolling_low - 1.0
        df[f"Vol_Z_{w}"] = (volume - vol_mean) / volume.rolling(w).std()
        df[f"Vol_Ratio_{w}"] = volume / vol_mean - 1.0

    rsi = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    df["RSI_14_N"] = (rsi - 50.0) / 50.0
    macd = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    df["MACD_N"] = macd.macd() / close
    df["MACD_SIGNAL_N"] = macd.macd_signal() / close
    df["MACD_DIFF_N"] = macd.macd_diff() / close
    bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    df["BB_PBAND"] = bb.bollinger_pband()
    df["BB_WBAND_N"] = bb.bollinger_wband() / 100.0
    atr = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14)
    df["ATR_N"] = atr.average_true_range() / close
    stoch = ta.momentum.StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
    df["STOCH_K_N"] = stoch.stoch() / 100.0
    df["STOCH_D_N"] = stoch.stoch_signal() / 100.0
    adx = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14)
    df["ADX_N"] = adx.adx() / 100.0
    mfi = ta.volume.MFIIndicator(high=high, low=low, close=close, volume=volume, window=14)
    df["MFI_N"] = mfi.money_flow_index() / 100.0
    obv = ta.volume.OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
    df["OBV_Slope_10"] = obv.diff(10) / (volume.rolling(20).mean() * 10)

    dow, month = df["Date"].dt.dayofweek, df["Date"].dt.month
    df["DOW_SIN"] = np.sin(2 * np.pi * dow / 7)
    df["DOW_COS"] = np.cos(2 * np.pi * dow / 7)
    df["MONTH_SIN"] = np.sin(2 * np.pi * (month - 1) / 12)
    df["MONTH_COS"] = np.cos(2 * np.pi * (month - 1) / 12)
    return df


def compute_hybrid_score(prob_up, sentiment_avg):
    sentiment_norm = (sentiment_avg + 1.0) / 2.0
    combined = float(np.clip(WEIGHT_MODEL * prob_up + WEIGHT_SENTIMENT * sentiment_norm, 0.0, 1.0))
    rec = "BUY" if combined >= BUY_THRESHOLD else "SELL" if combined <= SELL_THRESHOLD else "HOLD"
    return combined, rec


def yahoo_ticker(symbol: str, exchange: str) -> str:
    """'TCS.NSE', exchange='NSE' -> 'TCS.NS'"""
    base = symbol.split(".")[0].upper()
    suffix = EXCHANGE_SUFFIX.get((exchange or "").upper(), ".NS")
    return base + suffix


def fetch_news_headlines(ticker_obj, max_headlines=10):
    try:
        news = ticker_obj.news or []
        heads = []
        for item in news[:max_headlines]:
            content = item.get("content", item)
            title = content.get("title") or content.get("headline") or content.get("summary") or ""
            if title:
                heads.append(title.strip())
        return heads
    except Exception:
        return []


def get_nifty50_watchlist():
    """Static Nifty-50 list -> [{'symbol': ..., 'exchange': 'NSE'}, ...]"""
    return [{"symbol": s, "exchange": "NSE"} for s in NIFTY_50_SYMBOLS]


def ensure_predictions_table(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                ticker            TEXT PRIMARY KEY,
                symbol            TEXT,
                exchange          TEXT,
                current_price     NUMERIC,
                prob_up           NUMERIC,
                pred_direction    TEXT,
                sentiment_label   TEXT,
                sentiment_score   NUMERIC,
                combined_score    NUMERIC,
                recommendation    TEXT,
                updated_at        TIMESTAMPTZ DEFAULT now()
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS news_sentiment (
                id          SERIAL PRIMARY KEY,
                ticker      TEXT NOT NULL,
                headline    TEXT NOT NULL,
                label       TEXT,
                score       NUMERIC,
                confidence  NUMERIC,
                created_at  TIMESTAMPTZ DEFAULT now(),
                UNIQUE (ticker, headline)
            );
        """)
    conn.commit()


def insert_news_sentiment(conn, ticker: str, results: list):
    if not results:
        return
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """INSERT INTO news_sentiment (ticker, headline, label, score, confidence)
               VALUES %s ON CONFLICT (ticker, headline) DO NOTHING;""",
            [(ticker, r["headline"], r["label"], r["score"], r["confidence"]) for r in results],
        )
    conn.commit()


def upsert_prediction(conn, row: dict):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO predictions (ticker, symbol, exchange, current_price, prob_up,
                pred_direction, sentiment_label, sentiment_score, combined_score,
                recommendation, updated_at)
            VALUES (%(ticker)s, %(symbol)s, %(exchange)s, %(current_price)s, %(prob_up)s,
                %(pred_direction)s, %(sentiment_label)s, %(sentiment_score)s, %(combined_score)s,
                %(recommendation)s, now())
            ON CONFLICT (ticker) DO UPDATE SET
                current_price = EXCLUDED.current_price,
                prob_up = EXCLUDED.prob_up,
                pred_direction = EXCLUDED.pred_direction,
                sentiment_label = EXCLUDED.sentiment_label,
                sentiment_score = EXCLUDED.sentiment_score,
                combined_score = EXCLUDED.combined_score,
                recommendation = EXCLUDED.recommendation,
                updated_at = now();
        """, row)
    conn.commit()


def process_ticker(symbol: str, exchange: str) -> dict | None:
    yt = yahoo_ticker(symbol, exchange)
    if yt not in portfolio_models:
        log.warning(f"No trained model for {yt}, skipping")
        return None

    hist = yf.download(yt, period=LOOKBACK_PERIOD, interval="1d", progress=False, auto_adjust=False)
    if hist.empty or len(hist) < 60:
        log.warning(f"Insufficient OHLCV data for {yt}")
        return None

    hist = hist.reset_index()
    hist.columns = [c[0] if isinstance(c, tuple) else c for c in hist.columns]
    hist = hist.rename(columns={"index": "Date"})[["Date", "Open", "High", "Low", "Close", "Volume"]]
    hist["Date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None)

    eng = create_stationary_features(hist)
    try:
        eng["Stock_ID"] = stock_encoder.transform([yt])[0]
    except ValueError:
        eng["Stock_ID"] = 0

    eng.replace([np.inf, -np.inf], np.nan, inplace=True)
    valid = eng.dropna(subset=feature_columns)
    if valid.empty:
        log.warning(f"No valid feature rows for {yt}")
        return None

    latest = valid.sort_values("Date").iloc[[-1]]
    last_close = float(latest["Close"].values[0])
    X_in = latest[feature_columns]

    model = portfolio_models[yt]
    prob_up = float(model.predict_proba(X_in)[0][1])
    pred_dir = int(model.predict(X_in)[0])

    sentiment = {"avg_score": 0.0, "sentiment_label": "Neutral", "results": []}
    ticker_obj = yf.Ticker(yt)
    if finbert is not None:
        headlines = fetch_news_headlines(ticker_obj, max_headlines=MAX_HEADLINES)
        sentiment = finbert.analyze_batch(headlines)

    combined, rec = compute_hybrid_score(prob_up, sentiment["avg_score"])

    try:
        price = float(ticker_obj.fast_info.last_price)
        if np.isnan(price) or price <= 0:
            raise ValueError
    except Exception:
        price = last_close

    return {
        "ticker": yt,
        "symbol": symbol,
        "exchange": exchange,
        "current_price": round(price, 2),
        "prob_up": round(prob_up, 4),
        "pred_direction": "UP" if pred_dir == 1 else "DOWN",
        "sentiment_label": sentiment["sentiment_label"],
        "sentiment_score": round(sentiment["avg_score"], 4),
        "combined_score": round(combined, 4),
        "recommendation": rec,
        "news_results": sentiment["results"],
    }


def run_job():
    log.info("Job started")
    conn = psycopg2.connect(DATABASE_URL)
    try:
        ensure_predictions_table(conn)
        watchlist = get_nifty50_watchlist()
        log.info(f"Processing {len(watchlist)} Nifty-50 symbols")

        for entry in watchlist:
            symbol, exchange = entry["symbol"], entry["exchange"]
            try:
                result = process_ticker(symbol, exchange)
                if result:
                    upsert_prediction(conn, result)          # push this stock's prediction
                    insert_news_sentiment(conn, result["ticker"], result["news_results"])  # push its FinBERT news
                    log.info(f"{result['ticker']}: {result['recommendation']} (score={result['combined_score']})")
            except Exception as e:
                log.exception(f"Failed processing {symbol}: {e}")
    finally:
        conn.close()
    log.info("Job finished")


if __name__ == "__main__":
    run_job()  # run once immediately on startup
    scheduler = BlockingScheduler()
    scheduler.add_job(run_job, "interval", minutes=30, id="stock_prediction_job")
    log.info("Scheduler started - running every 30 minutes")
    scheduler.start()