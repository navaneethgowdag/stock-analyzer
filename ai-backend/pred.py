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
from pathlib import Path
from datetime import datetime


load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

BASE_DIR = Path(__file__).resolve().parent
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

MODELS_DIR = BASE_DIR / "models"
ENABLE_SENTIMENT = os.environ.get("ENABLE_SENTIMENT", "true").lower() == "true"
MAX_HEADLINES = int(os.environ.get("MAX_HEADLINES", "10"))
print("BASE_DIR:", BASE_DIR)
print("MODELS_DIR:", MODELS_DIR)
print("MODEL EXISTS:", (MODELS_DIR / "portfolio_models_dict.pkl").exists())
EXCHANGE_SUFFIX = {"NSE": ".NS", "BSE": ".BO"}
BUY_THRESHOLD, SELL_THRESHOLD = 0.70, 0.40
WEIGHT_MODEL, WEIGHT_SENTIMENT = 0.70, 0.30
LOOKBACK_PERIOD = "2y"

from urllib.parse import urlparse

if DATABASE_URL:
    parsed = urlparse(DATABASE_URL)
    print("DATABASE HOST:", parsed.hostname)
    print("DATABASE NAME:", parsed.path)
else:
    print("DATABASE_URL is EMPTY")

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
            ALTER TABLE predictions
            ADD COLUMN IF NOT EXISTS previous_close NUMERIC(12, 2);
        """)
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS news_sentiment (
                id          SERIAL PRIMARY KEY,
                ticker      TEXT NOT NULL,
                headline    TEXT NOT NULL,
                label       TEXT,
                score       NUMERIC,
                confidence  NUMERIC,
                updated_at  TIMESTAMPTZ DEFAULT now(),
                UNIQUE (ticker, headline)
            );
        """)
        
    conn.commit()
def ensure_price_history_table(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS price_history (
                id          BIGSERIAL PRIMARY KEY,
                ticker      TEXT NOT NULL,
                price       NUMERIC NOT NULL,
                recorded_at TIMESTAMPTZ DEFAULT now()
            );

            CREATE INDEX IF NOT EXISTS idx_price_history_ticker_time
            ON price_history(ticker, recorded_at DESC);
        """)

    conn.commit()
    
    
def ensure_alerts_table(conn):
    """
    Create the alerts table if it does not already exist.
    """

    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id BIGSERIAL PRIMARY KEY,

                ticker TEXT NOT NULL,

                alert_type TEXT NOT NULL,

                title TEXT NOT NULL,

                message TEXT NOT NULL,

                severity TEXT NOT NULL,

                value NUMERIC,

                reference_id TEXT,

                created_at TIMESTAMPTZ DEFAULT NOW()
            );

            CREATE INDEX IF NOT EXISTS idx_alerts_ticker_created
            ON alerts(ticker, created_at DESC);
        """)

    conn.commit()
def ensure_news_sentiment_table(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS news_sentiment (
                id BIGSERIAL PRIMARY KEY,
                ticker TEXT NOT NULL,
                headline TEXT NOT NULL,
                label TEXT,
                score NUMERIC,
                confidence NUMERIC,
                created_at TIMESTAMPTZ DEFAULT NOW(),

                CONSTRAINT news_sentiment_ticker_headline_unique
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
            """
            INSERT INTO news_sentiment
                (ticker, headline, label, score, confidence, created_at)
            VALUES %s
            ON CONFLICT (ticker, headline)
            DO UPDATE SET
                label = EXCLUDED.label,
                score = EXCLUDED.score,
                confidence = EXCLUDED.confidence,
                created_at = NOW();
            """,
            [
                (
                    ticker,
                    r["headline"],
                    r["label"],
                    r["score"],
                    r["confidence"],
                    datetime.now()
                )
                for r in results
            ],
        )

    conn.commit()
    
def insert_price_history(conn, ticker: str, price: float):
    if price is None or price <= 0:
        return

    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO price_history
                (ticker, price, recorded_at)
            VALUES
                (%s, %s, now())
        """, (ticker, price))

    conn.commit()

def insert_alert(
    conn,
    ticker: str,
    alert_type: str,
    title: str,
    message: str,
    severity: str,
    value=None,
    reference_id=None,
):
    """
    Insert a new alert into the alerts table.

    Duplicate alerts with the same ticker, type and reference
    are avoided when a reference_id is supplied.
    """

    with conn.cursor() as cur:

        if reference_id:

            cur.execute("""
                SELECT id
                FROM alerts
                WHERE ticker = %s
                  AND alert_type = %s
                  AND reference_id = %s
                LIMIT 1;
            """, (
                ticker,
                alert_type,
                reference_id
            ))

            if cur.fetchone():
                return

        cur.execute("""
            INSERT INTO alerts (
                ticker,
                alert_type,
                title,
                message,
                severity,
                value,
                reference_id,
                created_at
            )
            VALUES (
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                NOW()
            );
        """, (
            ticker,
            alert_type,
            title,
            message,
            severity,
            value,
            reference_id
        ))

    conn.commit()
    
def generate_news_alerts(conn, result: dict):
    """
    Generate alerts for strongly positive or negative news.

    FinBERT results are expected to contain:
        headline
        label
        score
        confidence
    """

    ticker = result["ticker"]

    news_results = result.get("news_results", [])

    if not news_results:
        return

    for news in news_results:

        headline = news.get("headline", "").strip()
        label = str(news.get("label", "")).strip().lower()
        score = float(news.get("score", 0) or 0)
        confidence = float(news.get("confidence", 0) or 0)

        if not headline:
            continue

        # Strong positive news
        if label == "positive" and confidence >= 0.80:

            insert_alert(
                conn=conn,
                ticker=ticker,
                alert_type="NEWS",
                title=f"Positive news for {ticker}",
                message=headline,
                severity="POSITIVE",
                value=score,
                reference_id=f"NEWS:{ticker}:{headline}",
            )

        # Strong negative news
        elif label == "negative" and confidence >= 0.80:

            insert_alert(
                conn=conn,
                ticker=ticker,
                alert_type="NEWS",
                title=f"Negative news for {ticker}",
                message=headline,
                severity="NEGATIVE",
                value=score,
                reference_id=f"NEWS:{ticker}:{headline}",
            )
            
            
def generate_price_alert(conn, ticker: str, current_price: float):
    """
    Generate an alert when the current price has moved significantly
    compared with the previous recorded price.

    Threshold:
        +/- 3%
    """

    if current_price is None or current_price <= 0:
        return

    with conn.cursor() as cur:

        cur.execute("""
            SELECT price
            FROM price_history
            WHERE ticker = %s
            ORDER BY recorded_at DESC
            LIMIT 1 OFFSET 1;
        """, (ticker,))

        row = cur.fetchone()

    if not row:
        return

    previous_price = float(row[0])

    if previous_price <= 0:
        return

    change_percent = (
        (current_price - previous_price)
        / previous_price
    ) * 100

    # Ignore normal movements
    if abs(change_percent) < 3:
        return

    if change_percent > 0:

        alert_type = "PRICE_SURGE"
        severity = "POSITIVE"
        title = f"{ticker} price surge"

        message = (
            f"{ticker} increased by "
            f"{change_percent:.2f}% "
            f"from the previous recorded price."
        )

    else:

        alert_type = "PRICE_DROP"
        severity = "NEGATIVE"
        title = f"{ticker} price drop"

        message = (
            f"{ticker} decreased by "
            f"{abs(change_percent):.2f}% "
            f"from the previous recorded price."
        )

    insert_alert(
        conn=conn,
        ticker=ticker,
        alert_type=alert_type,
        title=title,
        message=message,
        severity=severity,
        value=round(change_percent, 2),
        reference_id=None,
    )
    
    
def upsert_prediction(conn, row: dict):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO predictions (ticker, symbol, exchange, current_price, previous_close, prob_up,
                pred_direction, sentiment_label, sentiment_score, combined_score,
                recommendation, updated_at)
            VALUES (%(ticker)s, %(symbol)s, %(exchange)s, %(current_price)s, %(previous_close)s, %(prob_up)s,
                %(pred_direction)s, %(sentiment_label)s, %(sentiment_score)s, %(combined_score)s,
                %(recommendation)s, now())
            ON CONFLICT (ticker) DO UPDATE SET
                current_price = EXCLUDED.current_price,
                previous_close = EXCLUDED.previous_close,
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
    valid_sorted = valid.sort_values("Date")
    previous_close = float(valid_sorted["Close"].iloc[-2]) if len(valid_sorted) >= 2 else None
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
        "previous_close": round(previous_close, 2) if previous_close is not None else None,
        "prob_up": round(prob_up, 4),
        "pred_direction": "UP" if pred_dir == 1 else "DOWN",
        "sentiment_label": sentiment["sentiment_label"],
        "sentiment_score": round(sentiment["avg_score"], 4),
        "combined_score": round(combined, 4),
        "recommendation": rec,
        "news_results": sentiment["results"],
    }
    
def get_user_watchlist(conn):
    """
    Fetch all stocks currently present in the watchlist.
    """

    with conn.cursor() as cur:

        cur.execute("""
            SELECT
                id,
                user_id,
                symbol,
                company_name,
                exchange
            FROM watchlist
            ORDER BY id;
        """)

        rows = cur.fetchall()

    return [
        {
            "id": row[0],
            "user_id": row[1],
            "symbol": row[2],
            "company_name": row[3],
            "exchange": row[4]
        }
        for row in rows
    ]

def run_job():

    log.info("Job started")

    conn = psycopg2.connect(DATABASE_URL)

    try:
        ensure_predictions_table(conn)
        ensure_news_sentiment_table(conn)
        ensure_price_history_table(conn)
        ensure_alerts_table(conn)

        watchlist = get_user_watchlist(conn)

        for entry in watchlist:

            symbol = entry["symbol"]
            exchange = entry["exchange"]

            try:

                result = process_ticker(symbol, exchange)

                if result:

                    # 1. Prediction
                    upsert_prediction(
                        conn,
                        result
                    )

                    # 2. FinBERT news sentiment
                    insert_news_sentiment(
                        conn,
                        result["ticker"],
                        result["news_results"]
                    )
                    
                    # Generate important news alerts
                    generate_news_alerts(
                        conn,
                        result
                    )

                    insert_price_history(
                        conn,
                        result["ticker"],
                        result["current_price"]
                    )

                    # Detect sudden price movement
                    generate_price_alert(
                        conn,
                        result["ticker"],
                        result["current_price"]
                    )
                    
                    # 4. Important news alerts
                    generate_news_alerts(
                        conn,
                        result
                    )

                    # 5. Sudden price movement alerts
                    generate_price_alert(
                        conn,
                        result
                    )

            except Exception as e:

                log.exception(
                    f"Failed processing {symbol}: {e}"
                )

    finally:

        conn.close()

    log.info("Job finished")

if __name__ == "__main__":
    run_job()  # run once immediately on startup
    scheduler = BlockingScheduler()
    scheduler.add_job(run_job, "interval", minutes=30, id="stock_prediction_job")
    log.info("Scheduler started - running every 30 minutes")
    scheduler.start()