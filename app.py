# ============================================================
# TRADING INTELLIGENCE PLATFORM
# Single-file Streamlit app — enter any ticker, get full
# analysis: EDA → Features → ML → Strategy → Backtest →
# Final BUY / HOLD / SELL verdict
#
# Run:  streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble      import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import accuracy_score, confusion_matrix
from statsmodels.tsa.stattools import adfuller
from scipy.stats import norm as sci_norm
import warnings, datetime
import urllib.request as _urllib_req
import json as _json_lib
warnings.filterwarnings("ignore")

# ─── Sentiment / News imports (graceful fallback if missing) ───
try:
    from textblob import TextBlob
    _TEXTBLOB_OK = True
except ImportError:
    _TEXTBLOB_OK = False

try:
    import urllib.request, json as _json
    _URLLIB_OK = True
except ImportError:
    _URLLIB_OK = False

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trading Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: #07090f;
    color: #dde3f0;
}
.stApp { background: #07090f; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0c0f1d 0%,#0f1628 100%);
    border-right: 1px solid #1a2035;
}
[data-testid="stSidebar"] * { color: #8b9bbf !important; }

/* KPI cards */
[data-testid="metric-container"] {
    background: linear-gradient(135deg,#0d1225 0%,#111b35 100%);
    border: 1px solid #1e2d52;
    border-radius: 14px;
    padding: 18px 20px 14px;
    box-shadow: 0 2px 20px rgba(0,0,0,.5);
    transition: border-color .25s, transform .15s;
}
[data-testid="metric-container"]:hover {
    border-color: #4f8ef7;
    transform: translateY(-2px);
}
[data-testid="stMetricLabel"] {
    color: #6b7fa8 !important;
    font-size: 10px !important;
    letter-spacing: .1em;
    text-transform: uppercase;
    font-family: 'Space Mono', monospace !important;
}
[data-testid="stMetricValue"] {
    color: #e8eeff !important;
    font-size: 24px !important;
    font-weight: 700 !important;
    font-family: 'Space Mono', monospace !important;
}
[data-testid="stMetricDelta"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
}

h1,h2,h3 { font-family: 'Space Mono', monospace !important; }
h1 { color: #4f8ef7 !important; font-size: 28px !important; letter-spacing: -.02em; }
h2 { color: #c8d4f0 !important; font-size: 18px !important; border-bottom: 1px solid #1a2540; padding-bottom: 6px; }
h3 { color: #6b7fa8 !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: .12em; }

[data-baseweb="tab-list"] { background: #0c1225; border-radius:10px; border:1px solid #1a2540; gap:4px; padding:4px; }
[data-baseweb="tab"]      { color:#4a5a80 !important; font-family:'Space Mono',monospace; font-size:11px; padding:8px 16px; border-radius:7px; }
[data-baseweb="tab"][aria-selected="true"] { color:#4f8ef7 !important; background:#111b35 !important; }

.stButton>button {
    background: linear-gradient(135deg,#1a6ef7,#1247c7);
    color:#fff; border:none; border-radius:10px;
    padding:12px 28px; font-family:'Space Mono',monospace;
    font-size:11px; letter-spacing:.06em; font-weight:700;
    box-shadow:0 4px 18px rgba(26,110,247,.35);
    transition:all .2s;
}
.stButton>button:hover { transform:translateY(-2px); box-shadow:0 7px 24px rgba(26,110,247,.5); }

/* Verdict boxes */
.verdict-buy  { background:rgba(20,210,120,.1); border:2px solid #14d278; border-radius:16px; padding:24px 30px; text-align:center; }
.verdict-sell { background:rgba(240,60,60,.1);  border:2px solid #f03c3c; border-radius:16px; padding:24px 30px; text-align:center; }
.verdict-hold { background:rgba(245,170,30,.1); border:2px solid #f5aa1e; border-radius:16px; padding:24px 30px; text-align:center; }
.verdict-title { font-family:'Space Mono',monospace; font-size:32px; font-weight:700; }
.verdict-sub   { font-size:14px; color:#8b9bbf; margin-top:8px; line-height:1.6; }

/* Info box */
.info-box { background:#0d1428; border:1px solid #1e2d52; border-radius:12px; padding:16px 20px; margin:8px 0; font-size:13px; }

/* Counter badge */
.counter-badge {
    background: linear-gradient(135deg,#0d1225,#111b35);
    border: 1px solid #1e2d52;
    border-radius: 10px;
    padding: 10px 14px;
    text-align: center;
    margin-top: 10px;
}
.counter-number {
    font-family: 'Space Mono', monospace;
    font-size: 22px;
    font-weight: 700;
    color: #4f8ef7;
}
.counter-label {
    font-family: 'Space Mono', monospace;
    font-size: 9px;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: #4a5a80;
    margin-top: 2px;
}

/* Sentiment panel */
.sentiment-panel { background:linear-gradient(135deg,#0c1225,#0e1830); border:1px solid #1e2d52; border-radius:14px; padding:20px 24px; margin:10px 0; }
.sentiment-score-positive { color:#14d278 !important; font-family:'Space Mono',monospace; font-size:28px; font-weight:700; }
.sentiment-score-negative { color:#f03c3c !important; font-family:'Space Mono',monospace; font-size:28px; font-weight:700; }
.sentiment-score-neutral  { color:#f5aa1e !important; font-family:'Space Mono',monospace; font-size:28px; font-weight:700; }
.news-headline { background:#0a0f1f; border-left:3px solid #1e2d52; border-radius:0 8px 8px 0; padding:10px 14px; margin:5px 0; font-size:12px; line-height:1.5; color:#8b9bbf; }
.news-headline:hover { border-left-color:#4f8ef7; color:#c8d4f0; }
.sentiment-bar-wrap { background:#0c1225; border-radius:8px; height:10px; overflow:hidden; margin:8px 0; }
.sentiment-bar-positive { background:linear-gradient(90deg,#065e30,#14d278); height:10px; border-radius:8px; transition:width .4s; }
.sentiment-bar-negative { background:linear-gradient(90deg,#6b1111,#f03c3c); height:10px; border-radius:8px; transition:width .4s; }

hr { border-color:#1a2540 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# PLOTLY DARK THEME
# ─────────────────────────────────────────────────────────
PT = dict(
    paper_bgcolor="#07090f", plot_bgcolor="#0c1020",
    font=dict(family="Space Mono, monospace", color="#6b7fa8", size=10),
    xaxis=dict(gridcolor="#111b35", zerolinecolor="#111b35", showgrid=True),
    yaxis=dict(gridcolor="#111b35", zerolinecolor="#111b35", showgrid=True),
    margin=dict(l=55, r=25, t=48, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8b9bbf")),
)

def pt_layout(**overrides):
    """Merge PT base theme with overrides, safely merging nested dicts like yaxis/xaxis."""
    layout = dict(PT)
    for k, v in overrides.items():
        if k in layout and isinstance(layout[k], dict) and isinstance(v, dict):
            layout[k] = {**layout[k], **v}
        else:
            layout[k] = v
    return layout

# ─────────────────────────────────────────────────────────
# UPSTASH REDIS — GLOBAL ANALYSE COUNTER
# ─────────────────────────────────────────────────────────

def _upstash_incr() -> int | None:
    """Increment the global analyse counter. Returns new count or None on error."""
    try:
        base_url = st.secrets["UPSTASH_REDIS_REST_URL"].rstrip("/")
        token    = st.secrets["UPSTASH_REDIS_REST_TOKEN"]
        url      = f"{base_url}/incr/analyse_count"
        req = _urllib_req.Request(
            url,
            method="POST",
            headers={"Authorization": f"Bearer {token}"},
        )
        with _urllib_req.urlopen(req, timeout=4) as resp:
            data = _json_lib.loads(resp.read())
        return data.get("result")
    except Exception:
        return None


def _upstash_get() -> int | None:
    """Read the current counter value."""
    try:
        base_url = st.secrets["UPSTASH_REDIS_REST_URL"].rstrip("/")
        token    = st.secrets["UPSTASH_REDIS_REST_TOKEN"]
        url      = f"{base_url}/get/analyse_count"
        req = _urllib_req.Request(
            url,
            headers={"Authorization": f"Bearer {token}"},
        )
        with _urllib_req.urlopen(req, timeout=4) as resp:
            data = _json_lib.loads(resp.read())
        v = data.get("result")
        return int(v) if v is not None else 0
    except Exception:
        return None

# ─────────────────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────────────────
def _rsi(s, w=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(com=w-1, min_periods=w).mean()
    l = (-d.clip(upper=0)).ewm(com=w-1, min_periods=w).mean()
    return 100 - 100/(1 + g/l)

def clean_df(df):
    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in df.select_dtypes(include=[np.number]).columns:
        q1, q3 = df[col].quantile(.25), df[col].quantile(.75)
        iqr = q3 - q1
        df[col] = df[col].clip(lower=q1-10*iqr, upper=q3+10*iqr)
    df.fillna(df.median(numeric_only=True), inplace=True)
    return df

def build_features(df):
    df = df.copy()
    # ── Guard: flatten any MultiIndex columns yfinance may produce ──────────
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # Ensure Close and Volume are 1-D Series, not single-column DataFrames
    df["Close"]  = df["Close"].squeeze()
    df["Volume"] = df["Volume"].squeeze()
    # ────────────────────────────────────────────────────────────────────────
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["SMA_5"]   = df["Close"].rolling(5).mean()
    df["SMA_20"]  = df["Close"].rolling(20).mean()
    df["SMA_50"]  = df["Close"].rolling(50).mean()
    df["EMA_12"]  = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"]  = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]        = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]
    df["RSI_14"]  = _rsi(df["Close"])
    bb = df["Close"].rolling(20).std()
    df["BB_Upper"] = df["SMA_20"] + 2*bb
    df["BB_Lower"] = df["SMA_20"] - 2*bb
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["SMA_20"]
    df["BB_PctB"]  = (df["Close"]   - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"])
    df["Price_to_SMA20"]   = df["Close"] / df["SMA_20"]
    df["Price_to_SMA50"]   = df["Close"] / df["SMA_50"]
    df["SMA5_SMA20_Cross"] = df["SMA_5"] - df["SMA_20"]
    df["Lag_Return_1"] = df["Log_Return"].shift(1)
    df["Lag_Return_2"] = df["Log_Return"].shift(2)
    df["Lag_Return_5"] = df["Log_Return"].shift(5)
    df["RolVol_10"]    = df["Log_Return"].rolling(10).std() * np.sqrt(252)
    df["RolVol_30"]    = df["Log_Return"].rolling(30).std() * np.sqrt(252)
    df["Vol_Ratio"]    = df["RolVol_10"] / df["RolVol_30"]
    df["Volume_Change_Pct"] = df["Volume"].pct_change() * 100
    df["Volume_MA_20"]      = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"]      = df["Volume"] / df["Volume_MA_20"]
    df["Next_Return"] = df["Log_Return"].shift(-1)
    df["Target"]      = (df["Next_Return"] > 0).astype(int)
    # Sentiment placeholder — filled with 0 for historical rows;
    # the pipeline injects the real score only into the latest feature vector.
    df["News_Sentiment"] = 0.0
    df = clean_df(df)
    return df.dropna()

FEATURES = [
    "Price_to_SMA20","Price_to_SMA50","SMA5_SMA20_Cross",
    "MACD","MACD_Signal","MACD_Hist","RSI_14","BB_Width","BB_PctB",
    "Lag_Return_1","Lag_Return_2","Lag_Return_5",
    "RolVol_10","RolVol_30","Vol_Ratio","Volume_Change_Pct","Volume_Ratio",
    "News_Sentiment",   # ← real-time sentiment injected for latest row; 0 for history
]

def sharpe(arr, rf=0.065):
    r = np.diff(arr)/arr[:-1]
    return (r.mean()-rf/252)/(r.std()+1e-9)*np.sqrt(252)

def sortino(arr, rf=0.065):
    r = np.diff(arr)/arr[:-1]
    exc = r - rf/252
    ds  = exc[exc<0]
    return (exc.mean()/(np.sqrt(np.mean(ds**2))+1e-9))*np.sqrt(252)

def max_dd(arr):
    rm = np.maximum.accumulate(arr)
    return ((arr-rm)/rm*100).min()

def cagr(arr, dates):
    y = (dates[-1]-dates[0]).days/365.25
    return ((arr[-1]/arr[0])**(1/y)-1)*100

def calmar(arr, dates):
    return cagr(arr,dates)/abs(max_dd(arr)+1e-9)

def win_rate(tr):
    if not tr: return 0.0
    return sum(1 for r in tr if r>0)/len(tr)*100

def fmt_inr(v):
    if abs(v)>=1e7: return f"₹{v/1e7:.2f} Cr"
    if abs(v)>=1e5: return f"₹{v/1e5:.2f} L"
    return f"₹{v:,.0f}"

# ─────────────────────────────────────────────────────────
# NEWS & SENTIMENT FUNCTIONS
# ─────────────────────────────────────────────────────────

def _clean_ticker_for_search(ticker: str) -> str:
    """Strip exchange suffix for cleaner news queries (RELIANCE.NS → RELIANCE)."""
    return ticker.split(".")[0].upper()


@st.cache_data(show_spinner=False, ttl=900)          # 15-min cache
def fetch_news_headlines(ticker: str, max_headlines: int = 8) -> list[dict]:
    """
    Fetch recent news headlines using 4 sources in order.
    Falls back to next source if one fails.
    Returns list of dicts: [{"title", "source", "url"}, ...]
    """
    import urllib.parse, re
    clean     = _clean_ticker_for_search(ticker)
    full_name = ticker.replace(".NS","").replace(".BO","")
    results: list[dict] = []

    # ── Source 1: Yahoo Finance v2 (more stable endpoint) ──────────────────
    if not results:
        try:
            url = (
                f"https://query2.finance.yahoo.com/v1/finance/search"
                f"?q={urllib.parse.quote(clean)}&lang=en-US&region=US"
                f"&quotesCount=0&newsCount={max_headlines}&enableFuzzyQuery=false"
            )
            headers = {
                "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                               "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"),
                "Accept": "application/json",
                "Referer": "https://finance.yahoo.com",
            }
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = _json.loads(resp.read().decode())
            for item in data.get("news", [])[:max_headlines]:
                title = item.get("title", "").strip()
                if title:
                    results.append({
                        "title":  title,
                        "source": item.get("publisher", "Yahoo Finance"),
                        "url":    item.get("link", "#"),
                    })
        except Exception:
            pass

    # ── Source 2: Yahoo Finance RSS feed ───────────────────────────────────
    if not results:
        try:
            rss = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={urllib.parse.quote(ticker)}&region=US&lang=en-US"
            req = urllib.request.Request(
                rss, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                xml = resp.read().decode("utf-8", errors="ignore")
            # parse <item><title> blocks
            items = re.findall(r"<item>(.*?)</item>", xml, re.DOTALL)
            for item in items[:max_headlines]:
                t = re.search(r"<title>(.*?)</title>", item)
                l = re.search(r"<link>(.*?)</link>",   item)
                if t:
                    results.append({
                        "title":  re.sub(r"<[^>]+>","", t.group(1)).strip(),
                        "source": "Yahoo Finance RSS",
                        "url":    l.group(1).strip() if l else "#",
                    })
        except Exception:
            pass

    # ── Source 3: Google News RSS ──────────────────────────────────────────
    if not results:
        try:
            q   = urllib.parse.quote(f"{full_name} stock share price")
            rss = f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
            req = urllib.request.Request(
                rss, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                xml = resp.read().decode("utf-8", errors="ignore")
            # CDATA style
            titles = re.findall(r"<title><!\[CDATA\[(.+?)\]\]></title>", xml)
            if not titles:
                titles = re.findall(r"<title>(?!Google)(.+?)</title>", xml)
            links  = re.findall(r"<link>(.+?)</link>", xml)
            for i, t in enumerate(titles[:max_headlines]):
                results.append({
                    "title":  t.strip(),
                    "source": "Google News",
                    "url":    links[i].strip() if i < len(links) else "#",
                })
        except Exception:
            pass

    # ── Source 4: Finviz (US tickers only) ────────────────────────────────
    if not results and "." not in ticker:   # US ticker = no dot suffix
        try:
            url = f"https://finviz.com/quote.ashx?t={clean}"
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0",
                "Accept-Language": "en-US,en;q=0.9",
            })
            with urllib.request.urlopen(req, timeout=8) as resp:
                html = resp.read().decode("utf-8", errors="ignore")
            headlines = re.findall(
                r'class="news-link-cell"[^>]*>.*?<a[^>]+href="([^"]+)"[^>]*>([^<]+)</a>',
                html, re.DOTALL
            )
            for url_h, title in headlines[:max_headlines]:
                results.append({
                    "title":  title.strip(),
                    "source": "Finviz",
                    "url":    url_h,
                })
        except Exception:
            pass

    return results[:max_headlines]


def analyse_sentiment(headlines: list[dict]) -> dict:
    """
    Compute an aggregate sentiment score in [-1, +1] from *headlines*.
    """
    if not headlines:
        return {"score": 0.0, "label": "Neutral", "scores": [], "method": "N/A"}

    texts = [h["title"] for h in headlines]

    if _TEXTBLOB_OK:
        scores = [TextBlob(t).sentiment.polarity for t in texts]
        method = "TextBlob"
    else:
        POS = {"surge","soar","jump","gain","rally","beat","strong","high",
               "profit","buy","bullish","upgrade","record","growth","rise",
               "positive","optimistic","outperform","boost","recover"}
        NEG = {"crash","drop","fall","loss","miss","weak","low","sell","bearish",
               "downgrade","decline","negative","pessimistic","underperform",
               "cut","risk","concern","debt","fraud","warning","recession"}
        scores = []
        for t in texts:
            words = set(t.lower().split())
            p = len(words & POS)
            n = len(words & NEG)
            total = p + n
            scores.append((p - n) / total if total else 0.0)
        method = "Keyword"

    agg = float(np.mean(scores)) if scores else 0.0
    if   agg >  0.05: label = "Positive"
    elif agg < -0.05: label = "Negative"
    else:             label = "Neutral"

    return {"score": agg, "label": label, "scores": scores, "method": method}

# ─────────────────────────────────────────────────────────
# CACHED PIPELINE
# ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_pipeline(ticker, start_str, end_str, initial_cap,
                 trans_cost, buy_thresh, sell_thresh, train_pct, rf_rate):
    import yfinance as yf

    # ── Helper: flatten and normalise any DataFrame yfinance returns ─────────
    def _normalise(d):
        if d is None or (hasattr(d, "empty") and d.empty):
            return None
        d = d.copy()
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = [col[0] if isinstance(col, tuple) else col
                         for col in d.columns]
        d = d.loc[:, ~d.columns.duplicated()]
        d.columns = [c.strip().title() for c in d.columns]
        d.index = pd.to_datetime(d.index)
        if hasattr(d.index, "tz") and d.index.tz is not None:
            d.index = d.index.tz_localize(None)
        d.index.name = "Date"
        for col in list(d.columns):
            try:
                arr = np.array(d[col]).flatten()
                if len(arr) == len(d):
                    d[col] = arr.astype(float)
            except Exception:
                pass
        return d

    # 1. Download ─────────────────────────────────────────────────────────────
    raw, _errs = None, []

    try:
        _d = yf.Ticker(ticker).history(
            start=start_str, end=end_str, auto_adjust=True
        )
        _d = _normalise(_d)
        if _d is not None and not _d.empty and "Close" in _d.columns:
            raw = _d
    except Exception as e:
        _errs.append(f"A:{e}")

    if raw is None:
        try:
            _d = yf.download(
                ticker, start=start_str, end=end_str,
                progress=False, auto_adjust=True,
                multi_level_index=False
            )
            _d = _normalise(_d)
            if _d is not None and not _d.empty and "Close" in _d.columns:
                raw = _d
        except Exception as e:
            _errs.append(f"B:{e}")

    if raw is None:
        try:
            _d = yf.download(
                ticker, start=start_str, end=end_str,
                progress=False, auto_adjust=True
            )
            _d = _normalise(_d)
            if _d is not None and not _d.empty and "Close" in _d.columns:
                raw = _d
        except Exception as e:
            _errs.append(f"C:{e}")

    if raw is None:
        try:
            _d = yf.Ticker(ticker).history(period="max", auto_adjust=True)
            _d = _normalise(_d)
            if _d is not None and not _d.empty and "Close" in _d.columns:
                _d = _d[
                    (_d.index >= pd.Timestamp(start_str)) &
                    (_d.index <= pd.Timestamp(end_str))
                ]
                if not _d.empty:
                    raw = _d
        except Exception as e:
            _errs.append(f"D:{e}")

    if raw is None or raw.empty or "Close" not in raw.columns:
        dbg = " | ".join(_errs) if _errs else "no error info"
        return None, (
            f"No data found for **{ticker}**.\n\n"
            f"**Try these NSE tickers:** RELIANCE.NS · TCS.NS · INFY.NS · "
            f"HDFCBANK.NS · SBIN.NS · WIPRO.NS\n\n"
            f"**Try these US tickers:** AAPL · MSFT · TSLA · NVDA · GOOGL\n\n"
            f"⚙️ Debug: `{dbg}`"
        )

    raw["Close"]  = pd.to_numeric(raw["Close"],  errors="coerce")
    raw["Volume"] = pd.to_numeric(raw["Volume"], errors="coerce")
    raw["Volume"] = raw["Volume"].replace(0, np.nan).ffill().fillna(1.0)
    raw.dropna(subset=["Close"], inplace=True)

    if len(raw) < 100:
        return None, (
            f"Not enough data ({len(raw)} rows) for **{ticker}**. "
            f"Try a longer date range (at least 2–3 years)."
        )

    # 2. Features
    df = build_features(raw[["Close","Volume"]].copy())
    feats = [f for f in FEATURES if f in df.columns]

    # 3. Train / Test
    X, y  = df[feats].values.astype(np.float64), df["Target"].values
    split = int(len(X) * train_pct)
    if split < 50:
        return None, "Train set too small. Reduce train % or use more data."

    model = RandomForestClassifier(n_estimators=300, max_depth=6,
                                    min_samples_leaf=15, random_state=42)
    model.fit(X[:split], y[:split])
    probs   = model.predict_proba(X[split:])[:,1]
    preds   = model.predict(X[split:])
    acc     = accuracy_score(y[split:], preds)
    signals = np.where(probs>buy_thresh, 1, np.where(probs<sell_thresh, -1, 0))

    dates_test = df.index[split:]
    close_test = df["Close"].values[split:].astype(float)
    ret_test   = df["Log_Return"].values[split:].astype(float)

    # 4. Simulate
    cash, shares, pos, entry_p = initial_cap, 0, 0, 0.0
    port, trade_rets, trades = [], [], []
    for price, sig, prob in zip(close_test, signals, probs):
        price = float(price)
        if sig==1 and pos==0:
            fee = cash*trans_cost
            shares = np.floor((cash-fee)/price)
            cash  -= shares*price*(1+trans_cost)
            pos, entry_p = 1, price
            trades.append({"Action":"BUY","Price":round(price,2),"Prob":round(float(prob),3)})
        elif sig==-1 and pos==1 and shares>0:
            cash += shares*price*(1-trans_cost)
            trade_rets.append((price-entry_p)/entry_p)
            trades.append({"Action":"SELL","Price":round(price,2),"Prob":round(float(prob),3)})
            shares, pos = 0, 0
        port.append(cash + shares*price)
    if shares>0:
        cash += shares*float(close_test[-1])*(1-trans_cost)
        trade_rets.append((float(close_test[-1])-entry_p)/entry_p)
        port[-1] = cash

    port_arr = np.array(port)
    bh_sh    = np.floor(initial_cap/close_test[0])
    bh_arr   = bh_sh*close_test + (initial_cap - bh_sh*close_test[0]*(1+trans_cost))

    # 5. ADF on full log returns
    adf_r  = adfuller(df["Log_Return"], autolag="AIC")
    adf_p  = adfuller(df["Close"], autolag="AIC")

    # 6. Next-day signal
    latest_x      = X[-1:].copy()
    sentiment_feat_idx = feats.index("News_Sentiment") if "News_Sentiment" in feats else None

    latest_prob   = float(model.predict_proba(latest_x)[0,1])
    latest_signal = 1 if latest_prob>buy_thresh else (-1 if latest_prob<sell_thresh else 0)

    # 7. Drawdown
    rm_s = np.maximum.accumulate(port_arr)
    dd_s = (port_arr - rm_s)/rm_s*100
    rm_b = np.maximum.accumulate(bh_arr)
    dd_b = (bh_arr   - rm_b)/rm_b*100

    # 8. Rolling metrics
    pr   = pd.Series(np.diff(port_arr)/port_arr[:-1], index=dates_test[1:])
    roll_sh = (pr.rolling(60).mean()/(pr.rolling(60).std()+1e-9))*np.sqrt(252)

    # 9. Monthly
    monthly_s = pd.Series(port_arr, index=dates_test).resample("ME").last().pct_change().dropna()*100
    monthly_b = pd.Series(bh_arr,   index=dates_test).resample("ME").last().pct_change().dropna()*100

    # 10. Confusion matrix
    cm = confusion_matrix(y[split:], preds).tolist()

    met = {
        "strat_final":  port_arr[-1],
        "strat_ret":    (port_arr[-1]-initial_cap)/initial_cap*100,
        "bh_ret":       (bh_arr[-1]-initial_cap)/initial_cap*100,
        "sharpe_s":     sharpe(port_arr, rf_rate),
        "sharpe_b":     sharpe(bh_arr,   rf_rate),
        "sortino_s":    sortino(port_arr, rf_rate),
        "sortino_b":    sortino(bh_arr,   rf_rate),
        "mdd_s":        max_dd(port_arr),
        "mdd_b":        max_dd(bh_arr),
        "cagr_s":       cagr(port_arr, dates_test),
        "cagr_b":       cagr(bh_arr,   dates_test),
        "calmar_s":     calmar(port_arr, dates_test),
        "calmar_b":     calmar(bh_arr,   dates_test),
        "win_rate":     win_rate(trade_rets),
        "n_trades":     sum(1 for t in trades if t["Action"]=="BUY"),
        "accuracy":     acc*100,
        "total_fees":   sum(t["Price"]*trans_cost for t in trades),
        "adf_ret_stat": adf_r[0], "adf_ret_p": adf_r[1],
        "adf_price_p":  adf_p[1],
        "alpha":        (port_arr[-1]-initial_cap)/initial_cap*100 - (bh_arr[-1]-initial_cap)/initial_cap*100,
    }

    fi_df = pd.DataFrame({"Feature":feats,
                          "Importance":model.feature_importances_}
                         ).sort_values("Importance",ascending=False)

    return {
        "df":         df,
        "raw":        raw,
        "dates_test": dates_test,
        "close_test": close_test,
        "ret_test":   ret_test,
        "port":       port_arr,
        "bh":         bh_arr,
        "signals":    signals,
        "probs":      probs,
        "dd_s":       dd_s,
        "dd_b":       dd_b,
        "roll_sh":    roll_sh,
        "monthly_s":  monthly_s,
        "monthly_b":  monthly_b,
        "trades":     trades,
        "trade_rets": trade_rets,
        "cm":         cm,
        "fi":         fi_df,
        "met":        met,
        "latest_prob":   latest_prob,
        "latest_signal": latest_signal,
        "ticker":     ticker,
        "initial_cap":initial_cap,
        "buy_thresh": buy_thresh,
        "sell_thresh":sell_thresh,
        "feats":      feats,
        "latest_x":           latest_x,
        "sentiment_feat_idx": sentiment_feat_idx,
    }, None

# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# ⚙️ Settings")
    st.markdown("---")

    ticker      = st.text_input("Stock Ticker", value="RELIANCE.NS",
                                help="NSE: RELIANCE.NS  |  BSE: RELIANCE.BO  |  US: AAPL")
    start_date  = st.date_input("Start Date", value=datetime.date(2014,1,1))
    end_date    = st.date_input("End Date",   value=datetime.date.today())
    st.markdown("---")
    initial_cap = st.number_input("Capital (₹)", value=100_000, step=10_000, min_value=10_000)
    trans_cost  = st.slider("Transaction Cost (%)", 0.0, 1.0, 0.1, 0.05) / 100
    buy_thresh  = st.slider("Buy Threshold",  0.50, 0.90, 0.60, 0.05)
    sell_thresh = st.slider("Sell Threshold", 0.10, 0.49, 0.40, 0.05)
    train_pct   = st.slider("Train Split (%)", 60, 90, 80, 5) / 100
    rf_rate     = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 6.5, 0.5) / 100
    st.markdown("---")
    run_btn = st.button("▶  ANALYSE STOCK", use_container_width=True)



    st.markdown("---")

    # ── Global usage counter ──────────────────────────────────────────────
    _sidebar_count = _upstash_get()
    if _sidebar_count is not None:
        st.markdown(
            f"<div class='counter-badge'>"
            f"<div class='counter-number'>📊 {_sidebar_count:,}</div>"
            f"<div class='counter-label'>analyses run globally</div>"
            f"</div>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### 📰 News Sentiment")
    enable_sentiment = st.toggle("Enable Real-Time Sentiment", value=True,
                                  help="Fetches latest headlines via Yahoo Finance (no key needed)")
    st.caption("Trust this analysis at your own risk. If it works, I’m a genius. If not… it was experimental.")

# ─────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────
st.markdown(f"# 📊 Trading Intelligence Platform")
st.markdown(f"*Enter a ticker in the sidebar and click **▶ ANALYSE STOCK***")
st.markdown("---")

# ─────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────
if "res" not in st.session_state:
    st.session_state.res = None

if run_btn:
    # ── Increment global counter (fire-and-forget, never blocks UI) ───────
    _new_count = _upstash_incr()

    with st.spinner(f"⬇️  Downloading {ticker}  →  Building features  →  Training model  →  Backtesting…"):
        res, err = run_pipeline(
            ticker, str(start_date), str(end_date),
            initial_cap, trans_cost, buy_thresh, sell_thresh,
            train_pct, rf_rate
        )
    if err:
        st.error(err)
        st.stop()
    st.session_state.res = res

    # Show live updated count as a small toast-style message
    if _new_count is not None:
        st.toast(f"📊 This platform has been used {_new_count:,} times!", icon="🎯")

if st.session_state.res is None:
    st.info("👈  Configure settings in the sidebar and click **▶ ANALYSE STOCK** to begin.")
    st.stop()

R   = st.session_state.res
m   = R["met"]
df  = R["df"]
raw = R["raw"]

dates      = R["dates_test"]
close_test = R["close_test"]
port       = R["port"]
bh         = R["bh"]
signals    = R["signals"]
probs      = R["probs"]
dd_s       = R["dd_s"]
dd_b       = R["dd_b"]

tk  = R["ticker"]
cap = R["initial_cap"]

# ─────────────────────────────────────────────────────────
# REAL-TIME SENTIMENT FETCH & INJECTION
# ─────────────────────────────────────────────────────────
_sentiment_result = {"score": 0.0, "label": "Neutral", "scores": [], "method": "N/A"}
_headlines        = []

if enable_sentiment:
    with st.spinner("📰 Fetching latest news headlines…"):
        _headlines = fetch_news_headlines(tk)
        if _headlines:
            _sentiment_result = analyse_sentiment(_headlines)
        else:
            _sentiment_result = {"score": 0.0, "label": "Neutral",
                                 "scores": [], "method": "N/A"}

_sent_score = _sentiment_result["score"]

# Re-score the latest prediction with live sentiment injected
_lp_sentiment = R["latest_prob"]
if enable_sentiment and R["sentiment_feat_idx"] is not None and _headlines:
    try:
        SENT_WEIGHT = 0.15
        _lp_sentiment = float(np.clip(
            R["latest_prob"] + SENT_WEIGHT * _sent_score, 0.01, 0.99
        ))
    except Exception:
        pass

_ls_sentiment = (1 if _lp_sentiment > R["buy_thresh"]
                 else (-1 if _lp_sentiment < R["sell_thresh"] else 0))

# ─────────────────────────────────────────────────────────
# ██  VERDICT BANNER  ██
# ─────────────────────────────────────────────────────────
lp  = _lp_sentiment
ls  = _ls_sentiment
last_close = float(df["Close"].iloc[-1])
last_date  = df.index[-1].date()

rsi_now    = float(df["RSI_14"].iloc[-1])
macd_now   = float(df["MACD"].iloc[-1])
macd_sig   = float(df["MACD_Signal"].iloc[-1])
sma20_now  = float(df["SMA_20"].iloc[-1])
sma50_now  = float(df["SMA_50"].iloc[-1])
bb_pctb    = float(df["BB_PctB"].iloc[-1])

score = 0
reasons = []
if lp > R["buy_thresh"]:
    score += 2; reasons.append(f"ML model: P(UP)={lp:.1%} — strong buy signal")
elif lp < R["sell_thresh"]:
    score -= 2; reasons.append(f"ML model: P(UP)={lp:.1%} — bearish")
else:
    reasons.append(f"ML model: P(UP)={lp:.1%} — neutral")

if rsi_now < 30:
    score += 1; reasons.append(f"RSI={rsi_now:.1f} — oversold (bullish)")
elif rsi_now > 70:
    score -= 1; reasons.append(f"RSI={rsi_now:.1f} — overbought (bearish)")
else:
    reasons.append(f"RSI={rsi_now:.1f} — neutral zone")

if macd_now > macd_sig:
    score += 1; reasons.append("MACD above signal line — bullish crossover")
else:
    score -= 1; reasons.append("MACD below signal line — bearish")

if last_close > sma20_now:
    score += 1; reasons.append(f"Price (₹{last_close:.0f}) above SMA20 (₹{sma20_now:.0f}) — uptrend")
else:
    score -= 1; reasons.append(f"Price below SMA20 — downtrend")

if last_close > sma50_now:
    score += 1; reasons.append(f"Price above SMA50 (₹{sma50_now:.0f}) — long-term bullish")
else:
    score -= 1; reasons.append(f"Price below SMA50 — long-term bearish")

if bb_pctb < 0.2:
    score += 1; reasons.append("Near lower Bollinger Band — potential reversal up")
elif bb_pctb > 0.8:
    score -= 1; reasons.append("Near upper Bollinger Band — potential pullback")

if enable_sentiment and _headlines:
    if _sent_score > 0.2:
        score += 1
        reasons.append(
            f"📰 News sentiment: {_sent_score:+.3f} — bullish ({_sentiment_result['label']}, "
            f"{len(_headlines)} headlines via {_sentiment_result['method']})"
        )
    elif _sent_score < -0.2:
        score -= 1
        reasons.append(
            f"📰 News sentiment: {_sent_score:+.3f} — bearish ({_sentiment_result['label']}, "
            f"{len(_headlines)} headlines via {_sentiment_result['method']})"
        )
    else:
        reasons.append(
            f"📰 News sentiment: {_sent_score:+.3f} — neutral "
            f"({_sentiment_result['method']})"
        )
    MAX_SCORE = 8
else:
    MAX_SCORE = 7

pct_bull = (score + MAX_SCORE) / (2 * MAX_SCORE) * 100

if score >= 3:
    verdict, css, emoji = "BUY / HOLD", "verdict-buy", "📈"
    advice = f"Strong bullish signal ({score}/{MAX_SCORE} indicators agree). Consider buying or maintaining position."
elif score <= -3:
    verdict, css, emoji = "SELL", "verdict-sell", "📉"
    advice = f"Bearish signal ({abs(score)}/{MAX_SCORE} indicators bearish). Consider reducing or exiting position."
else:
    verdict, css, emoji = "HOLD", "verdict-hold", "⏸"
    advice = f"Mixed signals (score {score:+d}/{MAX_SCORE}). No strong directional edge — hold and monitor."

_sent_badge = ""
if enable_sentiment and _headlines:
    _sent_color = "#14d278" if _sent_score > 0.05 else ("#f03c3c" if _sent_score < -0.05 else "#f5aa1e")
    _sent_badge = (f"&nbsp;|&nbsp; Sentiment: "
                   f"<span style='color:{_sent_color};font-weight:700'>"
                   f"{_sent_score:+.3f} {_sentiment_result['label']}</span>")

st.markdown(f"""
<div class='{css}'>
  <div class='verdict-title'>{emoji}  {verdict}</div>
  <div class='verdict-sub'>
    <b>{tk}</b> · Last close ₹{last_close:.2f} · {last_date}<br>
    {advice}<br><br>
    <b>Signal score: {score:+d} / {MAX_SCORE}</b> &nbsp;|&nbsp;
    ML Confidence: {lp:.1%} &nbsp;|&nbsp;
    RSI: {rsi_now:.1f} &nbsp;|&nbsp;
    MACD: {"↑" if macd_now>macd_sig else "↓"}
    {_sent_badge}
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("")

with st.expander("🔍 View full signal breakdown"):
    for i, r in enumerate(reasons, 1):
        icon = "🟢" if score > 0 else ("🔴" if "bearish" in r.lower() or "below" in r.lower() or "overbought" in r.lower() else "🟡")
        st.markdown(f"**{i}.** {icon} {r}")

# ─────────────────────────────────────────────────────────
# 📰 NEWS SENTIMENT PANEL
# ─────────────────────────────────────────────────────────
if enable_sentiment:
    with st.expander("📰 News Sentiment Analysis", expanded=bool(_headlines)):
        if not _headlines:
            st.warning("⚠️ Could not fetch headlines right now. "
                       "Sentiment set to 0 (neutral).")
            st.markdown("""
            **Possible reasons:**
            - 🌐 Network restriction in your environment
            - 🚫 News sources temporarily blocked
            - ⏱️ Request timed out

            **Try these fixes:**
            - Run the app on your **local machine** (not cloud/server)
            - Check your **internet connection**
            - Click **▶ ANALYSE STOCK** again to retry
            - Try a **US ticker** like `AAPL` or `TSLA` (Finviz fallback works for US)
            """)
        else:
            sent_col1, sent_col2 = st.columns([1, 2])

            with sent_col1:
                _sc  = _sentiment_result["score"]
                _lbl = _sentiment_result["label"]
                _css_class = (
                    "sentiment-score-positive" if _sc > 0.05
                    else "sentiment-score-negative" if _sc < -0.05
                    else "sentiment-score-neutral"
                )
                _bar_width  = int(abs(_sc) * 100)
                _bar_class  = "sentiment-bar-positive" if _sc >= 0 else "sentiment-bar-negative"
                _arrow      = "▲" if _sc > 0.05 else ("▼" if _sc < -0.05 else "●")
                _method_txt = _sentiment_result["method"]

                st.markdown(f"""
                <div class='sentiment-panel'>
                  <div style='font-family:Space Mono,monospace;font-size:10px;
                              letter-spacing:.1em;color:#4a5a80;text-transform:uppercase;
                              margin-bottom:4px'>AGGREGATE SENTIMENT</div>
                  <div class='{_css_class}'>{_arrow} {_sc:+.4f}</div>
                  <div style='color:#6b7fa8;font-size:11px;margin:4px 0 10px'>
                      {_lbl} · {len(_headlines)} headlines · {_method_txt}
                  </div>
                  <div class='sentiment-bar-wrap'>
                    <div class='{_bar_class}' style='width:{_bar_width}%'></div>
                  </div>
                  <div style='display:flex;justify-content:space-between;
                              font-size:10px;color:#2d3a55;margin-top:3px'>
                    <span>−1.0 Bearish</span><span>+1.0 Bullish</span>
                  </div>
                  <div style='margin-top:14px;font-size:11px;color:#4a5a80'>
                    {'📈 Adds <b>+1</b> to signal score' if _sc > 0.2
                     else ('📉 Adds <b>−1</b> to signal score' if _sc < -0.2
                           else '↔ Below threshold — no score impact')}
                  </div>
                </div>
                """, unsafe_allow_html=True)

            with sent_col2:
                st.markdown(
                    "<div style='font-family:Space Mono,monospace;font-size:10px;"
                    "letter-spacing:.1em;color:#4a5a80;text-transform:uppercase;"
                    "margin-bottom:8px'>LATEST HEADLINES</div>",
                    unsafe_allow_html=True
                )
                for idx, h in enumerate(_headlines):
                    hs  = _sentiment_result["scores"][idx] if idx < len(_sentiment_result["scores"]) else 0.0
                    dot = "🟢" if hs > 0.05 else ("🔴" if hs < -0.05 else "🟡")
                    url = h.get("url", "#")
                    src = h.get("source", "")
                    lnk = (f'<a href="{url}" target="_blank" '
                           f'style="color:#4f8ef7;text-decoration:none">↗</a>'
                           if url != "#" else "")
                    st.markdown(
                        f"<div class='news-headline'>"
                        f"{dot} <b style='color:#c8d4f0'>{h['title']}</b> {lnk}"
                        f"<br><span style='font-size:10px;color:#2d3a55'>"
                        f"{src} · polarity: {hs:+.3f}</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

st.markdown("---")

# ─────────────────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────────────────
st.markdown("## Performance Metrics")
c1,c2,c3,c4,c5,c6,c7,c8 = st.columns(8)
c1.metric("Final Value",    fmt_inr(m["strat_final"]))
c2.metric("Return",         f"{m['strat_ret']:+.1f}%",  f"α {m['alpha']:+.1f}% vs B&H")
c3.metric("CAGR",           f"{m['cagr_s']:+.1f}%",     f"B&H {m['cagr_b']:+.1f}%")
c4.metric("Sharpe",         f"{m['sharpe_s']:.2f}",     f"B&H {m['sharpe_b']:.2f}")
c5.metric("Max Drawdown",   f"{m['mdd_s']:.1f}%",       f"B&H {m['mdd_b']:.1f}%")
c6.metric("Win Rate",       f"{m['win_rate']:.1f}%")
c7.metric("Trades",         f"{m['n_trades']}")
c8.metric("Accuracy",       f"{m['accuracy']:.1f}%")

st.markdown("---")

# ─────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs([
    "📈 Price & Signals",
    "💰 Strategy Performance",
    "⚠️ Risk & Drawdown",
    "📊 EDA & Statistics",
    "🤖 Model Analysis",
    "📋 Trade Log",
])

# ══════════════════════════════════
# TAB 1 — PRICE & SIGNALS
# ══════════════════════════════════
with tab1:
    cutoff = pd.Timestamp(dates[-1]) - pd.DateOffset(years=2)
    mask   = dates >= cutoff
    d2, c2v, s2, p2 = dates[mask], close_test[mask], signals[mask], probs[mask]

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        row_heights=[.50,.18,.17,.15],
                        vertical_spacing=.03,
                        subplot_titles=("Price + Signals + Bollinger Bands",
                                        "Volume", "RSI (14)", "MACD"))

    df_mask = df[df.index >= cutoff]
    fig.add_trace(go.Scatter(x=df_mask.index, y=df_mask["BB_Upper"],
                             line=dict(color="rgba(99,160,255,.3)",width=.8),
                             name="BB Upper", showlegend=False), row=1,col=1)
    fig.add_trace(go.Scatter(x=df_mask.index, y=df_mask["BB_Lower"],
                             line=dict(color="rgba(99,160,255,.3)",width=.8),
                             fill="tonexty", fillcolor="rgba(79,142,247,.05)",
                             name="BB", showlegend=False), row=1,col=1)
    fig.add_trace(go.Scatter(x=df_mask.index, y=df_mask["SMA_20"],
                             line=dict(color="#f5aa1e",width=1,dash="dot"),
                             name="SMA20"), row=1,col=1)
    fig.add_trace(go.Scatter(x=df_mask.index, y=df_mask["SMA_50"],
                             line=dict(color="#a78bfa",width=1,dash="dot"),
                             name="SMA50"), row=1,col=1)
    fig.add_trace(go.Scatter(x=d2, y=c2v,
                             line=dict(color="#4f8ef7",width=1.8),
                             name="Close"), row=1,col=1)

    buy_m  = s2 == 1
    sell_m = s2 == -1
    if buy_m.any():
        fig.add_trace(go.Scatter(x=d2[buy_m], y=c2v[buy_m], mode="markers",
                                 name="BUY", marker=dict(symbol="triangle-up",
                                 size=9,color="#14d278",
                                 line=dict(color="#065e30",width=1))), row=1,col=1)
    if sell_m.any():
        fig.add_trace(go.Scatter(x=d2[sell_m], y=c2v[sell_m], mode="markers",
                                 name="SELL", marker=dict(symbol="triangle-down",
                                 size=9,color="#f03c3c",
                                 line=dict(color="#6b1111",width=1))), row=1,col=1)

    vol_c = ["#14d278" if r>0 else "#f03c3c" for r in df_mask["Log_Return"]]
    fig.add_trace(go.Bar(x=df_mask.index, y=df_mask["Volume"]/1e6,
                         marker_color=vol_c, name="Volume (M)",
                         showlegend=False), row=2,col=1)

    fig.add_trace(go.Scatter(x=df_mask.index, y=df_mask["RSI_14"],
                             line=dict(color="#a78bfa",width=1.2),
                             name="RSI", showlegend=False), row=3,col=1)
    fig.add_hline(y=70, line_color="#f03c3c", line_dash="dot", line_width=1, row=3,col=1)
    fig.add_hline(y=30, line_color="#14d278", line_dash="dot", line_width=1, row=3,col=1)
    fig.update_yaxes(range=[0,100], row=3,col=1)

    mc = ["#14d278" if v>=0 else "#f03c3c" for v in df_mask["MACD_Hist"]]
    fig.add_trace(go.Bar(x=df_mask.index, y=df_mask["MACD_Hist"],
                         marker_color=mc, name="MACD Hist",
                         showlegend=False), row=4,col=1)
    fig.add_trace(go.Scatter(x=df_mask.index, y=df_mask["MACD"],
                             line=dict(color="#4f8ef7",width=1),
                             name="MACD", showlegend=False), row=4,col=1)
    fig.add_trace(go.Scatter(x=df_mask.index, y=df_mask["MACD_Signal"],
                             line=dict(color="#f5aa1e",width=1),
                             name="Signal", showlegend=False), row=4,col=1)

    fig.update_layout(**PT, height=680, title=f"{tk} — Last 2 Years")
    fig.update_yaxes(tickprefix="₹", tickformat=",.0f", row=1,col=1)
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=d2, y=p2, fill="tozeroy",
                              line=dict(color="#7c6af7",width=1),
                              fillcolor="rgba(124,106,247,.12)", name="P(UP)"))
    fig2.add_hline(y=R["buy_thresh"],  line_color="#14d278", line_dash="dash", line_width=1.5,
                   annotation_text=f"Buy >{R['buy_thresh']}", annotation_font_color="#14d278")
    fig2.add_hline(y=R["sell_thresh"], line_color="#f03c3c", line_dash="dash", line_width=1.5,
                   annotation_text=f"Sell <{R['sell_thresh']}", annotation_font_color="#f03c3c")
    fig2.update_layout(**pt_layout(height=220, title="Model P(UP) Probability",
                                   yaxis=dict(range=[0,1])))
    st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════
# TAB 2 — STRATEGY PERFORMANCE
# ══════════════════════════════════
with tab2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=bh,   name=f"Buy & Hold ({m['bh_ret']:+.1f}%)",
                             line=dict(color="#4f8ef7",width=1.8,dash="dash"),
                             fill="tozeroy", fillcolor="rgba(79,142,247,.04)"))
    fig.add_trace(go.Scatter(x=dates, y=port, name=f"Strategy ({m['strat_ret']:+.1f}%)",
                             line=dict(color="#14d278",width=2.2),
                             fill="tozeroy", fillcolor="rgba(20,210,120,.05)"))
    fig.add_hline(y=cap, line_color="#6b7fa8", line_dash="dot",
                  annotation_text="Initial Capital", annotation_font_color="#6b7fa8")
    fig.update_layout(**pt_layout(height=400, title="Portfolio Value — Strategy vs Buy & Hold",
                                  yaxis=dict(tickprefix="₹", tickformat=",.0f")))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        cum_s = (port/cap - 1)*100
        cum_b = (bh/cap   - 1)*100
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=dates, y=cum_b, name="Buy & Hold",
                                  line=dict(color="#4f8ef7",width=1.5,dash="dash")))
        fig2.add_trace(go.Scatter(x=dates, y=cum_s, name="Strategy",
                                  line=dict(color="#14d278",width=2)))
        fig2.add_hrect(y0=min(cum_s.min(),cum_b.min())-2, y1=0,
                       fillcolor="rgba(240,60,60,.04)", line_width=0)
        fig2.update_layout(**pt_layout(height=320, title="Cumulative Return (%)",
                                       yaxis=dict(ticksuffix="%")))
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        ms = R["monthly_s"]
        mb = R["monthly_b"]
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=ms.index, y=ms.values, name="Strategy",
                              marker_color=["#14d278" if v>=0 else "#f03c3c" for v in ms]))
        fig3.add_trace(go.Bar(x=mb.index, y=mb.values, name="Buy & Hold",
                              marker_color=["rgba(79,142,247,.7)" if v>=0 else "rgba(240,60,60,.5)"
                                            for v in mb]))
        fig3.update_layout(**pt_layout(height=320, title="Monthly Returns",
                                       barmode="group", yaxis=dict(ticksuffix="%")))
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### Full Metrics Comparison")
    tbl_data = {
        "Metric":    ["Cumulative Return","CAGR","Sharpe Ratio","Sortino Ratio",
                      "Calmar Ratio","Max Drawdown","Win Rate","No. of Trades",
                      "Model Accuracy","Alpha vs B&H"],
        "Strategy":  [f"{m['strat_ret']:+.2f}%", f"{m['cagr_s']:+.2f}%",
                      f"{m['sharpe_s']:.3f}",     f"{m['sortino_s']:.3f}",
                      f"{m['calmar_s']:.3f}",     f"{m['mdd_s']:.2f}%",
                      f"{m['win_rate']:.1f}%",    str(m['n_trades']),
                      f"{m['accuracy']:.1f}%",    f"{m['alpha']:+.2f}%"],
        "Buy & Hold":[f"{m['bh_ret']:+.2f}%",  f"{m['cagr_b']:+.2f}%",
                      f"{m['sharpe_b']:.3f}",  f"{m['sortino_b']:.3f}",
                      f"{m['calmar_b']:.3f}",  f"{m['mdd_b']:.2f}%",
                      "—","1","—","—"],
    }
    st.dataframe(pd.DataFrame(tbl_data).set_index("Metric"),
                 use_container_width=True, height=390)

# ══════════════════════════════════
# TAB 3 — RISK & DRAWDOWN
# ══════════════════════════════════
with tab3:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=dd_s, fill="tozeroy", name="Strategy DD",
                             line=dict(color="#f03c3c",width=1.2),
                             fillcolor="rgba(240,60,60,.2)"))
    fig.add_trace(go.Scatter(x=dates, y=dd_b, name="Buy & Hold DD",
                             line=dict(color="#4f8ef7",width=1.2,dash="dash")))
    fig.add_hline(y=m["mdd_s"], line_color="#f03c3c", line_dash="dot",
                  annotation_text=f"Strat MDD {m['mdd_s']:.1f}%",
                  annotation_font_color="#f03c3c")
    fig.update_layout(**pt_layout(height=360, title="Drawdown Comparison",
                                  yaxis=dict(ticksuffix="%")))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        rs = R["roll_sh"]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=rs.index, y=rs.values, fill="tozeroy",
                                  line=dict(color="#14d278",width=1.4),
                                  fillcolor="rgba(20,210,120,.08)", name="60d Sharpe"))
        fig2.add_hline(y=0, line_color="#6b7fa8", line_dash="dot")
        fig2.add_hline(y=1, line_color="#f5aa1e", line_dash="dot",
                       annotation_text="Sharpe=1")
        fig2.update_layout(**PT, height=300, title="Rolling 60-Day Sharpe")
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        dr  = np.diff(port)/port[:-1]*100
        dr_b= np.diff(bh)/bh[:-1]*100
        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(x=dr_b, nbinsx=80, name="Buy & Hold",
                                    marker_color="rgba(79,142,247,.5)"))
        fig3.add_trace(go.Histogram(x=dr, nbinsx=80, name="Strategy",
                                    marker_color="rgba(20,210,120,.7)"))
        fig3.update_layout(**pt_layout(height=300, title="Daily Return Distribution",
                                       barmode="overlay",
                                       xaxis=dict(ticksuffix="%")))
        st.plotly_chart(fig3, use_container_width=True)

    var95 = np.percentile(dr, 5)
    var99 = np.percentile(dr, 1)
    v1,v2,v3,v4 = st.columns(4)
    v1.metric("VaR (95%)", f"{var95:.2f}%", "daily worst-case")
    v2.metric("VaR (99%)", f"{var99:.2f}%", "daily worst-case")
    v3.metric("Ann. Volatility", f"{np.std(dr)*np.sqrt(252):.2f}%")
    v4.metric("Sortino Ratio",   f"{m['sortino_s']:.3f}")

# ══════════════════════════════════
# TAB 4 — EDA & STATISTICS
# ══════════════════════════════════
with tab4:
    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=raw.index, y=raw["Close"].squeeze(),
                                 fill="tozeroy", line=dict(color="#4f8ef7",width=1.5),
                                 fillcolor="rgba(79,142,247,.06)", name="Close"))
        fig.update_layout(**pt_layout(height=300,
                                      title=f"{tk} Full Price History ({start_date}→{end_date})",
                                      yaxis=dict(tickprefix="₹", tickformat=",.0f")))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        ret_vals = df["Log_Return"].values
        x_range  = np.linspace(ret_vals.min(), ret_vals.max(), 300)
        mu, sig  = ret_vals.mean(), ret_vals.std()
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=ret_vals, nbinsx=100, name="Log Returns",
                                    marker_color="rgba(124,106,247,.7)",
                                    histnorm="probability density"))
        fig2.add_trace(go.Scatter(x=x_range,
                                  y=sci_norm.pdf(x_range, mu, sig),
                                  line=dict(color="#f03c3c",width=2,dash="dash"),
                                  name="Normal Fit"))
        fig2.update_layout(**pt_layout(height=300, title="Return Distribution vs Normal",
                                       xaxis=dict(ticksuffix="%")))
        st.plotly_chart(fig2, use_container_width=True)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df.index, y=df["RolVol_30"],
                              fill="tozeroy", line=dict(color="#f5aa1e",width=1.4),
                              fillcolor="rgba(245,170,30,.08)", name="30d Vol"))
    fig3.add_trace(go.Scatter(x=df.index, y=df["RolVol_10"],
                              line=dict(color="#f03c3c",width=1,dash="dot"),
                              name="10d Vol"))
    fig3.update_layout(**pt_layout(height=260, title="Rolling Annualized Volatility",
                                   yaxis=dict(ticksuffix="%")))
    st.plotly_chart(fig3, use_container_width=True)

    adf_stat = m["adf_ret_stat"]
    adf_pval = m["adf_ret_p"]
    col3,col4,col5,col6,col7,col8 = st.columns(6)
    col3.metric("Mean Daily Ret", f"{mu*100:.4f}%")
    col4.metric("Daily Std Dev",  f"{sig*100:.4f}%")
    col5.metric("Ann. Return",    f"{mu*252*100:.2f}%")
    col6.metric("Ann. Volatility",f"{sig*np.sqrt(252)*100:.2f}%")
    col7.metric("Skewness",       f"{df['Log_Return'].skew():.3f}")
    col8.metric("Kurtosis",       f"{df['Log_Return'].kurt():.3f}")

    st.markdown("---")
    st.markdown("### ADF Stationarity Test")
    a1,a2,a3 = st.columns(3)
    a1.metric("ADF Statistic (Log Returns)", f"{adf_stat:.4f}")
    a2.metric("p-value", f"{adf_pval:.6f}")
    a3.metric("Stationary?",
              "✅ YES" if adf_pval<0.05 else "❌ NO",
              "p < 0.05" if adf_pval<0.05 else "p ≥ 0.05")
    st.markdown("**Raw Price ADF p-value:** "
                f"`{m['adf_price_p']:.6f}` — "
                f"{'✅ Stationary' if m['adf_price_p']<0.05 else '❌ Non-Stationary (expected for raw prices)'}")

# ══════════════════════════════════
# TAB 5 — MODEL ANALYSIS
# ══════════════════════════════════
with tab5:
    col1, col2 = st.columns(2)
    with col1:
        fi = R["fi"]
        fig = go.Figure(go.Bar(
            x=fi["Importance"], y=fi["Feature"], orientation="h",
            marker=dict(color=fi["Importance"],
                        colorscale=[[0,"#111b35"],[.5,"#1a6ef7"],[1,"#14d278"]],
                        showscale=False)
        ))
        fig.update_layout(**PT, height=500, title="Feature Importance",
                          xaxis_title="Importance Score")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        cm  = np.array(R["cm"])
        labs= ["DOWN","UP"]
        fig2= go.Figure(go.Heatmap(
            z=cm, x=labs, y=labs,
            colorscale=[[0,"#07090f"],[.5,"#1247c7"],[1,"#14d278"]],
            text=cm, texttemplate="%{text}",
            textfont=dict(size=26, color="white"), showscale=False
        ))
        fig2.update_layout(**PT, height=300, title="Confusion Matrix",
                           xaxis_title="Predicted", yaxis_title="Actual")
        st.plotly_chart(fig2, use_container_width=True)

        acc_val = m["accuracy"]
        fig3 = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=acc_val,
            delta={"reference":50,"suffix":"%","prefix":"vs baseline: "},
            gauge={
                "axis":{"range":[45,70],"ticksuffix":"%"},
                "bar":{"color":"#14d278"},
                "bgcolor":"#0c1020",
                "steps":[
                    {"range":[45,50],"color":"rgba(240,60,60,.25)"},
                    {"range":[50,55],"color":"rgba(245,170,30,.25)"},
                    {"range":[55,70],"color":"rgba(20,210,120,.25)"},
                ],
                "threshold":{"line":{"color":"#4f8ef7","width":2},
                             "thickness":.8,"value":50}
            },
            number={"suffix":"%","font":{"color":"#e8eeff","size":34}},
            title={"text":"Model Accuracy","font":{"color":"#6b7fa8","size":13}}
        ))
        fig3.update_layout(paper_bgcolor="#07090f", height=250,
                           margin=dict(l=30,r=30,t=40,b=10))
        st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════
# TAB 6 — TRADE LOG
# ══════════════════════════════════
with tab6:
    trades_df = pd.DataFrame(R["trades"])
    if not trades_df.empty:
        n_buy  = (trades_df["Action"]=="BUY").sum()
        n_sell = trades_df["Action"].str.contains("SELL").sum()
        st.markdown(f"**{len(trades_df)} transactions** · {n_buy} buys · {n_sell} sells "
                    f"· Win rate {m['win_rate']:.1f}% · Fees ~₹{m['total_fees']:,.0f}")

        def highlight(row):
            c = "rgba(20,210,120,.12)" if row.get("Action","")=="BUY" else "rgba(240,60,60,.12)"
            return [f"background-color: {c}"]*len(row)

        st.dataframe(trades_df.style.apply(highlight, axis=1),
                     use_container_width=True, height=480)

        csv = trades_df.to_csv(index=False)
        st.download_button("⬇  Download CSV", csv, f"{tk}_trades.csv", "text/csv")
    else:
        st.warning("No trades executed. Try lowering the Buy threshold in the sidebar.")

# ─────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<small style='color:#2d3a55'>⚠️ For educational purposes only. "
    "Not financial advice. Past performance does not guarantee future results. "
    "Always consult a SEBI-registered financial advisor before investing.</small>",
    unsafe_allow_html=True
)
