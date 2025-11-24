import base64 
import io
import pandas as pd
import numpy as np
from pathlib import Path


REQ5 = ["Date", "Ticker", "Close", "Returns", "Volume"] 

def read_csv(contents: str) -> pd.DataFrame:
    _t, b64 = contents.split(",", 1)
    df = pd.read_csv(io.BytesIO(base64.b64decode(b64)))

    # —— 1) 建立小写到原名的映射，便于匹配各种大小写/别名 —— 
    lower2orig = {c.strip().lower(): c for c in df.columns}

    # —— 2) 逐项确定“唯一来源列名” —— 
    date_col    = lower2orig.get("date")
    ticker_col  = lower2orig.get("ticker")

    # Close 优先级：close > adjusted
    close_col   = lower2orig.get("close") or lower2orig.get("adjusted")

    # Returns 优先级：returns > return > ret
    returns_col = (lower2orig.get("returns")
                   or lower2orig.get("return")
                   or lower2orig.get("ret"))

    # Volume/MV/MarketValue
    volume_col  = (lower2orig.get("volume")
                   or lower2orig.get("mv")
                   or lower2orig.get("marketvalue"))

    missing = [name for name, col in {
        "Date": date_col, "Ticker": ticker_col, "Close": close_col,
        "Returns": returns_col, "Volume": volume_col
    }.items() if col is None]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}. Expected {REQ5}")

    # —— 3) 只取以上 5 个来源列，并重命名为标准列名 —— 
    df = df[[date_col, ticker_col, close_col, returns_col, volume_col]].copy()
    df.columns = REQ5  # 重命名为统一的 5 列

    # —— 4) 基础清洗（全部是一维 Series，不会再触发 2D 错误）——
    df["Date"]   = pd.to_datetime(df["Date"], errors="coerce")
    df["Ticker"] = df["Ticker"].astype(str).str.strip()

    # 去掉千分位逗号后再转数值，确保是 1D Series
    for col in ["Close","Returns","Volume"]:
        df[col] = (df[col].astype(str).str.replace(",", "", regex=False))
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Date","Ticker"]).sort_values(["Date","Ticker"])
    return df

def _validate_schema(df):
    """
    Return error message if schema invalid, else None.
    """
    # what is missing in required columns
    missing = [c for c in REQ5 if c not in df.columns]
    # extra columns present
    extras  = [c for c in df.columns if c not in REQ5]

    if missing:
        return f"Missing required columns: {missing}. Expected: {REQ5}"
    return None

def sample_5col() -> pd.DataFrame:
    """
    简洁示例：5 列长表（Date, Ticker, Close, Returns, Volume）
    Volume = Market Value (收盘持仓市值)
    """
    import pandas as pd
    import numpy as np
    
    dates = pd.bdate_range("2024-01-02","2024-03-29", freq="B")
    tickers = ["AAPL","BAC","KO","AXP","CVX"]

    # 固定股数 (模拟真实持仓)
    shares = {
        "AAPL": 1500,
        "BAC":  3000,
        "KO":   1800,
        "AXP":  1200,
        "CVX":  1000,
    }

    rows = []
    day_range = pd.Series(range(len(dates)))
    
    for t in tickers:
        # base price + 轻微趋势
        base = 100 + 0.1 * day_range + (0.5 if t in ("AAPL","CVX") else 0.0)
        rets = base.pct_change().fillna(0.0) + 0.001  # 模拟收益
        close = 100 * (1 + rets).cumprod()
        
        # 收盘市值 = 股数 × 收盘价
        volume_mv = shares[t] * close.values
        
        tmp = pd.DataFrame({
            "Date": dates,
            "Ticker": t,
            "Close": close.values,
            "Returns": rets.values,
            "Volume": volume_mv,    # ✅ 市值权重依据
        })
        rows.append(tmp)

    return pd.concat(rows, ignore_index=True)

def _to_wide(df_long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # index=Date, columns=Ticker
    prices_w = df_long.pivot(index="Date", columns="Ticker", values="Close").sort_index()
    rets_w   = df_long.pivot(index="Date", columns="Ticker", values="Returns").sort_index()
    vol_w    = df_long.pivot(index="Date", columns="Ticker", values="Volume").sort_index()
    return prices_w, rets_w, vol_w

def historical_var(series: pd.Series, cl: float = 0.99, window: int = 252) -> float:
    s = pd.Series(series).dropna()
    if len(s) < max(30, int(window/4)):
        return np.nan
    sample = s.iloc[-min(window, len(s)):]
    q = np.percentile(sample, (1 - cl) * 100.0)
    return -float(q)  # 返回“损失幅度”为正数

def rolling_var(series: pd.Series, cl: float, window: int) -> pd.Series:
    s = pd.Series(series).dropna()
    if len(s) < window:
        return pd.Series(dtype=float, name=f"rVaR_{int(cl*100)}_{window}")
    r = -s.rolling(window).quantile((1 - cl)).dropna()
    r.name = f"VaR({int(cl*100)}%,{window}d)"
    return r

def compute_weights_eod(mv_w: pd.DataFrame) -> pd.DataFrame:
    """
    根据每日收盘持仓市值 MV 生成下一日的权重。
    每天权重都会动态变化。
    """
    mv = mv_w.copy()
    
    # 横向归一化（净敞口）
    denom = mv.sum(axis=1)
    w = mv.div(denom.replace(0, pd.NA), axis=0).fillna(0.0)

    # 收盘数据 → 下一日生效权重（避免前视偏差）
    w = w.shift(1)

    return w

def compute_portfolio_return(rets_w: pd.DataFrame, mv_w: pd.DataFrame) -> pd.Series:
    """
    用收盘 MV 算组合收益： w_{t-1} * r_t
    """
    w = compute_weights_eod(mv_w)
    rets = rets_w.reindex_like(w).fillna(0.0)
    port = (rets * w).sum(axis=1)
    return port.dropna().rename("Portfolio")


import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# === 风险代理：20日滚动波动率（年化），你也可以换成 |return| 或 20日VaR ===
def risk_proxy_from_returns(port: pd.Series,
                            window: int = 20,
                            annualize: bool = True,
                            trading_days: int = 252) -> pd.Series:
    """
    port: 组合日收益序列（index 为日期）
    返回：风险代理序列（波动率）
    """
    vol = port.rolling(window).std()
    if annualize:
        vol = vol * np.sqrt(trading_days)
    return vol.dropna().rename(f"RiskVol_{window}d")

# === 自动选择 d：ADF，若非平稳（p<0.05为平稳），则 d=1，否则 d=0 ===
def _auto_difference(series: pd.Series) -> int:
    s = pd.Series(series).dropna()
    if len(s) < 30:  # 样本太短，就直接 d=0
        return 0
    stat, pval, *_ = adfuller(s, autolag="AIC")
    return 0 if pval < 0.05 else 1

# === 简易网格搜索 (p,q) by AIC；d 可自动或指定 ===
def auto_arima_grid(series: pd.Series,
                    max_p: int = 3,
                    max_q: int = 3,
                    d: int = None,
                    enforce_pos: bool = True) -> Dict[str, Any]:
    """
    返回 {'order':(p,d,q), 'model':fitted_model, 'aic':aic}
    """
    y = pd.Series(series).astype(float).dropna()
    if d is None:
        d = _auto_difference(y)

    best = {"order": None, "model": None, "aic": np.inf}
    for p in range(0, max_p + 1):
        for q in range(0, max_q + 1):
            if p == 0 and d == 0 and q == 0:
                continue  # 跳过 (0,0,0)
            try:
                # 注意：波动率为非负，可以启用 enforce_stationarity/enforce_invertibility
                model = ARIMA(y, order=(p, d, q), enforce_stationarity=True, enforce_invertibility=True)
                res = model.fit()
                if res.aic < best["aic"]:
                    best = {"order": (p, d, q), "model": res, "aic": res.aic}
            except Exception:
                continue
    if best["model"] is None:
        # 回退方案：强行 (1,d,1)
        res = ARIMA(y, order=(1, d, 1)).fit()
        best = {"order": (1, d, 1), "model": res, "aic": res.aic}
    return best

def arima_forecast(series: pd.Series, horizon: int = 1,
                   max_p: int = 3, max_q: int = 3, d: int = None) -> Tuple[pd.Series, pd.DataFrame, Dict]:
    """
    对传入序列做 ARIMA 预测。
    返回：
      y_forecast: 预测点（index 为未来步）
      conf_int:   置信区间 DataFrame
      info:       {'order':(p,d,q), 'aic':..., 'last_value':...}
    """
    best = auto_arima_grid(series, max_p=max_p, max_q=max_q, d=d)
    res = best["model"]
    fc = res.get_forecast(steps=horizon)
    yhat = fc.predicted_mean
    ci = fc.conf_int(alpha=0.05)  # 95% CI

    info = {"order": best["order"], "aic": best["aic"], "last_value": float(series.dropna().iloc[-1])}
    return yhat, ci, info

# 组合 β（对 SPX）的滚动或截面估计：这里用最近 window 天
def beta_to_spx(port: pd.Series, spx: pd.Series, window: int = 252) -> float:
    s1 = port.dropna(); s2 = spx.dropna()
    df = pd.concat([s1, s2], axis=1, join="inner").tail(window)
    df.columns = ["port", "spx"]
    if len(df) < 30 or df["spx"].var() == 0:
        return np.nan
    return float(df["port"].cov(df["spx"]) / df["spx"].var())


# 组合收益（使用已存好的 EOD 权重）
def portfolio_return_from_w(rets_w: pd.DataFrame, w_w: pd.DataFrame) -> pd.Series:
    r = rets_w.reindex_like(w_w).fillna(0.0)
    w = w_w.fillna(0.0)
    return (r * w).sum(axis=1).dropna().rename("Portfolio")

# 预设历史场景（用日期区间从 SPX 序列实时计算累计跌幅）
SCENARIOS = {
    "COVID-19 Crash (Feb 19–Mar 23, 2020)": {"start": "2020-02-19", "end": "2020-03-23"},
    "GFC – Lehman Shock (Sep 12–Oct 10, 2008)": {"start": "2008-09-12", "end": "2008-10-10"},
    "Dot-com Peak to Trough (Mar 2000–Oct 2002)": {"start": "2000-03-24", "end": "2002-10-09"},
}


def scenario_shock_from_spx(spx_returns: pd.Series, start: str, end: str) -> float:
    win = spx_returns.loc[start:end].dropna()
    if win.empty:
        return np.nan
    cum = (1 + win).prod() - 1.0  # 该区间累计收益（负数=下跌）
    return float(cum)


def load_spx_returns_from_local(path_str: str) -> pd.Series:
    """
    从本地 CSV 读取 SPX 日收益（后台用，无可见 UI）。
    期望列：Date, Return（十进制日收益）
    返回：pd.Series（index=Date, name="^GSPC"）
    """
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"SPX file not found: {p}")

    df = pd.read_csv(p)
    # 轻度容错大小写
    df = df.rename(columns={c: c.strip().title() for c in df.columns})

    if "Date" not in df.columns or "Return" not in df.columns:
        raise ValueError("SPX CSV must have columns: Date, Return (decimal returns).")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")
    r = pd.to_numeric(df["Return"], errors="coerce").fillna(0.0)
    r.index = df["Date"].values
    return r.rename("^GSPC")

def compute_stress_linked(
    rets_w_data, w_w_data, d0, d1,
    spx_data, scn_key,
    beta_window=252, cl=0.995, win=252,
    scenarios=None, hist_var_fn=None
):
    import pandas as pd, numpy as np, plotly.express as px

    # 若缺输入，返回占位文本和空图
    if not (rets_w_data and w_w_data and spx_data and d0 and d1 and scn_key and scenarios):
        return "⏳ Waiting for inputs...", px.line()

    # 还原数据
    rets_w = pd.DataFrame(rets_w_data).set_index("Date")
    w_w    = pd.DataFrame(w_w_data).set_index("Date")
    for df in (rets_w, w_w):
        df.index = pd.to_datetime(df.index, errors="coerce"); df.sort_index(inplace=True)

    spx_df = pd.DataFrame(spx_data)
    spx_df["Date"] = pd.to_datetime(spx_df["Date"], errors="coerce")
    spx_df = spx_df.dropna(subset=["Date"]).sort_values("Date")
    spx_full = pd.Series(pd.to_numeric(spx_df["Return"], errors="coerce").fillna(0.0).values,
                         index=spx_df["Date"].values, name="^GSPC")

    # 与 Tab2 区间一致（用于 β 与当前 VaR）
    d0 = pd.to_datetime(d0); d1 = pd.to_datetime(d1)
    rets_w = rets_w.loc[d0:d1]; w_w = w_w.loc[d0:d1]
    if rets_w.empty or w_w.empty:
        return "⚠️ Portfolio returns/weights empty in selected range", px.line()

    spx_for_beta = spx_full.loc[d0:d1]
    rets_aligned = rets_w.reindex_like(w_w).fillna(0.0)
    w_aligned    = w_w.fillna(0.0)
    port = (rets_aligned * w_aligned).sum(axis=1).dropna().rename("Portfolio")
    if port.empty or spx_for_beta.empty:
        return "⚠️ No overlap between portfolio and SPX for beta/VaR", px.line()

    # β
    beta_window = int(beta_window)
    if len(port) >= beta_window and len(spx_for_beta) >= beta_window:
        cov = port.rolling(beta_window).cov(spx_for_beta)
        var = spx_for_beta.rolling(beta_window).var()
        beta = float((cov / var).iloc[-1])
    else:
        var_spx = float(spx_for_beta.var())
        beta = float(port.cov(spx_for_beta) / var_spx) if var_spx != 0 else np.nan

    # 情景冲击（用全样本 SPX）
    scn = scenarios[scn_key]
    start, end = (
        (pd.to_datetime(scn["start"]), pd.to_datetime(scn["end"]))
        if isinstance(scn, dict) else
        (pd.to_datetime(scn[0]), pd.to_datetime(scn[1]))
    )
    spx_slice = spx_full.loc[start:end]
    spx_shock = (1 + spx_slice).prod() - 1 if not spx_slice.empty else np.nan
    port_shock = beta * spx_shock if (np.isfinite(beta) and np.isfinite(spx_shock)) else np.nan

    # 当前 VaR
    cl = float(cl); win = int(win)
    if len(port) >= max(win, 30):
        if callable(hist_var_fn):
            var_val = hist_var_fn(port, cl=cl, window=win)
        else:
            var_val = -np.percentile(port.iloc[-win:], (1 - cl) * 100.0)
    else:
        var_val = np.nan

    # 摘要
    summary = (
        f"Scenario: {scn_key} | Range: {start.date()} → {end.date()} | "
        f"β={('NA' if not np.isfinite(beta) else f'{beta:.2f}')} | "
        f"SPX shock={('NA' if not np.isfinite(spx_shock) else f'{spx_shock:.2%}')} | "
        f"Portfolio shock(β×SPX)={('NA' if not np.isfinite(port_shock) else f'{port_shock:.2%}')} | "
        f"VaR({int(cl*100)}%, {win}d)={('NA' if not np.isfinite(var_val) else f'{var_val:.2%}')}"
    )

    # 图
    cum = (1 + port).cumprod()
    fig = px.line(cum.to_frame("Cumulative NAV"))
    fig.update_layout(title="Portfolio Stress: Scenario Shock vs Current NAV",
                      xaxis_title="Date", yaxis_title="Value")
    fig.add_vrect(x0=start, x1=end, fillcolor="red", opacity=0.15,
                  annotation_text=scn_key, annotation_position="top left")
    if np.isfinite(port_shock) and len(cum) > 0:
        nav_last = float(cum.iloc[-1]); nav_scn = nav_last * (1 + port_shock)
        fig.add_scatter(x=[cum.index[-1], cum.index[-1]],
                        y=[nav_last, nav_scn],
                        mode="markers+text",
                        text=["Now", f"Shock\n{port_shock:.1%}"],
                        textposition="top center",
                        name="Scenario Shock")
    return summary, fig