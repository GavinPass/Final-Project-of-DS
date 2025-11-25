#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
import customfunction as cf
from MacroFunction import GrabRate, GrabMacro

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPX_LOCAL_PATH = os.path.join(BASE_DIR, "SPX_return.csv")


df=pd.read_excel('ratesdata.xlsx',index_col=0)
rates = pd.DataFrame()
# for i in range(3):
#     rates=pd.concat([rates,GrabRate(i+2020)])
rates=GrabRate(2025)
rates=pd.concat([df,rates])
#rates.tail(5)
rates = rates[(rates != 0).any(axis=1)]


Macro_data=pd.DataFrame()
Macro_data['Inflation_rate']=GrabMacro('inflation')[1]
# Macro_data['Unemployment_rate']=GrabData('unemployment')[1]
Macro_data.index=GrabMacro('inflation')[0]

Macro_data=Macro_data.iloc[::-1]
#Macro_data.tail()


# In[ ]:


import base64
import dash
from dash import dcc
from dash import html
import plotly.express as px
from dash import dcc, html, dash_table, no_update, callback, exceptions
from dash.dependencies import Input, Output, State
import io
import xlsxwriter
import flask
from flask import send_file


# In[ ]:


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__,external_stylesheets=external_stylesheets)
app.config.suppress_callback_exceptions = True
server = app.server

# transfer local image to base64,because HTML can't directly show local image

# In[ ]:


# ======== 布局：Tabs（Tab1: Macro, Tab2: Portfolio） ========
app.layout = html.Div([
    html.H1("Daily Updated Economic Indicators and Portfolio Metrics", style={'fontSize':'40px','color':'black','textAlign':'center'}),
    dcc.Tabs(id='tabs', value='tab-macro', children=[
        dcc.Tab(label='Macro Dashboard', value='tab-macro', children=[
            html.Br(),
            dcc.Graph(id='ratechart'),
            html.Div([
                dcc.Dropdown(
                    id='yield_type',
                    options=[{'label':i,'value':i} for i in rates.columns],
                    multi=True,
                    value=['10-year rate'],
                    placeholder='choose rate type'
                ),
                dcc.DatePickerRange(
                    id='rate_date_range',
                    min_date_allowed=rates.index.min().date(),
                    max_date_allowed=rates.index.max().date(),
                    start_date=rates.index.min().date(),
                    end_date=rates.index.max().date(),
                )
            ]),
            html.Div([html.A('Download Rates Data', href='/download_excel/rate/')]),
            # html.Br(),
            # dcc.Graph(id='indexchart'),
            # html.Div([
            #     dcc.Dropdown(
            #         id='index_type',
            #         options=[{'label':j,'value':j} for j in Stock.columns],
            #         value='VIX'
            #     ),
            #     dcc.DatePickerRange(
            #         id='index_date_range',
            #         min_date_allowed=Stock.index.min().date(),
            #         max_date_allowed=Stock.index.max().date(),
            #         start_date=Stock.index.min().date(),
            #         end_date=Stock.index.max().date(),
            #     )
            # ]),
            # html.Div([html.A('Download Index Data', href='/download_excel/stock/')]),
            html.Br(),
            dcc.Graph(id='macrochart'),
            html.Div([
                dcc.DatePickerRange(
                    id='macro_date_range',
                    min_date_allowed=Macro_data.index.min().date(),
                    max_date_allowed=Macro_data.index.max().date(),
                    start_date=Macro_data.index.min().date(),
                    end_date=Macro_data.index.max().date(),
                )
            ]),
        ]),
        dcc.Tab(label='Portfolio Analytics', value='tab-portfolio', children=[
            html.Br(),
            html.Div([
                #上传组合文件 & 预览
                html.Div([
                    html.H3("Load Portfolio"),
                    html.Button("Sample Portfolio Data", id='btn_load_sample', n_clicks=0),
                    html.Div(style={'height':'6px'}),
                    dcc.Upload(
                        id='upload_portfolio',
                        children=html.Div(['Drag & Drop or ', html.A('Select CSV file')]),
                        multiple=False,
                        style={
                            'width':'100%','height':'60px','lineHeight':'60px','borderWidth':'1px',
                            'borderStyle':'dashed','borderRadius':'5px','textAlign':'center'
                        }
                    ),
                    html.Div(id="portfolio-msg", style={"marginTop":"8px","color":"#555","fontSize":"12px"}),
                    dash_table.DataTable(
                        id="portfolio-preview",
                        page_size=10,
                        style_table={"maxHeight":"260px","overflowY":"auto"},
                        style_cell={"fontFamily":"monospace","fontSize":12,"padding":"6px"},
                        style_header={"fontWeight":"bold"},
                    ),
                    # 全局的 Stores（如果你之前已经在别处定义，保持唯一即可）
                    dcc.Store(id="store-portfolio-long"),
                    dcc.Store(id="store-prices-wide"),
                    dcc.Store(id="store-returns-wide"),
                    dcc.Store(id="store-volume-wide"),   # MV 宽表
                    dcc.Store(id="store-weights-wide"), 
                ]),
                #参数选择
                html.Div([
                    html.H3("Portfolio Performance & VaR"),
                    html.Label("VaR Confidence"),
                    dcc.Dropdown(
                        id='pf_conf',
                        options=[
                            {'label':'95%','value':0.95},
                            {'label':'99%','value':0.99},
                            {'label':'99.5%','value':0.995},
                        ],
                        value=0.99, clearable=False
                    ),
                    html.Label("Window (trading days)"),
                    dcc.Dropdown(
                        id='pf_window',
                        options=[
                            {"label": "3M (~63)",  "value": 63},
                            {"label": "6M (~126)", "value": 126},
                            {"label": "1Y (~252)", "value": 252},
                            {"label": "2Y (~504)", "value": 504},
                        ],
                        value=252, clearable=False
                    ),
                    html.Label("Backtest Range"),
                    dcc.DatePickerRange(
                        id='pf_date',
                        display_format="MM/DD/YYYY",
                        # 这两个日期会在回调里根据数据动态修正
                        start_date=None,
                        end_date=None,
                        minimum_nights=0,
                        updatemode="bothdates"
                    ),
                    html.Div(id="var_summary", style={"marginTop": "12px", "fontFamily": "monospace"}),
                    dcc.Graph(id="pf_perf", style={"height": 340, "marginTop":"6px"}),
                    dcc.Graph(id="var_chart", style={"height": 320}),
                    html.H3("Weights"),
                    dcc.Graph(id="pf-weights-heat", style={"height": 360}),
                    dcc.Graph(id="pf-weights-lines", style={"height": 300}),
                ]),
                # ARIMA 预测模块
                html.Div([
                    html.H3("ARIMA Risk Forecast"),
                    html.Div(id="arima-summary", style={"marginTop":"8px","fontFamily":"monospace"}),
                    dcc.Graph(id="pf-arima-risk", style={"height": 320}),
                ]),
            ], className='row'),
        ]),
        dcc.Tab(label='Stress Testing', value='tab-stress', children=[
            # 后台用的不可见存储（不要重复定义这个 ID）
            dcc.Store(id="store-spx-returns"),
            # 可选：启动后在后台触发一次加载（如果你用了 init-once 的回调）
            dcc.Interval(id="init-once-stress", interval=100, n_intervals=0, max_intervals=1),

            html.Br(),
            html.Div([
                html.Div([
                    html.H3("Scenario & Parameters"),
                    dcc.Dropdown(
                        id="stress-scn",
                        options=[{"label": k, "value": k} for k in cf.SCENARIOS.keys()],
                        value="COVID-19 Crash (Feb 19–Mar 23, 2020)",
                        clearable=False
                    ),
                    html.Label("β lookback (days)"),
                    dcc.Dropdown(
                        id="beta_window",
                        options=[{"label": x, "value": x} for x in [126, 252, 504]],
                        value=252, clearable=False
                    ),
                    # html.Label("VaR CL"),
                    # dcc.Dropdown(
                    #     id="stress_cl",
                    #     options=[{"label":"99.5%","value":0.995},{"label":"99%","value":0.99}],
                    #     value=0.995, clearable=False
                    # ),
                    # html.Label("VaR window (days)"),
                    # dcc.Dropdown(
                    #     id="stress_win",
                    #     options=[{"label":x, "value":x} for x in [126, 252, 504]],
                    #     value=252, clearable=False
                    # ),
                ], className="six columns"),

                html.Div([
                    html.Div(id="stress-summary",
                            style={"fontFamily":"monospace","marginBottom":"8px","marginTop":"36px"}),
                    dcc.Graph(id="stress-figure", style={"height": 360, "marginTop":"4px"}),
                ], className="six columns"),
            ], className="row"),
        ]),                
    ])
])

# ======== 回调：Macro 两张图 ========
@app.callback(
    [Output('ratechart','figure'),
     Output('macrochart','figure')],
    [Input('yield_type','value'),
     Input('rate_date_range','start_date'),
     Input('rate_date_range','end_date'),
     Input('macro_date_range','start_date'),
     Input('macro_date_range','end_date')]
)

def update_figure(rateType, rateStart, rateEnd,macroStart,macroEnd):
    
    filterd1=rates.loc[rateStart:rateEnd][rateType]
    fig1=px.line(filterd1)
    fig1.update_layout(
        title='Treasury Interest Rate Since 2013',
        xaxis_title="Timeline",
        yaxis_title="Interest Rate",
        title_x=0.5
    )

    
    filterd3=Macro_data.loc[macroStart:macroEnd]
    fig3=px.line(filterd3)
    fig3.update_layout(
        title='last 10 Years Inflation',
        xaxis_title="Timeline",
        title_x=0.5
    )
    
    return fig1,fig3

# ======== 回调：上传 & 解析 Portfolio 文件 ========
@callback(
    Output("store-portfolio-long","data"),
    Output("store-prices-wide","data"),
    Output("store-returns-wide","data"),
    Output("store-volume-wide","data"),
    Output("portfolio-preview","data"),      
    Output("portfolio-preview","columns"), 
    Input("upload_portfolio","contents"),
    State("upload_portfolio","filename"),
    Input("btn_load_sample","n_clicks"),
    prevent_initial_call=True
)
def load_portfolio(contents, filename, n_clicks_sample):
    trigger = dash.ctx.triggered_id

    # 1) 数据来源
    if trigger == "btn_load_sample":
        if not n_clicks_sample:
            raise dash.exceptions.PreventUpdate
        df = cf.sample_5col()  # 5 列：Date/Ticker/Close/Returns/Volume(=MV)

    elif trigger == "upload_portfolio":
        if not contents:
            raise dash.exceptions.PreventUpdate
        # 读取并只保留 5 列（你的 cf.read_csv 要返回这 5 列）
        df = cf.read_csv(contents)
    else:
        raise dash.exceptions.PreventUpdate

    # 2) 基础规范化/校验
    df = df[["Date","Ticker","Close","Returns","Volume"]].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date","Ticker"]).sort_values(["Date","Ticker"])

    err = getattr(cf, "_validate_schema", lambda x: None)(df)
    if err:
        # 这里只返回 no_update，或你也可以抛 PreventUpdate
        return no_update, no_update, no_update, no_update, no_update, no_update

    # 3) 透视为宽表（价格 / 收益 / 市值）
    prices_w = df.pivot(index="Date", columns="Ticker", values="Close").sort_index()
    rets_w   = df.pivot(index="Date", columns="Ticker", values="Returns").sort_index()
    mv_w     = df.pivot(index="Date", columns="Ticker", values="Volume").sort_index()

    # ---- 4) 预览（前 10 行），把 Date 转成字符串展示更友好 ----
    preview_df = df.head(10).copy()
    preview_df["Date"] = preview_df["Date"].dt.strftime("%Y-%m-%d")
    preview_cols = [{"name": c, "id": c} for c in preview_df.columns]

    return (
        df.to_dict("records"),
        prices_w.reset_index().to_dict("records"),
        rets_w.reset_index().to_dict("records"),
        mv_w.reset_index().to_dict("records"),
        preview_df.to_dict("records"),
        preview_cols,
    )

@callback(
    Output("pf_date", "start_date"),
    Output("pf_date", "end_date"),
    Input("store-returns-wide", "data"),
    prevent_initial_call=True
)
def init_pf_dates(rets_w_data):
    if not rets_w_data:
        return no_update, no_update
    df = pd.DataFrame(rets_w_data).set_index("Date")
    df.index = pd.to_datetime(df.index)
    return df.index.min().date(), df.index.max().date()

@callback(
    Output("store-weights-wide", "data"),
    Input("store-volume-wide", "data"),
    prevent_initial_call=True
)
def build_weights(mv_w_data):
    if not mv_w_data:
        raise pd.errors.EmptyDataError("No MV data")
    mv = pd.DataFrame(mv_w_data).set_index("Date")
    mv.index = pd.to_datetime(mv.index)
    w = cf.compute_weights_eod(mv.set_axis(mv.columns, axis=1))
    return w.reset_index().to_dict("records")

@callback(
    Output("var_summary", "children"),
    Output("pf_perf", "figure"),
    Output("var_chart", "figure"),
    Output("pf-weights-heat", "figure"),
    Output("pf-weights-lines", "figure"),
    Input("store-returns-wide", "data"),
    Input("store-volume-wide", "data"),
    Input("store-weights-wide", "data"),
    Input("pf_conf", "value"),
    Input("pf_window", "value"),
    Input("pf_date", "start_date"),
    Input("pf_date", "end_date"),
    prevent_initial_call=True
)
def render_portfolio(rets_w_data, mv_w_data, w_w_data, cl, window, d0, d1):
    if not (rets_w_data and mv_w_data and w_w_data and d0 and d1):
        from dash import exceptions
        raise exceptions.PreventUpdate

    # --- 数据还原 ---
    rets = pd.DataFrame(rets_w_data).set_index("Date")
    mv   = pd.DataFrame(mv_w_data).set_index("Date")
    w    = pd.DataFrame(w_w_data).set_index("Date")

    for df in (rets, mv, w):
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)

    # 区间裁剪 & 对齐
    rets = rets.loc[str(d0):str(d1)]
    mv   = mv.loc[str(d0):str(d1)]
    w    = w.loc[str(d0):str(d1)]

    # 组合收益（用 MV 直接重算更稳，避免权重 store 与 MV 脱节）
    # 但为了响应性，我们按 store 的 w 来聚合；如需严谨可用 compute_portfolio_return(rets, mv)
    rets_aligned = rets.reindex_like(w).fillna(0.0)
    w_aligned    = w.fillna(0.0)
    port = (rets_aligned * w_aligned).sum(axis=1).dropna()
    port.name = "Portfolio"

    # --- Performance（累计收益曲线）---
    cum = (1 + port).cumprod()
    fig_perf = px.line(cum.to_frame("Cumulative"), labels={"index":"Date","value":"Cumulative NAV"})
    fig_perf.update_layout(title="Portfolio Cumulative Performance")

    # --- VaR ---
    var_value = cf.historical_var(port, cl=float(cl), window=int(window))
    rvar = cf.rolling_var(port, cl=float(cl), window=int(window))
    fig_var = px.line(port.to_frame("Return"))
    if not rvar.empty:
        fig_var.add_scatter(x=rvar.index, y=-rvar.values, mode="lines",
                            name=f"−VaR({int(cl*100)}%, {window}d)")
    fig_var.update_layout(title=f"Rolling VaR (CL={int(cl*100)}%, window={window}d)")

    summary = f"VaR({int(cl*100)}%, {window}d): {var_value:.4%}   |   Period: {pd.to_datetime(d0).date()} → {pd.to_datetime(d1).date()}"

    # --- 权重热力图（显示最近 N 天，避免太密）---
    N = 120  # 最近 120 个交易日
    w_recent = w_aligned.iloc[-N:] if len(w_aligned) > N else w_aligned
    # 为了热力图视觉清晰，按列顺序显示
    fig_heat = px.imshow(
        w_recent.T, aspect="auto",
        labels=dict(x="Date", y="Ticker", color="Weight"),
        origin="lower"
    )
    fig_heat.update_layout(title=f"Weight Heatmap (last {len(w_recent)} days)")

    # --- 权重变化折线（挑选权重平均绝对值最高的 TOP5）---
    mean_abs = w_aligned.abs().mean().sort_values(ascending=False)
    top = list(mean_abs.head(5).index)
    fig_wlines = px.line(w_aligned[top], labels={"value":"Weight","index":"Date"})
    fig_wlines.update_layout(title="Top-5 Weight Time Series (by |mean weight|)")

    return summary, fig_perf, fig_var, fig_heat, fig_wlines


#ARIMA 预测模块回调函数
@callback(
    Output("arima-summary", "children"),
    Output("pf-arima-risk", "figure"),
    Input("store-returns-wide", "data"),
    Input("store-weights-wide", "data"),
    Input("pf_date", "start_date"),
    Input("pf_date", "end_date"),
    prevent_initial_call=True
)
def render_arima(rets_w_data, w_w_data, d0, d1):
    if not (rets_w_data and w_w_data and d0 and d1):
        raise exceptions.PreventUpdate

    # 1) 还原宽表 & 组合收益（用已存好的 EOD 权重）
    rets = pd.DataFrame(rets_w_data).set_index("Date")
    w    = pd.DataFrame(w_w_data).set_index("Date")
    for df in (rets, w):
        df.index = pd.to_datetime(df.index); df.sort_index(inplace=True)

    rets = rets.loc[str(d0):str(d1)]
    w    = w.loc[str(d0):str(d1)]

    rets_aligned = rets.reindex_like(w).fillna(0.0)
    w_aligned    = w.fillna(0.0)
    port = (rets_aligned * w_aligned).sum(axis=1).dropna()
    if port.empty:
        raise exceptions.PreventUpdate

    # 2) 构造风险代理（20D年化波动率）+ 对齐日期
    risk_raw = cf.risk_proxy_from_returns(port, window=20, annualize=True)

    risk = pd.Series(
        risk_raw.values,
        index=port.index[-len(risk_raw):],   # 用最后 N 天的日期
        name="Risk"
    )

    # 3) ARIMA 预测（默认 5 日）
    yhat, ci, info = cf.arima_forecast(risk, horizon=5, max_p=3, max_q=3, d=None)

    # === 关键：给预测结果改成“未来日期”的 index ===
    last_dt = risk.index[-1]
    future_idx = pd.date_range(
        start=last_dt + pd.Timedelta(days=1),
        periods=len(yhat),
        freq="B"           # 或者 "D"，看你希望按交易日还是自然日
    )

    # 把 yhat 和 ci 都绑到 future_idx 上
    yhat = pd.Series(yhat.values, index=future_idx, name="Forecast")
    ci = pd.DataFrame(ci.values, index=future_idx, columns=ci.columns)

    # 4) 画图：历史风险 + 未来预测
    hist = risk  # 已经有名字 "Risk"
    fig = px.line(hist.to_frame(), labels={"index": "Date", "value": "Risk (ann.)"})

    fig.add_scatter(
        x=yhat.index, y=yhat.values,
        mode="lines+markers", name="Forecast"
    )

    fig.add_scatter(
        x=ci.index, y=ci.iloc[:, 0],
        mode="lines", line=dict(dash="dot"), name="Lower 95%"
    )
    fig.add_scatter(
        x=ci.index, y=ci.iloc[:, 1],
        mode="lines", line=dict(dash="dot"), name="Upper 95%"
    )

    fig.update_layout(title=f"ARIMA Forecast of Risk (order={info['order']}, AIC={info['aic']:.1f})")
    fig.update_xaxes(title="Date", type="date")

    summary = (
        f"Target: 20D annualized volatility | "
        f"ARIMA order={info['order']}, AIC={info['aic']:.1f} | "
        f"Last: {info['last_value']:.4f} | "
        f"T+1 forecast: {yhat.iloc[0]:.4f}"
    )
    return summary, fig

# ======== 回调：Stress Testing Tab 进入时后台加载 SPX 数据 ========
def _to_dt(x):
    return pd.to_datetime(x, errors="coerce")
@callback(
    Output("stress-summary", "children"),
    Output("stress-figure", "figure"),
    Input("tabs", "value"),
    Input("store-returns-wide", "data"),   # Tab2 算好的“资产日收益”宽表
    Input("store-weights-wide", "data"),   # Tab2 算好的 EOD 权重宽表
    Input("pf_date", "start_date"),        # Tab2 选择的区间
    Input("pf_date", "end_date"),
    Input("store-spx-returns", "data"),    # Tab3 背景加载的 SPX 日收益（列：Date, Return）
    Input("stress-scn", "value"),          # 选择的场景名（要是 cf.SCENARIOS 的 key）
    Input("beta_window", "value"),         # β 回看窗口（如 252）
    prevent_initial_call=False
)
def render_stress_only(tab_value,
                       rets_w_data, w_w_data, d0, d1,
                       spx_data, scn_key, beta_window):

    # 仅在 Tab3 运行
    if tab_value != "tab-stress":
        raise exceptions.PreventUpdate

    # 必要输入是否就绪（不就绪就先不更新，不报错）
    if not (rets_w_data and w_w_data and d0 and d1 and spx_data and scn_key):
        raise exceptions.PreventUpdate

    # ---------- 1) 还原数据 ----------
    rets_w = pd.DataFrame(rets_w_data).set_index("Date")
    w_w    = pd.DataFrame(w_w_data).set_index("Date")
    for df in (rets_w, w_w):
        df.index = _to_dt(df.index)
        df.sort_index(inplace=True)

    spx_df = pd.DataFrame(spx_data).copy()
    # 标准化列名（防止大小写不同）
    rename_map = {c: c.capitalize() for c in spx_df.columns}
    spx_df.rename(columns=rename_map, inplace=True)
    spx_df["Date"]   = _to_dt(spx_df["Date"])
    spx_df["Return"] = pd.to_numeric(spx_df["Return"], errors="coerce").fillna(0.0)
    spx_df = spx_df.dropna(subset=["Date"]).sort_values("Date")
    spx_full = pd.Series(spx_df["Return"].values, index=spx_df["Date"].values, name="SPX")

    # 与 Tab2 同区间（用于 β 和组合净值曲线）
    d0 = _to_dt(d0); d1 = _to_dt(d1)
    rets_w = rets_w.loc[d0:d1]
    w_w    = w_w.loc[d0:d1]

    # 组合日收益（EOD 权重聚合）
    if rets_w.empty or w_w.empty:
        return "⚠️ Empty returns/weights in selected range.", px.line()

    rets_aligned = rets_w.reindex_like(w_w).fillna(0.0)
    w_aligned    = w_w.fillna(0.0)
    port = (rets_aligned * w_aligned).sum(axis=1).dropna().rename("Portfolio")

    # β：用所选窗口滚动末值（样本不足就静态）
    beta_window = int(beta_window or 252)
    spx_for_beta = spx_full.loc[d0:d1]
    if len(port) >= beta_window and len(spx_for_beta) >= beta_window:
        cov = port.rolling(beta_window).cov(spx_for_beta)
        var = spx_for_beta.rolling(beta_window).var()
        beta = float((cov / var).iloc[-1])
    else:
        v = float(spx_for_beta.var()) if len(spx_for_beta) else np.nan
        beta = float(port.cov(spx_for_beta) / v) if v not in (None, 0, np.nan) else np.nan

    # ---------- 2) 读取场景 & 计算 SPX shock ----------
    scn = cf.SCENARIOS.get(scn_key)
    if scn is None:
        return f"⚠️ Scenario not found: {scn_key}", px.line()
    if isinstance(scn, dict):
        start, end = _to_dt(scn["start"]), _to_dt(scn["end"])
    else:
        start, end = _to_dt(scn[0]), _to_dt(scn[1])

    spx_slice = spx_full.loc[start:end]
    spx_shock = (1 + spx_slice).prod() - 1 if not spx_slice.empty else np.nan
    port_shock = beta * spx_shock if (np.isfinite(beta) and np.isfinite(spx_shock)) else np.nan

    # ---------- 3) 图：组合累计净值 + 场景阴影 + 冲击点 ----------
    cum = (1 + port).cumprod()
    fig = px.line(cum.to_frame("Cumulative NAV"))
    fig.update_layout(title="Portfolio Stress: Scenario Shock Only",
                      xaxis_title="Date", yaxis_title="Value")
    # 阴影标记场景区间（与当前显示区间无关，按场景时间）
    if pd.notna(start) and pd.notna(end):
        fig.add_vrect(x0=start, x1=end, fillcolor="red", opacity=0.15,
                      annotation_text=scn_key, annotation_position="top left")

    # 在曲线末端标出“冲击后”位置（β×SPX）
    if np.isfinite(port_shock) and len(cum) > 0:
        nav_last = float(cum.iloc[-1])
        nav_scn  = nav_last * (1 + port_shock)
        fig.add_scatter(x=[cum.index[-1], cum.index[-1]],
                        y=[nav_last, nav_scn],
                        mode="markers+text",
                        text=["Now", f"Shock\n{port_shock:.1%}"],
                        textposition="top center",
                        name="Scenario Shock")

    # ---------- 4) 摘要 ----------
    summary = (
        f"Scenario: {scn_key} | Range: {start.date()} → {end.date()} | "
        f"β={('NA' if not np.isfinite(beta) else f'{beta:.2f}')} | "
        f"SPX shock={('NA' if not np.isfinite(spx_shock) else f'{spx_shock:.2%}')} | "
        f"Portfolio shock(β×SPX)={('NA' if not np.isfinite(port_shock) else f'{port_shock:.2%}')}"
    )
    return summary, fig
@callback(
    Output("store-spx-returns", "data"),
    Input("init-once-stress", "n_intervals"),
    prevent_initial_call=False
)
def init_spx_returns(n):
    # 只在第一次触发时加载
    if n is None:
        from dash import exceptions
        raise exceptions.PreventUpdate

    # 读本地 CSV
    df_spx = pd.read_csv(SPX_LOCAL_PATH)

    # 这里假设你的 CSV 至少有日期和收益两列
    # 比如列名是：date, return 或 DATE, RETURN 等
    # 下面这段会全变成首字母大写：date -> Date, return -> Return
    rename_map = {c: c.capitalize() for c in df_spx.columns}
    df_spx.rename(columns=rename_map, inplace=True)

    # 如果你的收益那列叫别的名字，比如 'SPX_Return'，这里统一成 'Return'
    if "Return" not in df_spx.columns:
        # 尝试从常见命名里找一列
        for cand in ["Ret", "spx_return", "SPX_Return"]:
            if cand in df_spx.columns:
                df_spx["Return"] = df_spx[cand]
                break

    df_spx["Date"] = pd.to_datetime(df_spx["Date"], errors="coerce")
    df_spx["Return"] = pd.to_numeric(df_spx["Return"], errors="coerce")

    df_spx = df_spx.dropna(subset=["Date", "Return"]).sort_values("Date")

    return df_spx.to_dict("records")

if __name__=='__main__':

    app.run_server(debug=True)

