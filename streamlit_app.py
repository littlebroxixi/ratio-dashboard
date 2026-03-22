"""
指数比值套利监控看板 v4
IC/IF (中证500/沪深300) + IM/IC (中证1000/中证500) + IH/IC (上证50/中证500)
streamlit run streamlit_app.py
"""

import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

st.set_page_config(
    page_title="指数比值套利看板",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 盘中判断
now = datetime.now()
is_trading_hours = now.weekday() < 5 and (
    (now.hour == 9 and now.minute >= 30) or
    (10 <= now.hour <= 11) or
    (13 <= now.hour <= 14) or
    (now.hour == 11 and now.minute <= 30) or
    (now.hour == 15 and now.minute == 0)
)

# ============ 样式 ============
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }

.stApp { background: #080b12; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 0; max-width: 1600px; }

/* ===== 顶栏 ===== */
.topbar {
    display: flex; justify-content: space-between; align-items: center;
    padding: 16px 0 24px; border-bottom: 1px solid rgba(255,255,255,0.04);
    margin-bottom: 28px;
}
.topbar-left h1 {
    margin: 0; font-size: 1.6rem; font-weight: 700;
    background: linear-gradient(135deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.topbar-left p { margin: 4px 0 0; color: #4b5563; font-size: 0.8rem; }
.topbar-right {
    display: flex; align-items: center; gap: 16px; color: #6b7280; font-size: 0.8rem;
}
.live-dot {
    width: 8px; height: 8px; border-radius: 50%;
    display: inline-block; margin-right: 4px; animation: pulse 2s infinite;
}
.live-dot.on { background: #22c55e; box-shadow: 0 0 8px #22c55e; }
.live-dot.off { background: #6b7280; }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }

/* ===== 卡片 ===== */
.card {
    background: linear-gradient(145deg, rgba(17,24,39,0.9), rgba(15,20,35,0.95));
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 20px; padding: 28px 28px 20px;
    backdrop-filter: blur(20px);
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative; overflow: hidden;
}
.card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
}
.card:hover {
    border-color: rgba(96,165,250,0.3);
    box-shadow: 0 8px 32px rgba(96,165,250,0.08);
    transform: translateY(-2px);
}
.card-top { display: flex; justify-content: space-between; align-items: flex-start; }
.card-title { font-size: 1.3rem; font-weight: 700; color: #f1f5f9; letter-spacing: -0.02em; }
.card-desc { font-size: 0.72rem; color: #64748b; margin-top: 2px; }

/* 涨跌 badge */
.chg-badge {
    padding: 5px 14px; border-radius: 24px; font-size: 0.82rem; font-weight: 600;
}
.chg-up { background: rgba(239,68,68,0.12); color: #f87171; }
.chg-dn { background: rgba(34,197,94,0.12); color: #4ade80; }
.chg-flat { background: rgba(148,163,184,0.1); color: #94a3b8; }

/* 大数字 */
.big-num {
    font-size: 2.6rem; font-weight: 800; color: #f8fafc;
    margin: 12px 0 4px; letter-spacing: -0.03em;
    display: flex; align-items: baseline; gap: 14px;
}
.big-chg { font-size: 1rem; font-weight: 600; }
.big-chg.up { color: #f87171; }
.big-chg.dn { color: #4ade80; }

/* Z-Score 仪表条 */
.gauge-wrap { margin: 20px 0 8px; }
.gauge-labels {
    display: flex; justify-content: space-between;
    font-size: 0.62rem; color: #475569; margin-bottom: 8px;
    padding: 0 2px;
}
.gauge-labels span {
    width: 28px; text-align: center;
}
.gauge-track {
    position: relative; height: 10px; border-radius: 5px;
    background: rgba(255,255,255,0.04);
    overflow: visible; padding: 0;
}
.gauge-fill {
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    border-radius: 5px;
    background: linear-gradient(90deg,
        #22c55e 0%, #22c55e 14%,
        #3b9e6e 16.6%,
        #60a5fa 19%, #60a5fa 31%,
        #8a7ad4 33.3%,
        #ef4444 36%, #ef4444 64%,
        #8a7ad4 66.6%,
        #60a5fa 69%, #60a5fa 81%,
        #3b9e6e 83.3%,
        #22c55e 86%, #22c55e 100%
    );
    opacity: 0.85;
}
.gauge-ticks {
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    display: flex; justify-content: space-between;
    padding: 0;
}
.gauge-tick {
    width: 1px; height: 100%;
    background: rgba(255,255,255,0.12);
}
.gauge-tick:first-child, .gauge-tick:last-child { background: transparent; }
.gauge-ptr {
    position: absolute; top: 50%; width: 14px; height: 14px;
    border-radius: 50%; background: #fff;
    box-shadow: 0 0 10px rgba(255,255,255,0.5), 0 0 20px rgba(255,255,255,0.15);
    border: 2.5px solid #0f172a;
    transform: translate(-50%, -50%);
    z-index: 2;
}
.gauge-val {
    text-align: center; margin-top: 10px;
    font-size: 0.85rem; font-weight: 700;
}

/* 底部指标 */
.card-metrics {
    display: grid; grid-template-columns: repeat(4,1fr);
    gap: 8px; margin-top: 20px; padding-top: 16px;
    border-top: 1px solid rgba(255,255,255,0.04);
}
.cm-item { text-align: center; }
.cm-label { font-size: 0.65rem; color: #475569; margin-bottom: 3px; text-transform: uppercase; letter-spacing: 0.05em; }
.cm-val { font-size: 0.88rem; font-weight: 600; color: #cbd5e1; }

/* 方向提示 */
.signal-box {
    margin-top: 16px; padding: 14px 18px; border-radius: 14px;
    font-size: 0.82rem; font-weight: 500;
    display: flex; align-items: center; gap: 10px;
}
.signal-box.bullish, .signal-box.bearish {
    background: linear-gradient(135deg, rgba(239,68,68,0.08), rgba(239,68,68,0.02));
    border: 1px solid rgba(239,68,68,0.15); color: #fca5a5;
}
.signal-box.neutral {
    background: linear-gradient(135deg, rgba(100,116,139,0.08), rgba(100,116,139,0.02));
    border: 1px solid rgba(100,116,139,0.15); color: #94a3b8;
}
.signal-icon { font-size: 1.2rem; }

/* 按钮 */
.stButton > button {
    background: linear-gradient(135deg, rgba(96,165,250,0.1), rgba(167,139,250,0.1));
    border: 1px solid rgba(96,165,250,0.2); color: #93c5fd;
    border-radius: 12px; padding: 10px 20px; font-weight: 600;
    transition: all 0.3s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, rgba(96,165,250,0.2), rgba(167,139,250,0.2));
    border-color: rgba(96,165,250,0.4); transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(96,165,250,0.15);
}

/* 详情页返回按钮 */
.back-btn {
    display: inline-flex; align-items: center; gap: 6px;
    color: #64748b; font-size: 0.85rem; cursor: pointer;
    padding: 8px 0; transition: color 0.2s;
}
.back-btn:hover { color: #93c5fd; }

/* 详情页指标卡 */
.detail-metrics {
    display: grid; grid-template-columns: repeat(5,1fr);
    gap: 12px; margin: 20px 0;
}
.dm-card {
    background: rgba(15,23,42,0.6); border: 1px solid rgba(255,255,255,0.04);
    border-radius: 14px; padding: 16px; text-align: center;
}
.dm-label { font-size: 0.7rem; color: #475569; text-transform: uppercase; letter-spacing: 0.05em; }
.dm-val { font-size: 1.4rem; font-weight: 700; color: #e2e8f0; margin-top: 6px; }

/* ===== 操作建议面板 ===== */
.ops-panel {
    margin-top: 16px; padding: 0; display: flex; flex-direction: column; gap: 6px;
}
.ops-row {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 14px; border-radius: 10px;
    font-size: 0.78rem; color: #94a3b8;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.03);
}
.ops-row.active {
    background: rgba(34,197,94,0.08);
    border-color: rgba(34,197,94,0.2);
    color: #86efac;
}
.ops-row.done {
    background: rgba(96,165,250,0.06);
    border-color: rgba(96,165,250,0.15);
    color: #93c5fd;
}
.ops-row.waiting { color: #475569; }
.ops-dot {
    width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0;
}
.ops-dot.green { background: #22c55e; box-shadow: 0 0 6px rgba(34,197,94,0.5); }
.ops-dot.blue { background: #60a5fa; }
.ops-dot.gray { background: #334155; }
.ops-label { font-weight: 600; min-width: 48px; }
.ops-thresh { font-variant-numeric: tabular-nums; min-width: 42px; text-align: center; }
.ops-desc { flex: 1; }

/* ===== 小节标题 ===== */
.stMarkdown h5 {
    color: #e2e8f0 !important; font-size: 1rem !important;
    font-weight: 600 !important; letter-spacing: -0.01em;
    margin-bottom: 12px !important;
}

/* ===== 自定义表格 ===== */
.dark-table {
    width: 100%; border-collapse: separate; border-spacing: 0;
    border-radius: 14px; overflow: hidden;
    background: rgba(15,23,42,0.5);
    border: 1px solid rgba(255,255,255,0.05);
    font-size: 0.82rem;
}
.dark-table thead th {
    background: rgba(255,255,255,0.03);
    color: #64748b; font-weight: 600; font-size: 0.72rem;
    text-transform: uppercase; letter-spacing: 0.06em;
    padding: 12px 16px; text-align: left;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.dark-table tbody td {
    padding: 10px 16px; color: #cbd5e1;
    border-bottom: 1px solid rgba(255,255,255,0.025);
    transition: background 0.2s;
}
.dark-table tbody tr:last-child td { border-bottom: none; }
.dark-table tbody tr:hover td { background: rgba(96,165,250,0.04); }
.dark-table .num { font-variant-numeric: tabular-nums; font-weight: 500; color: #e2e8f0; }
.dark-table .z-pos { color: #f87171; font-weight: 600; }
.dark-table .z-neg { color: #4ade80; font-weight: 600; }
.dark-table .z-neutral { color: #94a3b8; font-weight: 500; }
.dark-table .highlight-row td { background: rgba(96,165,250,0.06); }
.dark-table .eff-best td { background: rgba(34,197,94,0.06); }

/* 隐藏 Streamlit 自带的 dataframe */
.stDataFrame { display: none !important; }

/* 首页列内紧凑间距 */
[data-testid="column"] [data-testid="stPlotlyChart"] {
    margin-top: -1rem; margin-bottom: -1rem;
}

/* 页脚 */
.footer {
    text-align: center; color: #334155; font-size: 0.7rem;
    padding: 24px 0 12px; margin-top: 24px;
    border-top: 1px solid rgba(255,255,255,0.03);
}
.footer a { color: #475569; text-decoration: none; }
</style>
""", unsafe_allow_html=True)


# ============ 数据获取 ============
@st.cache_data(ttl=60)
def get_realtime_quotes():
    try:
        df = ak.stock_zh_index_spot_sina()
        target = df[df['代码'].isin(['sh000300', 'sh000905', 'sh000852', 'sh000016'])]
        quotes = {}
        for _, row in target.iterrows():
            code = row['代码']
            if code == 'sh000300': quotes['hs300'] = float(row['最新价'])
            elif code == 'sh000905': quotes['zz500'] = float(row['最新价'])
            elif code == 'sh000852': quotes['zz1000'] = float(row['最新价'])
            elif code == 'sh000016': quotes['sz50'] = float(row['最新价'])
        return quotes
    except Exception:
        return None

@st.cache_data(ttl=86400)
def load_data():
    hs300 = ak.stock_zh_index_daily(symbol='sh000300')
    hs300['date'] = pd.to_datetime(hs300['date'])
    hs300 = hs300.set_index('date')[['close']].rename(columns={'close': 'hs300'})
    zz500 = ak.stock_zh_index_daily(symbol='sh000905')
    zz500['date'] = pd.to_datetime(zz500['date'])
    zz500 = zz500.set_index('date')[['close']].rename(columns={'close': 'zz500'})
    zz1000 = ak.stock_zh_index_daily(symbol='sh000852')
    zz1000['date'] = pd.to_datetime(zz1000['date'])
    zz1000 = zz1000.set_index('date')[['close']].rename(columns={'close': 'zz1000'})
    sz50 = ak.stock_zh_index_daily(symbol='sh000016')
    sz50['date'] = pd.to_datetime(sz50['date'])
    sz50 = sz50.set_index('date')[['close']].rename(columns={'close': 'sz50'})
    df = hs300.join(zz500, how='inner').join(zz1000, how='inner').join(sz50, how='inner')
    df = df[df.index >= '2020-01-01'].copy()
    df.dropna(inplace=True)
    return df


def calc_ratio(df, col_a, col_b):
    result = pd.DataFrame(index=df.index)
    result['ratio'] = df[col_a] / df[col_b]
    result['price_a'] = df[col_a]
    result['price_b'] = df[col_b]
    m, s = result['ratio'].mean(), result['ratio'].std()
    result['mean'] = m
    for i in [1, 2, 3]:
        result[f'upper_{i}'] = m + i * s
        result[f'lower_{i}'] = m - i * s
    result['zscore'] = (result['ratio'] - m) / s
    return result, m, s


def get_zone(z, pair):
    a = round(abs(z), 2)
    entry_thresh, _ = get_ops_thresholds(pair)
    # 从高到低匹配操作阈值
    for label, thresh, _ in reversed(entry_thresh):
        if a >= thresh:
            return label, "#22c55e"
    if a >= 1:
        return "关注", "#60a5fa"
    return "正常", "#94a3b8"


def get_signal(z, pair):
    if pair == "IC/IF":
        if z > 0: return "bearish", "↘", "中证500 相对偏强 — 做空 IC + 做多 IF"
        else: return "bullish", "↗", "中证500 相对偏弱 — 做多 IC + 做空 IF"
    elif pair == "IM/IC":
        if z > 0: return "bearish", "↘", "中证1000 相对偏强 — 做空 IM + 做多 IC"
        else: return "bullish", "↗", "中证1000 相对偏弱 — 做多 IM + 做空 IC"
    else:  # IH/IC
        if z > 0: return "bearish", "↘", "上证50 相对偏强 — 做空 IH + 做多 IC"
        else: return "bullish", "↗", "上证50 相对偏弱 — 做多 IH + 做空 IC"


def z_to_pct(z):
    """Z-Score 映射到 gauge 百分比 (0~100)"""
    return max(0, min(100, (z + 3) / 6 * 100))


def make_sparkline(data, height=130):
    recent = data.tail(120)  # 近半年
    z = recent.iloc[-1]['zscore']
    color = '#f87171' if z > 1 else '#4ade80' if z < -1 else '#60a5fa'

    fig = go.Figure()
    # ±2σ band
    fig.add_trace(go.Scatter(
        x=recent.index, y=recent['upper_2'], mode='lines',
        line=dict(color='rgba(96,165,250,0.3)', width=0.5, dash='dash'),
        showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=recent.index, y=recent['lower_2'], mode='lines',
        line=dict(color='rgba(96,165,250,0.3)', width=0.5, dash='dash'),
        fill='tonexty', fillcolor='rgba(96,165,250,0.04)',
        showlegend=False, hoverinfo='skip'
    ))
    # ±1σ band
    fig.add_trace(go.Scatter(
        x=recent.index, y=recent['upper_1'], mode='lines',
        line=dict(color='rgba(96,165,250,0.2)', width=0.5, dash='dot'),
        showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=recent.index, y=recent['lower_1'], mode='lines',
        line=dict(color='rgba(96,165,250,0.2)', width=0.5, dash='dot'),
        fill='tonexty', fillcolor='rgba(96,165,250,0.06)',
        showlegend=False, hoverinfo='skip'
    ))
    # ±3σ lines
    fig.add_trace(go.Scatter(
        x=recent.index, y=recent['upper_3'], mode='lines',
        line=dict(color='rgba(239,68,68,0.3)', width=0.5, dash='dash'),
        showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=recent.index, y=recent['lower_3'], mode='lines',
        line=dict(color='rgba(239,68,68,0.3)', width=0.5, dash='dash'),
        showlegend=False, hoverinfo='skip'
    ))
    # 均值线
    fig.add_trace(go.Scatter(
        x=recent.index, y=recent['mean'], mode='lines',
        line=dict(color='#a78bfa', width=1, dash='dot'), showlegend=False,
        hoverinfo='skip'
    ))
    # 比值线
    fig.add_trace(go.Scatter(
        x=recent.index, y=recent['ratio'], mode='lines',
        line=dict(color=color, width=2), showlegend=False,
        hovertemplate='%{y:.4f}<extra></extra>'
    ))
    # 最后一个点
    fig.add_trace(go.Scatter(
        x=[recent.index[-1]], y=[recent.iloc[-1]['ratio']], mode='markers',
        marker=dict(color=color, size=7, line=dict(color='#0f172a', width=2)),
        showlegend=False, hoverinfo='skip'
    ))
    # 右侧标注σ标签
    last = recent.iloc[-1]
    labels = [
        (last['upper_3'], '+3σ', '#ef4444'), (last['lower_3'], '-3σ', '#ef4444'),
        (last['upper_2'], '+2σ', '#60a5fa'), (last['lower_2'], '-2σ', '#60a5fa'),
        (last['upper_1'], '+1σ', '#475569'), (last['lower_1'], '-1σ', '#475569'),
        (last['mean'], '均值', '#a78bfa'),
    ]
    for yval, txt, clr in labels:
        fig.add_annotation(
            x=1, xref='paper', y=yval, yref='y',
            text=txt, showarrow=False,
            font=dict(size=9, color=clr),
            xanchor='left', xshift=4,
        )
    fig.update_layout(
        height=height, margin=dict(l=0,r=32,t=0,b=0),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            visible=False,
            showspikes=True, spikemode='across', spikesnap='cursor',
            spikethickness=1, spikecolor='rgba(148,163,184,0.35)', spikedash='4px 3px',
        ),
        yaxis=dict(visible=False),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='rgba(15,23,42,0.9)', bordercolor='rgba(255,255,255,0.1)',
            font=dict(color='#e2e8f0', size=12),
        ),
    )
    return fig


def make_detail_chart(data, title):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.72, 0.28], vertical_spacing=0.04)

    # 比值线
    fig.add_trace(go.Scatter(
        x=data.index, y=data['ratio'], name='比值',
        line=dict(color='#60a5fa', width=1.8)
    ), row=1, col=1)
    # 均值
    fig.add_trace(go.Scatter(
        x=data.index, y=data['mean'], name='均值',
        line=dict(color='#a78bfa', width=1, dash='dash')
    ), row=1, col=1)

    # 标准差通道
    bands = [
        (1, 'rgba(34,197,94,0.06)', 'rgba(34,197,94,0.4)'),
        (2, 'rgba(249,115,22,0.04)', 'rgba(249,115,22,0.3)'),
        (3, 'rgba(239,68,68,0.03)', 'rgba(239,68,68,0.25)'),
    ]
    for i, (n, fill, line_c) in enumerate(bands):
        fig.add_trace(go.Scatter(x=data.index, y=data[f'upper_{n}'],
            line=dict(color=line_c, width=0.5), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data[f'lower_{n}'],
            line=dict(color=line_c, width=0.5), fill='tonexty',
            fillcolor=fill, showlegend=False, name=f'±{n}σ'), row=1, col=1)

    # Z-Score
    colors = ['#ef4444' if abs(z)>=3 else '#f97316' if abs(z)>=2
              else '#22c55e' if abs(z)>=1 else '#334155' for z in data['zscore']]
    fig.add_trace(go.Bar(x=data.index, y=data['zscore'], name='Z-Score',
                         marker_color=colors, opacity=0.85), row=2, col=1)
    for v, c in [(2,'#f97316'),(-2,'#f97316'),(3,'#ef4444'),(-3,'#ef4444')]:
        fig.add_hline(y=v, line_dash="dot", line_color=c, line_width=0.5, row=2, col=1)
    fig.add_hline(y=0, line_color='#334155', line_width=0.5, row=2, col=1)

    fig.update_layout(
        title=dict(text=title, font=dict(color='#94a3b8', size=14), x=0),
        height=520, template='plotly_dark',
        paper_bgcolor='#0f1320', plot_bgcolor='#0f1320',
        legend=dict(orientation='h', y=1.06, font=dict(color='#64748b', size=11)),
        margin=dict(t=50, b=30, l=45, r=15),
        bargap=0.3,
    )
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.02)', zeroline=False)
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.02)', zeroline=False)
    fig.update_yaxes(title_text='比值', title_font=dict(color='#475569', size=11), row=1, col=1)
    fig.update_yaxes(title_text='Z-Score', range=[-4.5,4.5],
                     title_font=dict(color='#475569', size=11), row=2, col=1)
    return fig


PAIR_CONFIG = {
    'IC/IF': {'point_value': 100, 'margin': 200000},
    'IM/IC': {'point_value': 200, 'margin': 200000},
}

def calc_efficiency(data, pair=None):
    ANNUAL_OPP_RATE = 0.02  # 年化机会成本率 2%
    entry_levels = [3.0, 2.0]
    use_spread = pair in PAIR_CONFIG
    if use_spread:
        pv = PAIR_CONFIG[pair]['point_value']
        mg = PAIR_CONFIG[pair]['margin']
    results = []
    for entry_z in entry_levels:
        exit_candidates = np.arange(entry_z - 0.5, -0.1, -0.5)  # 3→2.5,2,1.5,1,0.5,0
        for exit_z in exit_candidates:
            exit_z = round(exit_z, 1)
            trades, in_trade = [], False
            for i in range(len(data)):
                z, date = data.iloc[i]['zscore'], data.index[i]
                if not in_trade:
                    if z >= entry_z:
                        in_trade, td, ed = True, 'short', date
                        if use_spread:
                            e_spread = data.iloc[i]['price_b'] - data.iloc[i]['price_a']
                        else:
                            er = data.iloc[i]['ratio']
                    elif z <= -entry_z:
                        in_trade, td, ed = True, 'long', date
                        if use_spread:
                            e_spread = data.iloc[i]['price_b'] - data.iloc[i]['price_a']
                        else:
                            er = data.iloc[i]['ratio']
                else:
                    rz = round(abs(z), 1)
                    ex = (td=='short' and rz<=exit_z) or (td=='long' and rz<=exit_z)
                    if ex:
                        h = len(data.loc[ed:date]) - 1
                        if use_spread:
                            x_spread = data.iloc[i]['price_b'] - data.iloc[i]['price_a']
                            # short: 做空比值 = 做多价差(B-A)，价差涨赚钱
                            # long:  做多比值 = 做空价差(B-A)，价差跌赚钱
                            spread_chg = (x_spread - e_spread) if td == 'short' else (e_spread - x_spread)
                            p = spread_chg * pv / mg * 100
                        else:
                            ratio = data.iloc[i]['ratio']
                            p = ((er - ratio) / er * 100) if td == 'short' else ((ratio - er) / er * 100)
                        trades.append({'h': h, 'p': p})
                        in_trade = False
            if trades:
                ap = np.mean([t['p'] for t in trades])
                ah = np.mean([t['h'] for t in trades])
                opp_cost = ah / 365 * ANNUAL_OPP_RATE * 100
                exit_label = f'{exit_z:g}σ' if exit_z > 0 else '均值'
                entry_label = f'{entry_z:g}σ'
                results.append({
                    '路径': f'{entry_label}→{exit_label}',
                    '样本数': len(trades),
                    '均收益%': f'{ap:.1f}%',
                    '均锁定天数': f'{int(ah)}天',
                    '机会成本%': f'{opp_cost:.1f}%',
                    '期望净收益%': f'{ap - opp_cost:.1f}%',
                })
    return pd.DataFrame(results)


def get_ops_thresholds(pair):
    """每个配对的操作阈值，基于历史回测结论"""
    if pair == "IC/IF":
        return [
            ('建仓', 2.4, '收益≈25% · 回撤<5.3%'),
            ('重仓', 2.8, '收益≈30% · 回撤<1.3%'),
            ('All-in', 3.0, '收益≈32% · 零回撤'),
        ], [
            ('减仓', 1.5, '已吃60%利润'),
            ('清仓', 0.5, '接近均值'),
        ]
    elif pair == "IM/IC":
        return [
            ('建仓', 1.8, '收益≈7.5% · 回撤<2.4%'),
            ('重仓', 2.4, '收益≈8.7% · 回撤<1.0%'),
            ('All-in', 3.0, '收益≈12% · 零回撤'),
        ], [
            ('减仓', 1.4, '7次均盈利'),
            ('清仓', 1.0, '效率走低'),
        ]
    else:  # IH/IC
        return [
            ('建仓', 2.4, '收益≈20% · 回撤<6.0%'),
            ('重仓', 2.8, '收益≈23% · 回撤<2.8%'),
            ('All-in', 3.0, '收益≈24% · 回撤≈0'),
        ], [
            ('减仓', 2.0, '效率峰值'),
            ('清仓', 1.5, '效率走低'),
        ]


def render_card(name, subtitle, data, pair):
    latest, prev = data.iloc[-1], data.iloc[-2]
    z, ratio = latest['zscore'], latest['ratio']
    chg = ratio - prev['ratio']
    chg_pct = chg / prev['ratio'] * 100
    zone_name, zone_color = get_zone(z, pair)

    # Badge
    if chg > 0:
        badge = f'<span class="chg-badge chg-up">▲ {chg_pct:.2f}%</span>'
        chg_html = f'<span class="big-chg up">+{chg:.4f}</span>'
    elif chg < 0:
        badge = f'<span class="chg-badge chg-dn">▼ {abs(chg_pct):.2f}%</span>'
        chg_html = f'<span class="big-chg dn">{chg:.4f}</span>'
    else:
        badge = '<span class="chg-badge chg-flat">— 0.00%</span>'
        chg_html = ''

    ptr_pct = z_to_pct(z)

    # ---- 顶部卡片：标题 + 大数字 ----
    st.markdown(f"""
    <div class="card">
        <div class="card-top">
            <div>
                <div class="card-title">{name}</div>
                <div class="card-desc">{subtitle}</div>
            </div>
            {badge}
        </div>
        <div class="big-num">{ratio:.4f} {chg_html}</div>
    </div>
    """, unsafe_allow_html=True)

    # ---- 近半年偏离折线图 ----
    st.plotly_chart(make_sparkline(data, height=130),
                    use_container_width=True,
                    config={'displayModeBar': False})

    # ---- 合并卡片：gauge + 操作建议 ----
    abs_z = round(abs(z), 2)
    entry_thresh, exit_thresh = get_ops_thresholds(pair)

    rows_html = ""
    for label, thresh, desc in entry_thresh:
        if abs_z >= thresh:
            dot_cls = "green"
            row_cls = "active" if abs_z < (thresh + 0.4) else "done"
        else:
            dot_cls = "gray"
            row_cls = "waiting"
        rows_html += f'<div class="ops-row {row_cls}"><span class="ops-dot {dot_cls}"></span><span class="ops-label">{label}</span><span class="ops-thresh">±{thresh}σ</span><span class="ops-desc">{desc}</span></div>'

    for label, thresh, desc in exit_thresh:
        if abs_z >= entry_thresh[0][1] and abs_z <= thresh:
            dot_cls = "green"
            row_cls = "active"
        else:
            dot_cls = "gray"
            row_cls = "waiting"
        rows_html += f'<div class="ops-row {row_cls}"><span class="ops-dot {dot_cls}"></span><span class="ops-label">{label}</span><span class="ops-thresh">±{thresh}σ</span><span class="ops-desc">{desc}</span></div>'

    st.markdown(f"""
    <div class="card" style="padding: 20px 24px 16px;">
        <div class="gauge-wrap" style="margin-top: 0;">
            <div class="gauge-labels">
                <span>-3σ</span><span>-2σ</span><span>-1σ</span>
                <span>均值</span>
                <span>+1σ</span><span>+2σ</span><span>+3σ</span>
            </div>
            <div class="gauge-track">
                <div class="gauge-fill"></div>
                <div class="gauge-ticks">
                    <span class="gauge-tick"></span>
                    <span class="gauge-tick"></span>
                    <span class="gauge-tick"></span>
                    <span class="gauge-tick"></span>
                    <span class="gauge-tick"></span>
                    <span class="gauge-tick"></span>
                    <span class="gauge-tick"></span>
                </div>
                <div class="gauge-ptr" style="left:{ptr_pct}%"></div>
            </div>
            <div class="gauge-val" style="color:{zone_color}">
                Z-Score: {z:+.2f}σ · {zone_name}区间
            </div>
        </div>
        <div class="ops-panel" style="margin-top: 14px;">{rows_html}</div>
    </div>
    """, unsafe_allow_html=True)

    # ---- 信号提示 ----
    if abs(z) >= 1.5:
        sig_type, icon, text = get_signal(z, pair)
    else:
        sig_type, icon, text = "neutral", "—", "比值在正常范围内，暂无操作信号"
    st.markdown(f"""
    <div class="signal-box {sig_type}">
        <span class="signal-icon">{icon}</span> {text}
    </div>
    """, unsafe_allow_html=True)


def render_detail(name, subtitle, data, mean, std, pair):
    latest = data.iloc[-1]
    z = latest['zscore']
    zone_name, zone_color = get_zone(z, pair)

    st.markdown(f"""
    <div class="detail-metrics">
        <div class="dm-card"><div class="dm-label">当前比值</div><div class="dm-val">{latest['ratio']:.4f}</div></div>
        <div class="dm-card"><div class="dm-label">Z-Score</div><div class="dm-val" style="color:{zone_color}">{z:+.2f}σ</div></div>
        <div class="dm-card"><div class="dm-label">均值</div><div class="dm-val">{mean:.4f}</div></div>
        <div class="dm-card"><div class="dm-label">标准差</div><div class="dm-val">{std:.4f}</div></div>
        <div class="dm-card"><div class="dm-label">区间</div><div class="dm-val" style="color:{zone_color}">{zone_name}</div></div>
    </div>
    """, unsafe_allow_html=True)

    if abs(z) >= 1.5:
        sig_type, icon, text = get_signal(z, pair)
        st.markdown(f'<div class="signal-box {sig_type}"><span class="signal-icon">{icon}</span> {text}</div>',
                    unsafe_allow_html=True)

    st.plotly_chart(make_detail_chart(data, f"{name} 比值走势 + 标准差通道"), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### 标准差通道")
        bands_data = [
            ('+3σ', latest['upper_3']), ('+2σ', latest['upper_2']),
            ('+1σ', latest['upper_1']), ('均值', latest['mean']),
            ('-1σ', latest['lower_1']), ('-2σ', latest['lower_2']),
            ('-3σ', latest['lower_3']),
        ]
        rows_html = ""
        for label, val in bands_data:
            hl = ' class="highlight-row"' if label == '均值' else ''
            rows_html += f'<tr{hl}><td>{label}</td><td class="num">{val:.4f}</td></tr>'
        st.markdown(f"""
        <table class="dark-table">
            <thead><tr><th>通道</th><th>数值</th></tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("##### 最近 10 个交易日")
        recent = data.tail(10)
        rows_html = ""
        for date, row in recent.iterrows():
            zv = row['zscore']
            z_cls = 'z-pos' if zv > 0.5 else 'z-neg' if zv < -0.5 else 'z-neutral'
            rows_html += f'<tr><td>{date.strftime("%Y-%m-%d")}</td><td class="num">{row["ratio"]:.4f}</td><td class="{z_cls}">{zv:+.2f}σ</td></tr>'
        st.markdown(f"""
        <table class="dark-table">
            <thead><tr><th>日期</th><th>比值</th><th>Z-Score</th></tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
        """, unsafe_allow_html=True)

    st.markdown("##### 赚钱效率分析")
    eff = calc_efficiency(data, pair)
    eff_rows = ""
    for _, row in eff.iterrows():
        eff_rows += f'<tr><td>{row["路径"]}</td><td class="num">{row["样本数"]}</td><td class="num">{row["均收益%"]}</td><td class="num">{row["均锁定天数"]}</td><td class="num">{row["机会成本%"]}</td><td class="num" style="color:#e0e0e0;font-weight:700">{row["期望净收益%"]}</td></tr>'
    st.markdown(f"""
    <table class="dark-table">
        <thead><tr><th>路径</th><th>样本数</th><th>均收益%</th><th>均锁定天数</th><th>机会成本%</th><th>期望净收益%</th></tr></thead>
        <tbody>{eff_rows}</tbody>
    </table>
    """, unsafe_allow_html=True)


# ============ 主逻辑 ============
if 'page' not in st.session_state:
    st.session_state.page = 'home'

def _load_all():
    """加载数据并计算比值，结果存入 session_state"""
    df = load_data()
    realtime = get_realtime_quotes()
    is_realtime = False
    if realtime and len(realtime) == 4:
        today = pd.Timestamp(datetime.now().date())
        row_data = {'hs300': realtime['hs300'], 'zz500': realtime['zz500'],
                    'zz1000': realtime['zz1000'], 'sz50': realtime['sz50']}
        if today not in df.index:
            df = pd.concat([df, pd.DataFrame(row_data, index=[today])])
            is_realtime = True
        else:
            df.loc[today] = row_data
            is_realtime = True
    st.session_state._df = df
    st.session_state._is_realtime = is_realtime
    st.session_state._ic_if = calc_ratio(df, 'zz500', 'hs300')
    st.session_state._im_ic = calc_ratio(df, 'zz1000', 'zz500')
    st.session_state._ih_ic = calc_ratio(df, 'sz50', 'zz500')

# 首次加载或手动刷新时才请求数据
if '_df' not in st.session_state:
    with st.spinner("加载数据中..."):
        _load_all()

df = st.session_state._df
is_realtime = st.session_state._is_realtime
ic_if, ic_if_mean, ic_if_std = st.session_state._ic_if
im_ic, im_ic_mean, im_ic_std = st.session_state._im_ic
ih_ic, ih_ic_mean, ih_ic_std = st.session_state._ih_ic

# ============ 首页 ============
if st.session_state.page == 'home':
    rt_html = '<span class="live-dot on"></span> 盘中实时' if is_realtime else '<span class="live-dot off"></span> 收盘数据'
    st.markdown(f"""
    <div class="topbar">
        <div class="topbar-left">
            <h1>📊 指数比值套利看板</h1>
            <p>均值回归策略 · 实时监控 · 辅助决策</p>
        </div>
        <div class="topbar-right">
            <span>{rt_html}</span>
            <span>数据截至 {df.index[-1].strftime('%Y-%m-%d')}</span>
            <span>{len(df)} 个交易日</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    _, btn_col = st.columns([9, 1])
    with btn_col:
        if st.button("🔄 刷新", key="refresh_top", use_container_width=True):
            get_realtime_quotes.clear()
            for k in ['_df','_is_realtime','_ic_if','_im_ic','_ih_ic']:
                st.session_state.pop(k, None)
            st.rerun()

    col1, col2, col3 = st.columns(3, gap="medium")
    with col1:
        render_card("IC / IF", "中证500 / 沪深300", ic_if, "IC/IF")
        if st.button("查看详细分析 →", key="b1", use_container_width=True):
            st.session_state.page = 'ic_if'; st.rerun()
    with col2:
        render_card("IM / IC", "中证1000 / 中证500", im_ic, "IM/IC")
        if st.button("查看详细分析 →", key="b2", use_container_width=True):
            st.session_state.page = 'im_ic'; st.rerun()
    with col3:
        render_card("IH / IC", "上证50 / 中证500", ih_ic, "IH/IC")
        if st.button("查看详细分析 →", key="b3", use_container_width=True):
            st.session_state.page = 'ih_ic'; st.rerun()

    st.markdown('<div class="footer">数据来源: AKShare · 仅供学习参考，不构成投资建议</div>', unsafe_allow_html=True)

# ============ 详情页 ============
elif st.session_state.page == 'ic_if':
    if st.button("← 返回首页"): st.session_state.page = 'home'; st.rerun()
    st.markdown(f'<div class="topbar-left"><h1>IC / IF 详细分析</h1><p>中证500 / 沪深300 · 2020年至今 · 全局标准差</p></div>', unsafe_allow_html=True)
    render_detail("IC/IF", "中证500/沪深300", ic_if, ic_if_mean, ic_if_std, "IC/IF")

elif st.session_state.page == 'im_ic':
    if st.button("← 返回首页"): st.session_state.page = 'home'; st.rerun()
    st.markdown(f'<div class="topbar-left"><h1>IM / IC 详细分析</h1><p>中证1000 / 中证500 · 2020年至今 · 全局标准差</p></div>', unsafe_allow_html=True)
    render_detail("IM/IC", "中证1000/中证500", im_ic, im_ic_mean, im_ic_std, "IM/IC")

elif st.session_state.page == 'ih_ic':
    if st.button("← 返回首页"): st.session_state.page = 'home'; st.rerun()
    st.markdown(f'<div class="topbar-left"><h1>IH / IC 详细分析</h1><p>上证50 / 中证500 · 2020年至今 · 全局标准差</p></div>', unsafe_allow_html=True)
    render_detail("IH/IC", "上证50/中证500", ih_ic, ih_ic_mean, ih_ic_std, "IH/IC")
