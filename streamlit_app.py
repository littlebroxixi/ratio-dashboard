"""
指数比值套利监控看板 v2
IC/IF (中证500/沪深300) + IM/IC (中证1000/中证500)
streamlit run ratio_dashboard.py
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

# 盘中自动刷新（每60秒）
now = datetime.now()
is_trading_hours = now.weekday() < 5 and (
    (now.hour == 9 and now.minute >= 30) or
    (10 <= now.hour <= 11) or
    (13 <= now.hour <= 14) or
    (now.hour == 11 and now.minute <= 30) or
    (now.hour == 15 and now.minute == 0)
)
if is_trading_hours:
    st.cache_data.clear()
    st_autorefresh_interval = 60
    st.markdown(
        f'<meta http-equiv="refresh" content="{st_autorefresh_interval}">',
        unsafe_allow_html=True
    )

# ============ 全局样式 ============
st.markdown("""
<style>
    /* 深色主题 */
    .stApp {
        background-color: #0a0e17;
    }

    /* 隐藏默认元素 */
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {padding-top: 1rem; padding-bottom: 0;}

    /* 标题 */
    .main-title {
        color: #e0e0e0;
        font-size: 1.5rem;
        font-weight: 600;
        padding: 0.5rem 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .main-subtitle {
        color: #6b7280;
        font-size: 0.85rem;
        margin-bottom: 1rem;
    }

    /* 卡片容器 */
    .ratio-card {
        background: linear-gradient(135deg, #131722 0%, #1a1f2e 100%);
        border: 1px solid #2a2e3e;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
        transition: all 0.3s ease;
    }
    .ratio-card:hover {
        border-color: #3b82f6;
        box-shadow: 0 0 20px rgba(59,130,246,0.1);
    }

    /* 卡片标题行 */
    .card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 4px;
    }
    .card-name {
        color: #f0f0f0;
        font-size: 1.2rem;
        font-weight: 700;
    }
    .card-sub {
        color: #6b7280;
        font-size: 0.75rem;
        margin-bottom: 16px;
    }

    /* 涨跌标签 */
    .badge-up {
        background: rgba(239,68,68,0.15);
        color: #ef4444;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-down {
        background: rgba(34,197,94,0.15);
        color: #22c55e;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-neutral {
        background: rgba(156,163,175,0.15);
        color: #9ca3af;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    /* 比值大数字 */
    .ratio-value {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 4px 0;
        display: flex;
        align-items: baseline;
        gap: 12px;
    }
    .ratio-change-up {color: #ef4444; font-size: 1rem; font-weight: 600;}
    .ratio-change-down {color: #22c55e; font-size: 1rem; font-weight: 600;}

    /* 区间标签 */
    .zone-tag {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 8px;
    }
    .zone-normal {background: rgba(34,197,94,0.15); color: #22c55e;}
    .zone-watch {background: rgba(234,179,8,0.15); color: #eab308;}
    .zone-trade {background: rgba(249,115,22,0.15); color: #f97316;}
    .zone-extreme {background: rgba(239,68,68,0.15); color: #ef4444;}

    /* 底部指标行 */
    .metrics-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-top: 16px;
        padding-top: 16px;
        border-top: 1px solid #2a2e3e;
    }
    .metric-item {
        text-align: center;
    }
    .metric-label {
        color: #6b7280;
        font-size: 0.7rem;
        margin-bottom: 4px;
    }
    .metric-value {
        color: #d1d5db;
        font-size: 0.9rem;
        font-weight: 600;
    }

    /* 方向建议 */
    .direction-box {
        background: rgba(59,130,246,0.08);
        border: 1px solid rgba(59,130,246,0.2);
        border-radius: 10px;
        padding: 12px 16px;
        margin-top: 12px;
        color: #93c5fd;
        font-size: 0.85rem;
    }

    /* 顶部统计栏 */
    .stats-bar {
        display: flex;
        align-items: center;
        gap: 8px;
        color: #9ca3af;
        font-size: 0.85rem;
        margin-bottom: 1rem;
        padding: 8px 0;
    }

    /* 刷新按钮 */
    .stButton > button {
        background: transparent;
        border: 1px solid #3b4252;
        color: #9ca3af;
        border-radius: 8px;
        padding: 4px 16px;
        font-size: 0.8rem;
    }
    .stButton > button:hover {
        border-color: #3b82f6;
        color: #3b82f6;
    }

    /* Tab样式 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: #131722;
        border-radius: 12px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #6b7280;
        border-radius: 8px;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: #1e2433;
        color: #f0f0f0;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: #131722;
        color: #9ca3af;
        border-radius: 8px;
    }

    /* Dataframe */
    .stDataFrame {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ============ 数据获取 ============
@st.cache_data(ttl=60)
def get_realtime_quotes():
    """获取盘中实时行情（新浪源），缓存60秒"""
    try:
        df = ak.stock_zh_index_spot_sina()
        target = df[df['代码'].isin(['sh000300', 'sh000905', 'sh000852'])]
        quotes = {}
        for _, row in target.iterrows():
            code = row['代码']
            if code == 'sh000300':
                quotes['hs300'] = float(row['最新价'])
            elif code == 'sh000905':
                quotes['zz500'] = float(row['最新价'])
            elif code == 'sh000852':
                quotes['zz1000'] = float(row['最新价'])
        return quotes
    except Exception:
        return None

@st.cache_data(ttl=3600)
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

    df = hs300.join(zz500, how='inner').join(zz1000, how='inner')
    df = df[df.index >= '2020-01-01'].copy()
    df.dropna(inplace=True)
    return df


def calc_ratio(df, col_a, col_b):
    result = pd.DataFrame(index=df.index)
    result['ratio'] = df[col_a] / df[col_b]
    global_mean = result['ratio'].mean()
    global_std = result['ratio'].std()
    result['mean'] = global_mean
    result['upper_1'] = global_mean + 1 * global_std
    result['lower_1'] = global_mean - 1 * global_std
    result['upper_2'] = global_mean + 2 * global_std
    result['lower_2'] = global_mean - 2 * global_std
    result['upper_3'] = global_mean + 3 * global_std
    result['lower_3'] = global_mean - 3 * global_std
    result['zscore'] = (result['ratio'] - global_mean) / global_std
    return result, global_mean, global_std


def get_zone_info(z):
    abs_z = abs(z)
    if abs_z < 1:
        return "正常", "zone-normal", "100%"
    elif abs_z < 2:
        return "关注", "zone-watch", f"{int(100 - (abs_z-1)*20)}%"
    elif abs_z < 3:
        return "建仓", "zone-trade", f"{int(60 - (abs_z-2)*20)}%"
    else:
        return "极端", "zone-extreme", f"{max(5, int(40 - (abs_z-3)*15))}%"


def get_direction(z, pair):
    if pair == "IC/IF":
        if z > 0:
            return "💡 中证500 相对偏强 → 做空IC + 做多IF"
        else:
            return "💡 中证500 相对偏弱 → 做多IC + 做空IF"
    else:
        if z > 0:
            return "💡 中证1000 相对偏强 → 做空IM + 做多IC"
        else:
            return "💡 中证1000 相对偏弱 → 做多IM + 做空IC"


def make_sparkline(data, height=80):
    """卡片内的迷你走势图"""
    recent = data.tail(60)
    fig = go.Figure()

    # 均值线
    fig.add_trace(go.Scatter(
        x=recent.index, y=recent['mean'],
        mode='lines', line=dict(color='#eab308', width=1, dash='dot'),
        showlegend=False
    ))

    # 比值线 - 根据最新zscore决定颜色
    z = recent.iloc[-1]['zscore']
    color = '#ef4444' if z > 0 else '#22c55e' if z < 0 else '#9ca3af'
    fig.add_trace(go.Scatter(
        x=recent.index, y=recent['ratio'],
        mode='lines', line=dict(color=color, width=2),
        showlegend=False
    ))

    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig


def make_detail_chart(data, title):
    """详情页的完整图表"""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3], vertical_spacing=0.05
    )

    fig.add_trace(go.Scatter(
        x=data.index, y=data['ratio'],
        name='比值', line=dict(color='#60a5fa', width=1.5)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=data.index, y=data['mean'],
        name='均值', line=dict(color='#eab308', width=1, dash='dash')
    ), row=1, col=1)

    # ±1σ
    fig.add_trace(go.Scatter(
        x=data.index, y=data['upper_1'], name='+1σ',
        line=dict(color='rgba(34,197,94,0.5)', width=0.5), showlegend=False
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=data.index, y=data['lower_1'], name='-1σ',
        line=dict(color='rgba(34,197,94,0.5)', width=0.5),
        fill='tonexty', fillcolor='rgba(34,197,94,0.05)', showlegend=False
    ), row=1, col=1)

    # ±2σ
    fig.add_trace(go.Scatter(
        x=data.index, y=data['upper_2'], name='+2σ',
        line=dict(color='rgba(249,115,22,0.5)', width=0.5), showlegend=False
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=data.index, y=data['lower_2'], name='-2σ',
        line=dict(color='rgba(249,115,22,0.5)', width=0.5),
        fill='tonexty', fillcolor='rgba(249,115,22,0.04)', showlegend=False
    ), row=1, col=1)

    # ±3σ
    fig.add_trace(go.Scatter(
        x=data.index, y=data['upper_3'], name='+3σ',
        line=dict(color='rgba(239,68,68,0.5)', width=0.5), showlegend=False
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=data.index, y=data['lower_3'], name='-3σ',
        line=dict(color='rgba(239,68,68,0.5)', width=0.5),
        fill='tonexty', fillcolor='rgba(239,68,68,0.03)', showlegend=False
    ), row=1, col=1)

    # Z-Score
    colors = ['#ef4444' if abs(z) >= 3 else '#f97316' if abs(z) >= 2
              else '#22c55e' if abs(z) >= 1 else '#4b5563' for z in data['zscore']]
    fig.add_trace(go.Bar(
        x=data.index, y=data['zscore'],
        name='Z-Score', marker_color=colors, opacity=0.8
    ), row=2, col=1)

    for val in [2, -2]:
        fig.add_hline(y=val, line_dash="dash", line_color="#f97316",
                      line_width=0.5, row=2, col=1)
    for val in [3, -3]:
        fig.add_hline(y=val, line_dash="dash", line_color="#ef4444",
                      line_width=0.5, row=2, col=1)
    fig.add_hline(y=0, line_color="#4b5563", line_width=0.5, row=2, col=1)

    fig.update_layout(
        title=dict(text=title, font=dict(color='#e0e0e0', size=16)),
        height=550, template='plotly_dark',
        paper_bgcolor='#131722', plot_bgcolor='#131722',
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    font=dict(color='#9ca3af')),
        margin=dict(t=60, b=40, l=50, r=20)
    )
    fig.update_xaxes(gridcolor='#1e2433', zeroline=False)
    fig.update_yaxes(gridcolor='#1e2433', zeroline=False)
    fig.update_yaxes(title_text='比值', title_font=dict(color='#6b7280'), row=1, col=1)
    fig.update_yaxes(title_text='Z-Score', range=[-4, 4],
                     title_font=dict(color='#6b7280'), row=2, col=1)
    return fig


def calc_efficiency(data):
    entry_levels = [1.0, 1.5, 2.0, 2.5, 3.0]
    exit_levels = [0, 0.5, 1.0]
    results = []
    for entry_z in entry_levels:
        for exit_z in exit_levels:
            if exit_z >= entry_z:
                continue
            trades = []
            in_trade = False
            for i in range(len(data)):
                z = data.iloc[i]['zscore']
                ratio = data.iloc[i]['ratio']
                date = data.index[i]
                if not in_trade:
                    if z >= entry_z:
                        in_trade, trade_dir = True, 'short'
                        entry_date, entry_ratio = date, ratio
                    elif z <= -entry_z:
                        in_trade, trade_dir = True, 'long'
                        entry_date, entry_ratio = date, ratio
                else:
                    should_exit = (trade_dir == 'short' and z <= exit_z) or \
                                  (trade_dir == 'long' and z >= -exit_z)
                    if should_exit:
                        hold = len(data.loc[entry_date:date]) - 1
                        pnl = ((entry_ratio - ratio) / entry_ratio * 100) if trade_dir == 'short' \
                            else ((ratio - entry_ratio) / entry_ratio * 100)
                        trades.append({'hold': hold, 'pnl': pnl})
                        in_trade = False
            if trades:
                wins = len([t for t in trades if t['pnl'] > 0])
                avg_pnl = np.mean([t['pnl'] for t in trades])
                avg_hold = np.mean([t['hold'] for t in trades])
                eff = avg_pnl / avg_hold * 100 if avg_hold > 0 else 0
                results.append({
                    '入场': f'±{entry_z}σ', '出场': f'±{exit_z}σ' if exit_z > 0 else '均值',
                    '次数': len(trades), '胜率': f'{wins/len(trades)*100:.0f}%',
                    '平均收益%': round(avg_pnl, 2), '持有日': int(avg_hold),
                    '总收益%': round(sum(t['pnl'] for t in trades), 2),
                    '效率': round(eff, 2)
                })
    return pd.DataFrame(results).sort_values('效率', ascending=False).reset_index(drop=True)


def render_card(name, subtitle, data, pair):
    """渲染单个卡片"""
    latest = data.iloc[-1]
    prev = data.iloc[-2]
    z = latest['zscore']
    ratio = latest['ratio']
    change = ratio - prev['ratio']
    change_pct = change / prev['ratio'] * 100

    zone_name, zone_class, strength = get_zone_info(z)

    # 涨跌badge
    if change > 0:
        badge = f'<span class="badge-up">▲ {change_pct:.2f}%</span>'
        change_html = f'<span class="ratio-change-up">+{change:.4f}</span>'
    elif change < 0:
        badge = f'<span class="badge-down">▼ {change_pct:.2f}%</span>'
        change_html = f'<span class="ratio-change-down">{change:.4f}</span>'
    else:
        badge = f'<span class="badge-neutral">- 0.00%</span>'
        change_html = f'<span style="color:#9ca3af">0.0000</span>'

    card_html = f"""
    <div class="ratio-card">
        <div class="card-header">
            <span class="card-name">{name}</span>
            {badge}
        </div>
        <div class="card-sub">{subtitle}</div>
        <div class="ratio-value">
            {ratio:.4f}
            {change_html}
            <span class="zone-tag {zone_class}">强度 {strength}</span>
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

    # 迷你走势图
    st.plotly_chart(make_sparkline(data, height=100), use_container_width=True,
                    config={'displayModeBar': False})

    # 底部指标
    metrics_html = f"""
    <div class="metrics-row">
        <div class="metric-item">
            <div class="metric-label">Z-Score</div>
            <div class="metric-value">{z:+.2f}σ</div>
        </div>
        <div class="metric-item">
            <div class="metric-label">区间</div>
            <div class="metric-value">{zone_name}</div>
        </div>
        <div class="metric-item">
            <div class="metric-label">均值</div>
            <div class="metric-value">{latest['mean']:.4f}</div>
        </div>
        <div class="metric-item">
            <div class="metric-label">±3σ范围</div>
            <div class="metric-value">{latest['lower_3']:.2f}~{latest['upper_3']:.2f}</div>
        </div>
    </div>
    """
    st.markdown(metrics_html, unsafe_allow_html=True)

    # 方向建议
    if abs(z) >= 1.5:
        st.markdown(f'<div class="direction-box">{get_direction(z, pair)}</div>',
                    unsafe_allow_html=True)


# ============ 主页面 ============

# 初始化页面状态
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# 加载数据
with st.spinner("加载中..."):
    df = load_data()

    # 盘中实时数据：追加当天最新价到历史数据末尾
    realtime = get_realtime_quotes()
    is_realtime = False
    if realtime and len(realtime) == 3:
        today = pd.Timestamp(datetime.now().date())
        if today not in df.index:
            new_row = pd.DataFrame({
                'hs300': [realtime['hs300']],
                'zz500': [realtime['zz500']],
                'zz1000': [realtime['zz1000']]
            }, index=[today])
            df = pd.concat([df, new_row])
            is_realtime = True
        else:
            # 今天已有日线数据，用实时价覆盖
            df.loc[today, 'hs300'] = realtime['hs300']
            df.loc[today, 'zz500'] = realtime['zz500']
            df.loc[today, 'zz1000'] = realtime['zz1000']
            is_realtime = True

ic_if, ic_if_mean, ic_if_std = calc_ratio(df, 'zz500', 'hs300')
im_ic, im_ic_mean, im_ic_std = calc_ratio(df, 'zz1000', 'zz500')

# ============ 首页 ============
if st.session_state.page == 'home':
    # 标题
    st.markdown('<div class="main-title">📊 指数比值套利看板</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">均值回归策略 · 实时监控 · 辅助决策</div>',
                unsafe_allow_html=True)

    # 顶部栏
    top_col1, top_col2 = st.columns([8, 2])
    with top_col1:
        rt_tag = ' &nbsp;|&nbsp; <span style="color:#22c55e">● 盘中实时</span>' if is_realtime else ''
        st.markdown(
            f'<div class="stats-bar">⏱ 共筛选出 <b style="color:#3b82f6">2</b> 组监控配对 &nbsp;|&nbsp; '
            f'数据截至 {df.index[-1].strftime("%Y-%m-%d")} &nbsp;|&nbsp; '
            f'{len(df)} 个交易日{rt_tag}</div>',
            unsafe_allow_html=True
        )
    with top_col2:
        if st.button("🔄 刷新数据"):
            st.cache_data.clear()
            st.rerun()

    # 两张卡片
    col1, col2 = st.columns(2, gap="large")

    with col1:
        render_card("IC / IF", "中证500 / 沪深300", ic_if, "IC/IF")
        if st.button("📈 查看详细分析", key="btn_icif", use_container_width=True):
            st.session_state.page = 'ic_if'
            st.rerun()

    with col2:
        render_card("IM / IC", "中证1000 / 中证500", im_ic, "IM/IC")
        if st.button("📈 查看详细分析", key="btn_imic", use_container_width=True):
            st.session_state.page = 'im_ic'
            st.rerun()

    # 页脚
    st.markdown("---")
    st.markdown(
        '<p style="text-align:center;color:#4b5563;font-size:0.75rem;">'
        '数据来源: AKShare &nbsp;|&nbsp; 仅供学习参考，不构成投资建议</p>',
        unsafe_allow_html=True
    )

# ============ IC/IF 详情页 ============
elif st.session_state.page == 'ic_if':
    if st.button("← 返回首页"):
        st.session_state.page = 'home'
        st.rerun()

    st.markdown('<div class="main-title">📈 IC/IF 详细分析</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">中证500 / 沪深300 · 2020年至今 · 全局标准差</div>',
                unsafe_allow_html=True)

    latest = ic_if.iloc[-1]
    z = latest['zscore']

    # 顶部指标
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("当前比值", f"{latest['ratio']:.4f}")
    m2.metric("Z-Score", f"{z:+.2f}σ")
    m3.metric("均值", f"{ic_if_mean:.4f}")
    m4.metric("标准差", f"{ic_if_std:.4f}")
    zone_name, _, _ = get_zone_info(z)
    m5.metric("区间", zone_name)

    if abs(z) >= 1.5:
        st.markdown(f'<div class="direction-box">{get_direction(z, "IC/IF")}</div>',
                    unsafe_allow_html=True)

    # 主图
    st.plotly_chart(
        make_detail_chart(ic_if, "IC/IF 比值走势 + 标准差通道"),
        use_container_width=True
    )

    # 标准差通道 + 最近数据
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**标准差通道**")
        channel = pd.DataFrame({
            '通道': ['+3σ', '+2σ', '+1σ', '均值', '-1σ', '-2σ', '-3σ'],
            '数值': [latest['upper_3'], latest['upper_2'], latest['upper_1'],
                    latest['mean'], latest['lower_1'], latest['lower_2'], latest['lower_3']]
        })
        channel['数值'] = channel['数值'].apply(lambda x: f"{x:.4f}")
        st.dataframe(channel, hide_index=True, use_container_width=True)

    with c2:
        st.markdown("**最近10个交易日**")
        recent = ic_if.tail(10)[['ratio', 'zscore']].copy()
        recent.columns = ['比值', 'Z-Score']
        recent.index = recent.index.strftime('%Y-%m-%d')
        recent['Z-Score'] = recent['Z-Score'].apply(lambda x: f"{x:+.2f}σ")
        st.dataframe(recent, use_container_width=True)

    # 赚钱效率
    st.markdown("**赚钱效率分析**")
    st.dataframe(calc_efficiency(ic_if), hide_index=True, use_container_width=True)

# ============ IM/IC 详情页 ============
elif st.session_state.page == 'im_ic':
    if st.button("← 返回首页"):
        st.session_state.page = 'home'
        st.rerun()

    st.markdown('<div class="main-title">📈 IM/IC 详细分析</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">中证1000 / 中证500 · 2020年至今 · 全局标准差</div>',
                unsafe_allow_html=True)

    latest = im_ic.iloc[-1]
    z = latest['zscore']

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("当前比值", f"{latest['ratio']:.4f}")
    m2.metric("Z-Score", f"{z:+.2f}σ")
    m3.metric("均值", f"{im_ic_mean:.4f}")
    m4.metric("标准差", f"{im_ic_std:.4f}")
    zone_name, _, _ = get_zone_info(z)
    m5.metric("区间", zone_name)

    if abs(z) >= 1.5:
        st.markdown(f'<div class="direction-box">{get_direction(z, "IM/IC")}</div>',
                    unsafe_allow_html=True)

    st.plotly_chart(
        make_detail_chart(im_ic, "IM/IC 比值走势 + 标准差通道"),
        use_container_width=True
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**标准差通道**")
        channel = pd.DataFrame({
            '通道': ['+3σ', '+2σ', '+1σ', '均值', '-1σ', '-2σ', '-3σ'],
            '数值': [latest['upper_3'], latest['upper_2'], latest['upper_1'],
                    latest['mean'], latest['lower_1'], latest['lower_2'], latest['lower_3']]
        })
        channel['数值'] = channel['数值'].apply(lambda x: f"{x:.4f}")
        st.dataframe(channel, hide_index=True, use_container_width=True)

    with c2:
        st.markdown("**最近10个交易日**")
        recent = im_ic.tail(10)[['ratio', 'zscore']].copy()
        recent.columns = ['比值', 'Z-Score']
        recent.index = recent.index.strftime('%Y-%m-%d')
        recent['Z-Score'] = recent['Z-Score'].apply(lambda x: f"{x:+.2f}σ")
        st.dataframe(recent, use_container_width=True)

    st.markdown("**赚钱效率分析**")
    st.dataframe(calc_efficiency(im_ic), hide_index=True, use_container_width=True)
