"""Index-ratio dashboard using rolling OLS trend and residual signals."""

from __future__ import annotations

from datetime import datetime

import akshare as ak
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from analytics import WINDOW_YEARS, build_pair_models, consensus_snapshot


st.set_page_config(
    page_title="指数比值趋势偏离看板",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


PAIR_CONFIG = {
    "ic_if": {
        "name": "IC / IF",
        "subtitle": "中证500 / 沪深300",
        "numerator": "zz500",
        "denominator": "hs300",
        "positive": "中证500相对偏强：做空IC + 做多IF",
        "negative": "中证500相对偏弱：做多IC + 做空IF",
    },
    "im_ic": {
        "name": "IM / IC",
        "subtitle": "中证1000 / 中证500",
        "numerator": "zz1000",
        "denominator": "zz500",
        "positive": "中证1000相对偏强：做空IM + 做多IC",
        "negative": "中证1000相对偏弱：做多IM + 做空IC",
    },
    "im_if": {
        "name": "IM / IF",
        "subtitle": "中证1000 / 沪深300",
        "numerator": "zz1000",
        "denominator": "hs300",
        "positive": "中证1000相对偏强：做空IM + 做多IF",
        "negative": "中证1000相对偏弱：做多IM + 做空IF",
    },
}


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'PingFang SC', sans-serif; }
.stApp { background: #fbfbf8; color: #303431; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.65rem 2.5rem 1.2rem; max-width: 1600px; }

.topbar {
    display: flex; justify-content: space-between; align-items: flex-start;
    gap: 24px; padding: 12px 0 22px; margin-bottom: 20px;
    border-bottom: 1px solid #e8e7e1;
}
.topbar h1 { margin: 0; color: #2f3431; font-size: 1.65rem; font-weight: 650; letter-spacing: -.02em; }
.topbar p { margin: 7px 0 0; color: #8b8e89; font-size: .82rem; }
.topbar-meta { color: #929590; font-size: .76rem; text-align: right; line-height: 1.8; }
.model-badge {
    display: inline-flex; align-items: center; padding: 5px 10px; border-radius: 999px;
    color: #6aa27b; background: #f0f7f1; border: 1px solid #dcecdf;
    font-size: .7rem; font-weight: 600; letter-spacing: .02em;
}
.live-dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; margin-right: 5px; }
.live-dot.on { background: #5bc77a; box-shadow: 0 0 7px rgba(91,199,122,.45); }
.live-dot.off { background: #b2b4af; }

[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255,255,255,.78);
    border-color: #e8e7e1 !important;
    border-radius: 16px !important;
    box-shadow: 0 8px 28px rgba(71,76,70,.045);
}
.pair-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; }
.pair-name { color: #303431; font-size: 1.15rem; font-weight: 650; }
.pair-sub { color: #9a9c97; font-size: .72rem; margin-top: 2px; }
.status-pill {
    padding: 5px 10px; border-radius: 999px; font-size: .69rem; font-weight: 600;
    color: #858984; background: #f5f5f1; border: 1px solid #e8e7e1;
}
.status-pill.negative { color: #3f9d66; background: #eff8f1; border-color: #d9ecde; }
.status-pill.positive { color: #c87255; background: #fcf2ed; border-color: #f0ddd4; }
.ratio-line { margin-top: 11px; display: flex; align-items: baseline; gap: 10px; }
.ratio-value { color: #343835; font-size: 2.15rem; font-weight: 650; letter-spacing: -.03em; }
.ratio-change { font-size: .82rem; font-weight: 600; }
.ratio-change.up { color: #d68462; }
.ratio-change.down { color: #55b978; }
.ratio-change.flat { color: #999c97; }

.window-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin: 4px 0 2px; }
.window-cell {
    padding: 10px 9px; border-radius: 11px; background: #fafaf7;
    border: 1px solid #ecebe5; text-align: center;
}
.window-label { color: #999c97; font-size: .66rem; }
.window-z { color: #454945; font-size: 1.05rem; font-weight: 650; margin: 3px 0; }
.window-meta { color: #aaaca6; font-size: .62rem; }
.window-cell.confirmed { border-color: #ebd8cd; background: #fdf7f2; }
.window-cell.strong { border-color: #e8cabc; background: #fcf1ea; }
.window-cell.negative .window-z { color: #4bad70; }
.window-cell.positive .window-z { color: #d37d5e; }

.decision-box {
    margin-top: 10px; padding: 11px 13px; border-radius: 11px;
    color: #858984; background: #fafaf7; border: 1px solid #ecebe5;
    font-size: .74rem; line-height: 1.55;
}
.decision-box.negative { color: #468f62; background: #f1f8f2; border-color: #dcecdf; }
.decision-box.positive { color: #bd7358; background: #fcf3ee; border-color: #efddd5; }

.stButton > button {
    width: 100%; color: #5f8e6f; background: #f5faf6;
    border: 1px solid #dfeae1; border-radius: 10px; font-weight: 600;
}
.stButton > button:hover { border-color: #bcd6c2; color: #3f7652; }

.detail-head { margin: 4px 0 12px; }
.detail-head h1 { margin: 0; color: #303431; font-size: 1.55rem; font-weight: 650; }
.detail-head p { margin: 5px 0 0; color: #969993; font-size: .8rem; }
.metric-grid { display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px; margin: 14px 0 18px; }
.metric-item {
    padding: 13px 12px; border-radius: 12px; text-align: center;
    background: rgba(255,255,255,.78); border: 1px solid #e8e7e1;
    box-shadow: 0 5px 18px rgba(71,76,70,.035);
}
.metric-label { color: #9a9d97; font-size: .66rem; }
.metric-value { color: #414541; font-size: 1.12rem; font-weight: 650; margin-top: 5px; }

.dark-table {
    width: 100%; border-collapse: separate; border-spacing: 0; overflow: hidden;
    border-radius: 12px; background: rgba(255,255,255,.72); border: 1px solid #e8e7e1;
    font-size: .76rem;
}
.dark-table th { padding: 10px 12px; color: #8e918c; background: #f6f6f2; text-align: left; }
.dark-table td { padding: 9px 12px; color: #555a55; border-top: 1px solid #eeede8; }
.dark-table .num { text-align: right; font-variant-numeric: tabular-nums; }
.dark-table .negative { color: #4bad70; font-weight: 600; }
.dark-table .positive { color: #d37d5e; font-weight: 600; }
.dark-table .selected td { background: #f0f7f2; }

.warmup-note {
    margin: -5px 0 14px; padding: 10px 13px; color: #858984; background: #f6f6f1;
    border-left: 3px solid #d8b98b; border-radius: 4px 10px 10px 4px; font-size: .75rem;
}
.method-note { color: #737772; font-size: .78rem; line-height: 1.75; }
.method-note code { color: #4f9667; background: #edf6ef; }
.footer { color: #b0b2ad; text-align: center; font-size: .68rem; padding: 24px 0 10px; }

[data-testid="stRadio"] label, [data-testid="stExpander"] details, h5 { color: #4c514d !important; }
[data-testid="stRadio"] [role="radiogroup"] { gap: 1.2rem; }
[data-testid="stExpander"] details { background: rgba(255,255,255,.55); border-color: #e7e6e0; }

@media (max-width: 900px) {
    .metric-grid { grid-template-columns: repeat(3, 1fr); }
    .topbar { flex-direction: column; }
    .topbar-meta { text-align: left; }
}
@media (max-width: 560px) {
    .block-container { padding: 1rem; }
    .metric-grid { grid-template-columns: repeat(2, 1fr); }
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(ttl=60)
def get_realtime_quotes():
    try:
        spot = ak.stock_zh_index_spot_sina()
        target = spot[spot["代码"].isin(["sh000300", "sh000905", "sh000852"])]
        mapping = {"sh000300": "hs300", "sh000905": "zz500", "sh000852": "zz1000"}
        return {mapping[row["代码"]]: float(row["最新价"]) for _, row in target.iterrows()}
    except Exception:
        return None


@st.cache_data(ttl=86400)
def load_data():
    series = []
    for symbol, column in (
        ("sh000300", "hs300"),
        ("sh000905", "zz500"),
        ("sh000852", "zz1000"),
    ):
        item = ak.stock_zh_index_daily(symbol=symbol)
        item["date"] = pd.to_datetime(item["date"])
        item = item.set_index("date")[["close"]].rename(columns={"close": column})
        series.append(item)
    df = series[0].join(series[1], how="inner").join(series[2], how="inner")
    return df[~df.index.duplicated(keep="last")].sort_index().dropna()


def signal_class(zscore):
    absolute = abs(float(zscore))
    if absolute >= 3.0:
        return "极端", "strong"
    if absolute >= 2.5:
        return "强偏离", "strong"
    if absolute >= 2.0:
        return "已确认", "confirmed"
    if absolute >= 1.0:
        return "观察", ""
    return "正常", ""


def direction_class(direction):
    return "positive" if direction > 0 else "negative" if direction < 0 else ""


def direction_text(pair_key, direction):
    config = PAIR_CONFIG[pair_key]
    if direction > 0:
        return config["positive"]
    if direction < 0:
        return config["negative"]
    return "暂无明确方向，等待窗口形成一致偏离"


def make_sparkline(model):
    recent = model.tail(252)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recent.index, y=recent["upper_2"], mode="lines",
        line=dict(color="rgba(116,190,143,.18)", width=.6), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=recent.index, y=recent["lower_2"], mode="lines",
        line=dict(color="rgba(116,190,143,.18)", width=.6),
        fill="tonexty", fillcolor="rgba(145,203,168,.09)", showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=recent.index, y=recent["center"], mode="lines",
        line=dict(color="#58bf77", width=1.25, dash="dash"), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=recent.index, y=recent["ratio"], mode="lines",
        line=dict(color="#d78963", width=2), showlegend=False,
        hovertemplate="%{x|%Y-%m-%d}<br>比值 %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        height=122, margin=dict(l=0, r=4, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False), yaxis=dict(visible=False), hovermode="x unified",
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="#e6e4de", font=dict(color="#444844")),
    )
    return fig


def make_detail_chart(base, model, title):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.69, 0.31], vertical_spacing=0.075,
    )
    band_styles = {
        3: ("rgba(237,218,164,.13)", "rgba(221,197,129,.10)"),
        2: ("rgba(157,207,179,.13)", "rgba(117,184,147,.10)"),
        1: ("rgba(165,166,214,.18)", "rgba(133,136,194,.10)"),
    }
    for level in (3, 2, 1):
        fill, line = band_styles[level]
        fig.add_trace(go.Scatter(
            x=model.index, y=model[f"upper_{level}"], mode="lines",
            line=dict(color=line, width=.55), showlegend=False, hoverinfo="skip",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=model.index, y=model[f"lower_{level}"], mode="lines",
            line=dict(color=line, width=.55), fill="tonexty", fillcolor=fill,
            name=f"±{level}残差σ", showlegend=False, hoverinfo="skip",
        ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=model.index, y=model["center"], name="滚动OLS中枢",
        line=dict(color="#58c878", width=2.0, dash="dash"),
        hovertemplate="趋势中枢 %{y:.4f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=base.index, y=base["ratio"], name="实际比值",
        line=dict(color="#d78963", width=1.8),
        hovertemplate="比值 %{y:.4f}<extra></extra>",
    ), row=1, col=1)

    colors = ["#b5a3f4" if z >= 0 else "#7ecce2" for z in model["zscore"]]
    fig.add_trace(go.Bar(
        x=model.index, y=model["zscore"], name="残差Z",
        marker_color=colors, opacity=.78,
        hovertemplate="残差Z %{y:+.2f}σ<extra></extra>",
    ), row=2, col=1)
    for value, color in (
        (2, "#63c4df"), (-2, "#63c4df"),
        (3, "#edca61"), (-3, "#edca61"),
        (4, "#63c4df"), (-4, "#63c4df"),
    ):
        fig.add_hline(y=value, line_dash="dash", line_color=color, line_width=1.05, row=2, col=1)
    fig.add_hline(y=0, line_color="#cfd3ce", line_width=.7, row=2, col=1)
    fig.update_layout(
        title=dict(text=title, font=dict(color="#3d423e", size=16), x=.03, y=.98),
        height=660, template="plotly_white",
        paper_bgcolor="#fbfbf8", plot_bgcolor="#fbfbf8",
        legend=dict(
            orientation="h", y=1.055, x=.03,
            font=dict(color="#888c87", size=11), bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(t=62, b=30, l=62, r=28), bargap=.22, hovermode="x unified",
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="#e6e4de", font=dict(color="#444844")),
    )
    fig.update_xaxes(gridcolor="#ecece7", gridwidth=.7, zeroline=False, tickfont=dict(color="#929691"))
    fig.update_yaxes(gridcolor="#e7e8e3", gridwidth=.7, zeroline=False, tickfont=dict(color="#929691"))
    fig.update_yaxes(title_text="比值", title_font=dict(color="#737873"), row=1, col=1)
    fig.update_yaxes(title_text="残差Z", title_font=dict(color="#737873"), row=2, col=1)
    return fig


def render_window_grid(models):
    cells = []
    for years in WINDOW_YEARS:
        latest = models[years].iloc[-1]
        zscore = float(latest["zscore"])
        _, level_class = signal_class(zscore)
        sign_class = "positive" if zscore > 0 else "negative" if zscore < 0 else ""
        cells.append(
            f'<div class="window-cell {level_class} {sign_class}">'
            f'<div class="window-label">{years}年窗口</div>'
            f'<div class="window-z">{zscore:+.2f}σ</div>'
            f'<div class="window-meta">斜率 {latest["slope_year"]:+.3f}/年</div>'
            f'</div>'
        )
    st.markdown(f'<div class="window-grid">{"".join(cells)}</div>', unsafe_allow_html=True)


def render_pair_card(pair_key, pair):
    config = PAIR_CONFIG[pair_key]
    base = pair["base"]
    models = pair["models"]
    snapshot = consensus_snapshot(models)
    latest_ratio = float(base.iloc[-1]["ratio"])
    previous_ratio = float(base.iloc[-2]["ratio"])
    change_pct = (latest_ratio / previous_ratio - 1.0) * 100.0
    change_class = "up" if change_pct > 0 else "down" if change_pct < 0 else "flat"
    change_arrow = "▲" if change_pct > 0 else "▼" if change_pct < 0 else "—"
    tone = direction_class(snapshot.direction)

    st.markdown(
        f"""
        <div class="pair-head">
            <div><div class="pair-name">{config['name']}</div><div class="pair-sub">{config['subtitle']}</div></div>
            <span class="status-pill {tone}">{snapshot.label} · {snapshot.confirmation_count}/3</span>
        </div>
        <div class="ratio-line">
            <span class="ratio-value">{latest_ratio:.4f}</span>
            <span class="ratio-change {change_class}">{change_arrow} {abs(change_pct):.2f}%</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.plotly_chart(make_sparkline(models[5]), width="stretch", config={"displayModeBar": False})
    render_window_grid(models)
    decision = direction_text(pair_key, snapshot.direction)
    if snapshot.confirmation_count < 2:
        decision = f"{decision}；目前确认度不足，等待至少两个窗口超过±2σ"
    st.markdown(f'<div class="decision-box {tone}">{decision}</div>', unsafe_allow_html=True)


def multi_window_table(models, selected_years):
    rows = []
    for years in WINDOW_YEARS:
        latest = models[years].iloc[-1]
        zscore = float(latest["zscore"])
        zone, _ = signal_class(zscore)
        zclass = "positive" if zscore > 0 else "negative" if zscore < 0 else ""
        selected = "selected" if years == selected_years else ""
        rows.append(
            f'<tr class="{selected}"><td>{years}年</td>'
            f'<td class="num">{latest["center"]:.4f}</td>'
            f'<td class="num">{latest["residual_sigma"]:.4f}</td>'
            f'<td class="num">{latest["slope_year"]:+.4f}</td>'
            f'<td class="num {zclass}">{zscore:+.2f}σ</td><td>{zone}</td></tr>'
        )
    return (
        '<table class="dark-table"><thead><tr><th>窗口</th><th class="num">趋势中枢</th>'
        '<th class="num">残差σ</th><th class="num">年化斜率</th><th class="num">Z</th><th>状态</th>'
        f'</tr></thead><tbody>{"".join(rows)}</tbody></table>'
    )


def recent_table(model):
    rows = []
    for date, row in model.tail(10).iterrows():
        zscore = float(row["zscore"])
        zclass = "positive" if zscore > 0 else "negative" if zscore < 0 else ""
        rows.append(
            f'<tr><td>{date.strftime("%Y-%m-%d")}</td><td class="num">{row["ratio"]:.4f}</td>'
            f'<td class="num">{row["center"]:.4f}</td><td class="num {zclass}">{zscore:+.2f}σ</td></tr>'
        )
    return (
        '<table class="dark-table"><thead><tr><th>日期</th><th class="num">比值</th>'
        f'<th class="num">中枢</th><th class="num">残差Z</th></tr></thead><tbody>{"".join(rows)}</tbody></table>'
    )


def render_detail(pair_key, pair):
    config = PAIR_CONFIG[pair_key]
    base = pair["base"]
    models = pair["models"]
    snapshot = consensus_snapshot(models)
    if st.button("← 返回首页", key=f"back_{pair_key}"):
        st.session_state.page = "home"
        st.rerun()

    st.markdown(
        f'<div class="detail-head"><h1>{config["name"]} 趋势偏离分析</h1>'
        f'<p>{config["subtitle"]} · 滚动OLS趋势中枢 · 窗口内回归残差标准差</p></div>',
        unsafe_allow_html=True,
    )
    selected_years = st.radio(
        "观察窗口", WINDOW_YEARS, index=1, horizontal=True,
        format_func=lambda value: f"{value}年", key=f"window_{pair_key}",
    )
    model = models[selected_years]
    latest = model.iloc[-1]
    zone, _ = signal_class(latest["zscore"])
    metric_values = (
        ("当前比值", f'{latest["ratio"]:.4f}'),
        ("趋势中枢", f'{latest["center"]:.4f}'),
        ("残差标准差", f'{latest["residual_sigma"]:.4f}'),
        ("残差Z", f'{latest["zscore"]:+.2f}σ'),
        ("年化斜率", f'{latest["slope_year"]:+.4f}'),
        ("多周期确认", f'{snapshot.confirmation_count}/3 · {zone}'),
    )
    cards = "".join(
        f'<div class="metric-item"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>'
        for label, value in metric_values
    )
    st.markdown(f'<div class="metric-grid">{cards}</div>', unsafe_allow_html=True)
    model_start = model.index[0]
    source_start = base.index[0]
    st.markdown(
        f'<div class="warmup-note">完整比值从 {source_start:%Y-%m-%d} 开始显示；'
        f'{selected_years}年模型需先积累 {selected_years * 252:,} 个交易日，'
        f'因此中枢、通道和残差Z从 {model_start:%Y-%m-%d} 才开始出现。</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        make_detail_chart(base, model, f"{selected_years}年滚动OLS：比值、趋势中枢与残差通道"),
        width="stretch",
        config={"displayModeBar": True, "displaylogo": False},
    )

    left, right = st.columns([1.05, .95], gap="large")
    with left:
        st.markdown("##### 多周期当前状态")
        st.markdown(multi_window_table(models, selected_years), unsafe_allow_html=True)
    with right:
        st.markdown("##### 最近10个交易日")
        st.markdown(recent_table(model), unsafe_allow_html=True)

    with st.expander("计算口径与使用说明"):
        st.markdown(
            f"""
            <div class="method-note">
            <p><strong>{selected_years}年滚动窗口：</strong>每天只使用当天以前的
            <code>{selected_years * 252}</code>个交易日拟合比值趋势线，当前日期不会参与自己的中枢计算。</p>
            <p><strong>为什么前几年没有中枢：</strong>这段是模型的“准备期”。原始比值仍完整显示，
            但在凑够{selected_years}年样本以前，不生成中枢和Z值，避免拿几个月数据冒充{selected_years}年规律。</p>
            <p><strong>趋势中枢：</strong>最小二乘趋势线向前预测到当天得到的合理比值。</p>
            <p><strong>残差标准差：</strong><code>sqrt(SSE / (N - 2))</code>，减去2是因为趋势线估计了截距和斜率。</p>
            <p><strong>确认规则：</strong>至少两个窗口在同一方向超过±2σ才算多周期确认；±2.5σ以上标为强偏离。
            旧版全局均值、固定标准差和基于旧Z值的收益阈值已移除，避免不同口径混用。</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def load_all_models():
    df = load_data()
    realtime = get_realtime_quotes()
    is_realtime = False
    if realtime and len(realtime) == 3:
        today = pd.Timestamp(datetime.now().date())
        row = {"hs300": realtime["hs300"], "zz500": realtime["zz500"], "zz1000": realtime["zz1000"]}
        if today not in df.index:
            df = pd.concat([df, pd.DataFrame(row, index=[today])])
        else:
            df.loc[today] = row
        df = df.sort_index()
        is_realtime = True
    pairs = {
        key: build_pair_models(df, config["numerator"], config["denominator"])
        for key, config in PAIR_CONFIG.items()
    }
    return df, pairs, is_realtime


if "page" not in st.session_state:
    st.session_state.page = "home"
if "_pairs" not in st.session_state:
    try:
        with st.spinner("加载完整历史并计算滚动OLS..."):
            df, pairs, is_realtime = load_all_models()
            st.session_state._df = df
            st.session_state._pairs = pairs
            st.session_state._is_realtime = is_realtime
    except Exception as exc:
        st.error(f"数据加载失败：{exc}")
        st.stop()

df = st.session_state._df
pairs = st.session_state._pairs
is_realtime = st.session_state._is_realtime


if st.session_state.page == "home":
    live = '<span class="live-dot on"></span>盘中实时' if is_realtime else '<span class="live-dot off"></span>收盘数据'
    st.markdown(
        f"""
        <div class="topbar">
            <div>
                <h1>指数比值趋势偏离看板</h1>
                <p>用滚动OLS识别动态中枢，用回归残差衡量偏离，不再假设长期均值固定不变</p>
            </div>
            <div class="topbar-meta">
                <span class="model-badge">3 / 5 / 7年多周期模型</span><br>
                {live} · 数据截至 {df.index[-1].strftime('%Y-%m-%d')} · {len(df):,}个共同交易日
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    refresh_col, spacer = st.columns([1, 8])
    with refresh_col:
        if st.button("刷新数据", key="refresh"):
            get_realtime_quotes.clear()
            load_data.clear()
            for key in ("_df", "_pairs", "_is_realtime"):
                st.session_state.pop(key, None)
            st.rerun()

    columns = st.columns(3, gap="medium")
    for column, (pair_key, pair) in zip(columns, pairs.items()):
        with column:
            with st.container(border=True):
                render_pair_card(pair_key, pair)
                if st.button("查看滚动历史 →", key=f"detail_{pair_key}"):
                    st.session_state.page = pair_key
                    st.rerun()
    st.markdown(
        '<div class="footer">数据来源：AKShare · 统计模型仅用于监控与研究，不构成交易建议</div>',
        unsafe_allow_html=True,
    )
else:
    render_detail(st.session_state.page, pairs[st.session_state.page])
