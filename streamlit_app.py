"""Index-ratio dashboard using fixed-window OLS trend signals."""

from __future__ import annotations

import os
from datetime import datetime

import akshare as ak
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics import WINDOW_YEARS, build_pair_models, consensus_snapshot, slice_asof
from data_sources import load_market_history


CORE_WINDOW_YEARS = (3, 5, 7)
DEFAULT_VISIBLE_YEARS = (1, 3, 5, 7)
GLOBAL_WINDOW = "global"
WINDOW_ORDER = (*WINDOW_YEARS, GLOBAL_WINDOW)


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
.stButton > button:disabled { color: #b6b8b3; background: #f2f2ee; border-color: #e5e4df; }
[class*="st-key-pair_card_"] {
    position: relative; cursor: pointer;
    transition: transform .16s ease, box-shadow .16s ease, border-color .16s ease;
}
[class*="st-key-pair_card_"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 34px rgba(71,76,70,.09);
    border-color: #d9ddd6 !important;
}
[class*="st-key-pair_card_"] [class*="st-key-detail_"] {
    position: absolute; inset: 0 !important; z-index: 20; margin: 0;
    width: 100% !important; max-width: none !important; height: 100% !important;
}
[class*="st-key-pair_card_"] [class*="st-key-detail_"] [data-testid="stButton"] {
    width: 100% !important; max-width: none !important; height: 100% !important;
}
[class*="st-key-pair_card_"] [class*="st-key-detail_"] button {
    width: 100% !important; max-width: none !important;
    height: 100% !important; min-height: 100%; padding: 0;
    opacity: 0; border: 0; cursor: pointer;
}
[class*="st-key-window_toggle_"] [data-testid="stBaseButton-primary"] {
    color: #ffffff !important; background: #d78963 !important; border-color: #d78963 !important;
    box-shadow: 0 3px 10px rgba(215,137,99,.18);
}
[class*="st-key-window_toggle_"] [data-testid="stBaseButton-secondary"] {
    color: #858984 !important; background: #f1f1ed !important; border-color: #e2e1dc !important;
}
[class*="st-key-window_toggle_"] button:disabled {
    color: #c1c3be !important; background: #f5f5f2 !important; border-color: #ecebe7 !important;
}

.detail-head { margin: 4px 0 12px; }
.detail-head h1 { margin: 0; color: #303431; font-size: 1.55rem; font-weight: 650; }
.detail-head p { margin: 5px 0 0; color: #969993; font-size: .8rem; }
.metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 14px 0 18px; }
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

.fixed-intro {
    margin: 2px 0 22px; padding: 11px 14px; color: #858984; background: #f6f6f1;
    border-left: 3px solid #d8b98b; border-radius: 4px 10px 10px 4px; font-size: .75rem;
}
.asof-note { color: #9a9d97; font-size: .74rem; margin: 4px 0 14px; text-align: right; }
.asof-note strong { color: #5f8e6f; font-weight: 600; }
.fixed-section { margin: 8px 0 12px; }
.fixed-title { color: #303431; font-size: 1.32rem; font-weight: 650; margin: 0 0 4px; }
.fixed-meta { color: #969993; font-size: .82rem; line-height: 1.65; }
.compare-title { color: #4a4f4b; font-size: .88rem; font-weight: 650; margin: 4px 0 2px; }
.compare-meta { color: #969993; font-size: .7rem; line-height: 1.55; min-height: 2.2rem; }
.section-divider { height: 1px; background: #e5e5df; margin: 30px 0 26px; }
.method-note { color: #737772; font-size: .78rem; line-height: 1.75; }
.method-note code { color: #4f9667; background: #edf6ef; }
[data-testid="stExpander"] details, h5 { color: #4c514d !important; }
[data-testid="stExpander"] details { background: rgba(255,255,255,.55); border-color: #e7e6e0; }
[data-testid="stDateInput"] input { background: #ffffff; border-color: #deddd7; color: #454945; }

@media (max-width: 900px) {
    .metric-grid { grid-template-columns: repeat(2, 1fr); }
    .topbar { flex-direction: column; }
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
    if datetime.now().weekday() >= 5:
        return None
    try:
        spot = ak.stock_zh_index_spot_sina()
        target = spot[spot["代码"].isin(["sh000300", "sh000905", "sh000852"])]
        mapping = {"sh000300": "hs300", "sh000905": "zz500", "sh000852": "zz1000"}
        return {mapping[row["代码"]]: float(row["最新价"]) for _, row in target.iterrows()}
    except Exception:
        return None


def read_secret(name):
    value = os.getenv(name)
    if value:
        return value
    try:
        return st.secrets.get(name)
    except Exception:
        return None


@st.cache_data(ttl=3600)
def load_data():
    return load_market_history(
        refresh_token=read_secret("IFIND_REFRESH_TOKEN"),
        access_token=read_secret("IFIND_ACCESS_TOKEN"),
    )


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


def make_detail_chart(
    model,
    title,
    center_name="OLS趋势线",
    center_color="#58c878",
    sigma_name="残差σ",
    y_range=None,
):
    fig = go.Figure()
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
        ))
        fig.add_trace(go.Scatter(
            x=model.index, y=model[f"lower_{level}"], mode="lines",
            line=dict(color=line, width=.55), fill="tonexty", fillcolor=fill,
            name=f"±{level}{sigma_name}", showlegend=False, hoverinfo="skip",
        ))
    fig.add_trace(go.Scatter(
        x=model.index, y=model["center"], name=center_name,
        line=dict(color=center_color, width=2.0, dash="dash"),
        hovertemplate=f"{center_name} %{{y:.4f}}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=model.index, y=model["ratio"], name="实际比值",
        line=dict(color="#d78963", width=1.8),
        hovertemplate="比值 %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#3d423e", size=14), x=.03, y=.98),
        height=390, template="plotly_white",
        paper_bgcolor="#fbfbf8", plot_bgcolor="#fbfbf8",
        legend=dict(
            orientation="h", y=1.055, x=.03,
            font=dict(color="#888c87", size=11), bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(t=62, b=34, l=62, r=28), hovermode="x unified",
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="#e6e4de", font=dict(color="#444844")),
    )
    fig.update_xaxes(gridcolor="#ecece7", gridwidth=.7, zeroline=False, tickfont=dict(color="#929691"))
    fig.update_yaxes(gridcolor="#e7e8e3", gridwidth=.7, zeroline=False, tickfont=dict(color="#929691"))
    fig.update_yaxes(title_text="比值", title_font=dict(color="#737873"))
    if y_range is not None:
        fig.update_yaxes(range=y_range)
    return fig


def comparison_y_range(*models):
    """Return one padded y-axis range shared by side-by-side charts."""

    lower = min(float(model[["ratio", "lower_3"]].min().min()) for model in models)
    upper = max(float(model[["ratio", "upper_3"]].max().max()) for model in models)
    padding = max((upper - lower) * 0.04, 0.005)
    return [lower - padding, upper + padding]


def render_window_grid(models):
    cells = []
    for years in CORE_WINDOW_YEARS:
        latest = models[years].iloc[-1]
        zscore = float(latest["zscore"])
        _, level_class = signal_class(zscore)
        sign_class = "positive" if zscore > 0 else "negative" if zscore < 0 else ""
        cells.append(
            f'<div class="window-cell {level_class} {sign_class}">'
            f'<div class="window-label">{years}年固定</div>'
            f'<div class="window-z">{zscore:+.2f}σ</div>'
            f'<div class="window-meta">斜率 {latest["slope_year"]:+.3f}/年</div>'
            f'</div>'
        )
    st.markdown(f'<div class="window-grid">{"".join(cells)}</div>', unsafe_allow_html=True)


def render_pair_card(pair_key, pair):
    config = PAIR_CONFIG[pair_key]
    base = pair["base"]
    models = pair["models"]
    snapshot = consensus_snapshot({years: models[years] for years in CORE_WINDOW_YEARS})
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
    st.markdown(f'<div class="decision-box {tone}">{decision}</div>', unsafe_allow_html=True)


def reset_asof_date(state_key, latest_date):
    st.session_state[state_key] = latest_date


def toggle_visible_window(state_key, years):
    selected = set(st.session_state.get(state_key, DEFAULT_VISIBLE_YEARS))
    if years in selected:
        if len(selected) > 1:
            selected.remove(years)
    else:
        selected.add(years)
    st.session_state[state_key] = [option for option in WINDOW_ORDER if option in selected]


def render_detail(pair_key, df):
    config = PAIR_CONFIG[pair_key]
    if st.button("← 返回首页", key=f"back_{pair_key}"):
        st.session_state.page = "home"
        st.rerun()

    st.markdown(
        f'<div class="detail-head"><h1>{config["name"]} 固定窗口分析</h1>'
        f'<p>{config["subtitle"]} · 左侧OLS趋势法 · 右侧平均值与普通标准差法</p></div>',
        unsafe_allow_html=True,
    )

    pair_history = df[[config["numerator"], config["denominator"]]].dropna()
    latest_date = pair_history.index[-1].date()
    earliest_date = pair_history.index[7 * 252 - 1].date()
    date_key = f"asof_date_{pair_key}"
    stored_date = st.session_state.get(date_key)
    if stored_date is None or stored_date < earliest_date or stored_date > latest_date:
        st.session_state[date_key] = latest_date
    requested_date = st.session_state[date_key]
    asof_df, actual_date = slice_asof(pair_history, requested_date)
    pair = build_pair_models(asof_df, config["numerator"], config["denominator"])
    base = pair["base"]
    models = pair["models"]
    mean_models = pair["mean_models"]
    global_model = pair["global_model"]
    global_mean_model = pair["global_mean_model"]
    snapshot = consensus_snapshot({years: models[years] for years in CORE_WINDOW_YEARS})
    visible_key = f"visible_windows_{pair_key}"
    available_options = [*models.keys(), GLOBAL_WINDOW]
    if visible_key not in st.session_state:
        st.session_state[visible_key] = [
            years for years in DEFAULT_VISIBLE_YEARS if years in models
        ]
    selected_windows = [
        option for option in WINDOW_ORDER
        if option in st.session_state[visible_key] and option in available_options
    ]
    if not selected_windows:
        selected_windows = [
            years for years in DEFAULT_VISIBLE_YEARS if years in models
        ]
    if selected_windows != st.session_state[visible_key]:
        st.session_state[visible_key] = selected_windows

    date_col, reset_col, *window_cols = st.columns(
        [2.6, 1.05, *([.82] * len(available_options))], vertical_alignment="bottom"
    )
    with reset_col:
        st.button(
            "回到最新",
            key=f"latest_{pair_key}",
            on_click=reset_asof_date,
            args=(date_key, latest_date),
        )
    with date_col:
        st.date_input(
            "观察日期（截至日）",
            min_value=earliest_date,
            max_value=latest_date,
            key=date_key,
            format="YYYY-MM-DD",
        )
    for column, option in zip(window_cols, available_options):
        label = "全局" if option == GLOBAL_WINDOW else f"{option}年"
        with column:
            st.button(
                label,
                key=f"window_toggle_{pair_key}_{option}",
                type="primary" if option in selected_windows else "secondary",
                on_click=toggle_visible_window,
                args=(visible_key, option),
            )
    adjusted = pd.Timestamp(requested_date).date() != actual_date.date()
    status_text = (
        f"{requested_date:%Y-%m-%d}不是交易日，已自动使用此前最近的交易日 "
        f"<strong>{actual_date:%Y-%m-%d}</strong>"
        if adjusted
        else f"所有计算仅使用 <strong>{actual_date:%Y-%m-%d}</strong> 及以前的数据"
    )
    global_years = len(base) / 252
    status_text += f"；全局窗口使用全部 {len(base):,} 个交易日（约{global_years:.1f}年）"
    st.markdown(f'<div class="asof-note">{status_text}</div>', unsafe_allow_html=True)

    latest_ratio = float(base.iloc[-1]["ratio"])
    metric_values = [("当前比值", f"{latest_ratio:.4f}")]
    for option in selected_windows:
        label = "全局" if option == GLOBAL_WINDOW else f"{option}年"
        model = global_model if option == GLOBAL_WINDOW else models[option]
        mean_model = global_mean_model if option == GLOBAL_WINDOW else mean_models[option]
        metric_values.extend((
            (f"{label} OLS Z", f'{model.iloc[-1]["zscore"]:+.2f}σ'),
            (f"{label} 均值Z", f'{mean_model.iloc[-1]["zscore"]:+.2f}σ'),
        ))
    metric_values.append(
        ("OLS多周期状态", f"{snapshot.label} · {snapshot.confirmation_count}/3")
    )
    cards = "".join(
        f'<div class="metric-item"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>'
        for label, value in metric_values
    )
    st.markdown(f'<div class="metric-grid">{cards}</div>', unsafe_allow_html=True)
    selected_labels = [
        "全局" if option == GLOBAL_WINDOW else f"{option}年" for option in selected_windows
    ]
    st.markdown(
        f'<div class="fixed-intro">本页已切片到 {actual_date:%Y-%m-%d}，之后的数据完全不参与计算。'
        f'当前展示 {"、".join(selected_labels)} 窗口。'
        '每个时间窗口左右使用完全相同的样本和纵轴尺度。左侧用OLS直线作为中枢，'
        '波动取OLS残差标准差；右侧用固定算术平均值作为中枢，波动取比值本身的普通样本标准差。</div>',
        unsafe_allow_html=True,
    )

    for position, option in enumerate(selected_windows):
        is_global = option == GLOBAL_WINDOW
        ols_model = global_model if is_global else models[option]
        mean_model = global_mean_model if is_global else mean_models[option]
        ols_latest = ols_model.iloc[-1]
        mean_latest = mean_model.iloc[-1]
        section_title = "全局窗口" if is_global else f"{option}年固定窗口"
        sample_kind = "全部可用数据" if is_global else f"{option * 252:,}日固定样本"
        st.markdown(
            f'<div class="fixed-section"><div class="fixed-title">{section_title}</div>'
            f'<div class="fixed-meta">共同样本 {ols_model.index[0]:%Y-%m-%d} 至 {ols_model.index[-1]:%Y-%m-%d}'
            f'｜{len(ols_model):,}个交易日｜{sample_kind}｜左右图使用相同纵轴</div></div>',
            unsafe_allow_html=True,
        )
        y_range = comparison_y_range(ols_model, mean_model)
        left, right = st.columns(2, gap="medium")
        with left:
            st.markdown(
                f'<div class="compare-title">OLS趋势＋残差标准差</div>'
                f'<div class="compare-meta">趋势中枢 {ols_latest["center"]:.5f}'
                f'｜年化斜率 {ols_latest["slope_year"]:+.5f}'
                f'｜残差σ {ols_latest["residual_sigma"]:.5f}'
                f'｜当前Z {ols_latest["zscore"]:+.2f}</div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                make_detail_chart(
                    ols_model,
                    "OLS趋势线与固定残差标准差通道",
                    center_name="OLS趋势线",
                    center_color="#58c878",
                    sigma_name="残差σ",
                    y_range=y_range,
                ),
                width="stretch",
                config={"displayModeBar": True, "displaylogo": False},
            )
        with right:
            st.markdown(
                f'<div class="compare-title">平均值＋普通标准差</div>'
                f'<div class="compare-meta">固定均值 {mean_latest["center"]:.5f}'
                f'｜样本σ {mean_latest["residual_sigma"]:.5f}'
                f'｜当前Z {mean_latest["zscore"]:+.2f}</div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                make_detail_chart(
                    mean_model,
                    "固定平均值与普通标准差通道",
                    center_name="固定平均值",
                    center_color="#7d83c7",
                    sigma_name="样本σ",
                    y_range=y_range,
                ),
                width="stretch",
                config={"displayModeBar": True, "displaylogo": False},
            )
        if position < len(selected_windows) - 1:
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    with st.expander("计算口径与使用说明"):
        st.markdown(
            """
            <div class="method-note">
            <p><strong>观察日期切片：</strong>先删除观察日期之后的全部行情，再从剩余数据末端截取所选固定窗口。
            因此历史观察日的中枢、标准差和Z值不会受到未来行情影响；节假日自动回退至此前最近交易日。</p>
            <p><strong>固定窗口：</strong>可选择1、3、5、7、10、15和20年；每个窗口截取最近
            <code>年数 × 252</code>个交易日，独立计算中枢和波动率。高亮按钮表示展示，灰色表示隐藏；
            历史数据不足的年份按钮不显示。</p>
            <p><strong>全局窗口：</strong>不强行凑15年或20年，而是使用观察日期以前能够取得的全部历史数据；
            左右两种方法仍使用完全相同的样本。</p>
            <p><strong>趋势中枢：</strong>当前比值在该固定趋势线上的对应位置。</p>
            <p><strong>残差标准差：</strong><code>sqrt(SSE / (N - 2))</code>，减去2是因为趋势线估计了截距和斜率。</p>
            <p><strong>平均值法：</strong>右侧中枢是固定窗口内全部比值的算术平均值；普通样本标准差为
            <code>sqrt(sum((x - mean)^2) / (N - 1))</code>。均值和标准差在该固定窗口内都保持不变。</p>
            <p><strong>左右对比：</strong>两张图使用同一批交易日和同一纵轴范围。OLS允许中枢随长期趋势倾斜，
            平均值法假设中枢是一条水平线，因此可以直观看出“趋势调整”对当前Z值的影响。</p>
            <p><strong>确认规则：</strong>至少两个窗口在同一方向超过±2σ才算多周期确认；±2.5σ以上标为强偏离。</p>
            <p><strong>重要限制：</strong>图中旧日期的Z使用了整个固定区间拟合出的趋势线，因此只能帮助理解当前样本结构，
            不能当作旧日期当时已经知道的交易信号。若以后做严格历史回测，仍需另用逐日滚动、且不偷看未来数据的版本。</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def load_all_models():
    df, data_metadata = load_data()
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
    return df, pairs, is_realtime, data_metadata


if "page" not in st.session_state:
    st.session_state.page = "home"
if "_pairs" not in st.session_state:
    try:
        with st.spinner("加载行情并计算3 / 5 / 7年固定窗口..."):
            df, pairs, is_realtime, data_metadata = load_all_models()
            st.session_state._df = df
            st.session_state._pairs = pairs
            st.session_state._is_realtime = is_realtime
            st.session_state._data_metadata = data_metadata
    except Exception as exc:
        st.error(f"数据加载失败：{exc}")
        st.stop()

df = st.session_state._df
pairs = st.session_state._pairs
is_realtime = st.session_state._is_realtime
data_metadata = st.session_state._data_metadata


if st.session_state.page == "home":
    st.markdown(
        """
        <div class="topbar">
            <div>
                <h1>指数比值趋势偏离看板</h1>
                <p>用固定OLS窗口对照当前比值位置，详情页可切换1年至20年及全局周期</p>
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
            for key in ("_df", "_pairs", "_is_realtime", "_data_metadata"):
                st.session_state.pop(key, None)
            st.rerun()

    columns = st.columns(3, gap="medium")
    for column, (pair_key, pair) in zip(columns, pairs.items()):
        with column:
            with st.container(border=True, key=f"pair_card_{pair_key}"):
                render_pair_card(pair_key, pair)
                if st.button(f"打开 {PAIR_CONFIG[pair_key]['name']} 详情", key=f"detail_{pair_key}"):
                    st.session_state.page = pair_key
                    st.rerun()
else:
    render_detail(st.session_state.page, df)
