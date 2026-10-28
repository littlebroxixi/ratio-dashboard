"""Render BTC/gold research within the dashboard's existing navigation."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from btc_gold import YEARS, snapshot, load_daily, load_research


def render_btc_gold_detail():
    st.markdown("""<style>.stApp{background:#fbfbf8;color:#303431}
    .block-container{max-width:1600px;padding-top:1.7rem}h1,h2,h3{color:#303431}
    [data-testid="stMetric"]{background:white;border:1px solid #e8e7e1;border-radius:12px;padding:16px}
    </style>""", unsafe_allow_html=True)
    if st.button("← 返回比值看板", key="btc_gold_back"):
        st.session_state.page = "home"
        st.rerun()
    st.title("BTC / 黄金日线研究")
    st.caption("BTCUSDm 日收盘 ÷ XAUUSDm 日收盘 · 美元等金额配平 · 原始服务器日期")

    data = load_daily()
    try:
        results = load_research()
    except (OSError, ValueError, KeyError) as exc:
        results = None
        st.warning(f"历史研究统计暂不可用：{exc}")
    research_end = results["metadata"]["end"] if results else "未知"
    st.caption(f"数据覆盖 {data.index[0]:%Y-%m-%d} 至 {data.index[-1]:%Y-%m-%d} · {len(data):,} 条共同日线 · MT5 导出快照，非实时更新")
    tab_position, tab_fixed, tab_conv, tab_data = st.tabs(["比值定位", "固定持有 1-6 个月", "收敛统计", "日线与口径"])


    def chart(model, title, yrange):
        fig = go.Figure()
        for level, color in ((3, "#e0cbbf"), (2, "#d8a580"), (1, "#d4bf88")):
            for side, sign in (("upper", "+"), ("lower", "-")):
                fig.add_trace(go.Scatter(x=model.index, y=model[f"{side}_{level}"],
                                        name=f"{sign}{level}σ", line=dict(color=color, width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=model.index, y=model.center, name="中枢", line=dict(color="#5d9873", width=2)))
        fig.add_trace(go.Scatter(x=model.index, y=model.ratio, name="BTC / 黄金", line=dict(color="#303c46", width=1.5)))
        fig.update_layout(title=title, height=360, margin=dict(l=30, r=20, t=45, b=35),
                          paper_bgcolor="#fbfbf8", plot_bgcolor="#fbfbf8", hovermode="x unified",
                          legend=dict(orientation="h", y=-.18), yaxis=dict(range=yrange, gridcolor="#e7e7df"),
                          xaxis=dict(gridcolor="#e7e7df"))
        return fig


    with tab_position:
        requested = st.date_input("观察日期（截至日）", value=data.index[-1].date(),
                                  min_value=data.index[0].date(), max_value=data.index[-1].date())
        cut = data.loc[:pd.Timestamp(requested)]
        actual = cut.index[-1]
        st.caption(f"实际观察日：{actual:%Y-%m-%d}；此后行情完全不参与定位计算。")
        metrics = st.columns(3)
        metrics[0].metric("BTC / 黄金", f"{cut.ratio.iloc[-1]:.5f}")
        metrics[1].metric("BTC / 美元", f"{cut.btc_close.iloc[-1]:,.2f}")
        metrics[2].metric("黄金 / 美元", f"{cut.gold_close.iloc[-1]:,.3f}")
        available = [y for y in YEARS if actual - pd.DateOffset(years=y) >= cut.index[0]]
        labels = [f"{y}年" for y in available] + (["全样本"] if len(cut) >= 3 else [])
        selected = st.multiselect("展示窗口", labels, default=[f"{y}年" for y in available])
        tables, all_models = [], {}
        for label in labels:
            rows, models = snapshot(cut, None if label == "全样本" else int(label[:-1]))
            tables.extend(rows)
            all_models[label] = models
        if tables:
            table = pd.DataFrame(tables)
            core = table[(table.method == "OLS") & table.years.isin([3, 5, 7])]
            high = int((core.z >= 2).sum()); low = int((core.z <= -2).sum())
            if max(high, low) >= 2:
                st.info(f"OLS 3/5/7 年核心窗口：{max(high, low)} 个窗口同向超过 2σ。仅为研究信号。")
            else:
                st.info("OLS 3/5/7 年核心窗口：尚未形成至少两个窗口同向超过 2σ 的确认。")
            st.dataframe(table[["window", "method", "n", "center", "sigma", "z"]].rename(columns={
                "window":"窗口", "method":"方法", "n":"样本数", "center":"当前中枢", "sigma":"标准差", "z":"当前Z"}),
                hide_index=True, width="stretch")
        if not tables:
            st.info("当前观察日不足 3 条日线，暂不计算统计通道。")
        for label in selected:
            models = all_models[label]
            ols, mean = models["OLS"], models["均值标准差"]
            bounds = np.concatenate([m[["ratio", "lower_3", "upper_3"]].to_numpy().ravel() for m in (ols, mean)])
            span = max(bounds) - min(bounds)
            yrange = [float(min(bounds) - .04 * span), float(max(bounds) + .04 * span)]
            st.subheader(f"{label}固定窗口")
            st.caption(f"{ols.index[0]:%Y-%m-%d} 至 {ols.index[-1]:%Y-%m-%d} · {len(ols):,} 条共同日线 · 左右同一纵轴")
            left, right = st.columns(2)
            left.plotly_chart(chart(ols, "OLS 趋势＋残差标准差", yrange), width="stretch", key=f"ols_{label}")
            right.plotly_chart(chart(mean, "均值＋普通标准差", yrange), width="stretch", key=f"mean_{label}")
        st.caption("固定图上的旧日期使用整个观察窗口拟合，仅用于描述当时窗口；历史事件信号另按每日滚动计算。")

    with tab_fixed:
        st.subheader("固定持有期事件研究")
        st.caption(f"本页为截至 {research_end} 的完整研究结果，不随定位页的历史观察日期改变。收益分母为两腿初始名义金额合计。")
        if results:
            method = st.selectbox("方法", ["OLS", "均值标准差"], key="fixed_method")
            year = st.selectbox("窗口年数", list(YEARS), index=1, key="fixed_year")
            rows = pd.DataFrame(results["fixed_summary"])
            rows = rows[(rows.method == method) & (rows.years == year)]
            st.dataframe(rows[["side","band","months","n","mean_return","median_return","win_rate","worst_return","worst_mae","mean_same_close"]], hide_index=True, width="stretch")
            st.caption("return：收盘信号后下一共同日开盘入场、预定到期日收盘退出的毛收益参考。same_close：原方法的信号日收盘事件收益。无样本的指标留空。")

    with tab_conv:
        st.subheader("12 个月内的 Z 值收敛")
        st.caption(f"截至 {research_end}；只统计后续观察期完整的事件。Z 使用逐日重估的中枢和标准差。")
        if results:
            method = st.selectbox("方法", ["OLS", "均值标准差"], key="conv_method")
            year = st.selectbox("窗口年数", list(YEARS), index=1, key="conv_year")
            kind = st.radio("目标口径", ["固定Z目标", "收敛幅度"], horizontal=True)
            rows = pd.DataFrame(results["convergence_summary"])
            rows = rows[(rows.method == method) & (rows.years == year) & (rows.target_kind == kind)]
            st.dataframe(rows[["side","band","target_parameter","events","already_reached","effective_n","hits","hit_rate","median_hit_days","mean_return_if_hit","win_rate_if_hit"]], hide_index=True, width="stretch")
            st.caption("side=-1 为负向偏离，side=+1 为正向偏离。固定Z参数为绝对值，目标符号跟随 side。入场已达到的目标单列并排除有效命中率。")

    with tab_data:
        st.dataframe(data.reset_index(), hide_index=True, width="stretch")
        st.download_button("下载日线底表", data.to_csv().encode("utf-8-sig"), "BTC_Gold_Daily.csv", "text/csv")
        st.markdown("""- 日线取两品种共同日期，保留周日，不补值、不插值。
    - 年窗口按日历截取，OLS 自变量沿用共同日线的观测序号。
    - 每腿初始名义金额 100,000 美元，数量固定；收益不是保证金收益率。
    - 毛收益参考未计点差、佣金、滑点、隔夜费和实际合约手数约束。日线标签不能证明两腿报价完全同步。
    - 不同窗口与不同偏离区间可能属于同一轮行情，事件统计不可直接复利成资金曲线。
    - 黄金／白银还缺直接白银／美元日线，未混入本次计算。""")
