"""Supplemental BTC/gold research below the shared pair-detail UI."""
import pandas as pd
import streamlit as st

from btc_gold import YEARS, load_research


def research_table(rows, columns, percentages=()):
    display = rows[list(columns)].rename(columns=columns).copy()
    if "方向" in display:
        display["方向"] = display["方向"].map({-1: "负向偏离", 1: "正向偏离"})
    formats = {columns[column]: "{:.2%}" for column in percentages}
    st.dataframe(display.style.format(formats, na_rep="—"), hide_index=True, width="stretch")


def render_btc_gold_research(data):
    try:
        results = load_research()
    except (OSError, ValueError, KeyError) as exc:
        st.warning(f"历史研究统计暂不可用：{exc}")
        return
    research_end = results["metadata"]["end"]
    with st.expander("固定持有 1—6 个月 · 收益统计"):
        st.markdown(f'<div class="method-note">截至 {research_end} 的完整事件研究，不随上方观察日期改变。收益分母为两腿初始名义金额合计。</div>', unsafe_allow_html=True)
        left, right = st.columns(2, gap="medium")
        method = left.selectbox("方法", ["OLS", "均值标准差"], key="fixed_method")
        year = right.selectbox("窗口年数", list(YEARS), index=1, key="fixed_year")
        rows = pd.DataFrame(results["fixed_summary"])
        rows = rows[(rows.method == method) & (rows.years == year)]
        columns = {"side": "方向", "band": "入场区间", "months": "持有月数", "n": "样本数",
                   "mean_return": "平均收益", "median_return": "中位收益", "win_rate": "胜率",
                   "worst_return": "最差收益", "worst_mae": "最大不利波动", "mean_same_close": "信号日收盘参考"}
        research_table(rows, columns, ("mean_return", "median_return", "win_rate", "worst_return", "worst_mae", "mean_same_close"))
        st.caption("收盘信号后下一共同日开盘入场、预定到期日收盘退出的毛收益参考。信号日收盘收益单列对照；无样本留空。")

    with st.expander("12 个月内 · Z 值收敛统计"):
        st.markdown(f'<div class="method-note">截至 {research_end}；只统计后续观察期完整的事件。Z 使用逐日重估的中枢和标准差。</div>', unsafe_allow_html=True)
        left, right = st.columns(2, gap="medium")
        method = left.selectbox("方法", ["OLS", "均值标准差"], key="conv_method")
        year = right.selectbox("窗口年数", list(YEARS), index=1, key="conv_year")
        kind = st.radio("目标口径", ["固定Z目标", "收敛幅度"], horizontal=True, key="conv_kind")
        rows = pd.DataFrame(results["convergence_summary"])
        rows = rows[(rows.method == method) & (rows.years == year) & (rows.target_kind == kind)]
        columns = {"side": "方向", "band": "入场区间", "target_parameter": "目标参数（σ）", "events": "事件数",
                   "already_reached": "入场已达到", "effective_n": "有效样本", "hits": "命中数", "hit_rate": "命中率",
                   "median_hit_days": "收敛天数中位数", "mean_return_if_hit": "命中平均收益", "win_rate_if_hit": "命中胜率"}
        research_table(rows, columns, ("hit_rate", "mean_return_if_hit", "win_rate_if_hit"))
        st.caption("固定Z参数为绝对值，目标符号跟随偏离方向。入场已达到的目标单列并排除有效命中率。")

    with st.expander("日线数据与收益口径"):
        st.caption(f"完整底表：{data.index[0]:%Y-%m-%d} 至 {data.index[-1]:%Y-%m-%d} · {len(data):,} 条共同日线 · MT5 导出快照")
        st.dataframe(data.reset_index(), hide_index=True, width="stretch")
        st.download_button("下载日线底表", data.to_csv().encode("utf-8-sig"), "BTC_Gold_Daily.csv", "text/csv")
        st.markdown('''<div class="method-note">
        <p>日线取两品种共同日期，保留周日，不补值、不插值。BTCUSDm 日收盘除以 XAUUSDm 日收盘。</p>
        <p>每腿初始名义金额 100,000 美元，数量固定；收益以两腿合计 200,000 美元计算，不是保证金收益率。</p>
        <p>毛收益未计点差、佣金、滑点、隔夜费和实际合约手数约束。日线标签不能证明两腿报价完全同步；不利波动按日线收盘采样。</p>
        <p>不同窗口与偏离区间可能属于同一轮行情，事件统计不可直接复利成资金曲线。</p>
        </div>''', unsafe_allow_html=True)
