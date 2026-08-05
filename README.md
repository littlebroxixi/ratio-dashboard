# 指数比值趋势偏离看板

用于监控以下三组指数比值：

- IC / IF：中证500 / 沪深300
- IM / IC：中证1000 / 中证500
- IM / IF：中证1000 / 沪深300

## 当前模型

看板使用3年、5年、7年滚动最小二乘模型。每个交易日只使用当天以前的
`年数 × 252`个交易日拟合趋势线，因此当天数据不会影响自己的趋势中枢。

- 趋势中枢：OLS趋势线对当天比值的一步预测
- 残差：当天实际比值减去趋势中枢
- 残差标准差：`sqrt(SSE / (N - 2))`
- Z-Score：`残差 / 残差标准差`
- 多周期确认：至少两个窗口在同一方向超过 `±2σ`

旧版的全局普通均值、固定标准差、建仓/重仓/All-in阈值和基于旧Z值的收益效率结论已移除，避免不同统计口径混用。

## 运行

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## 测试

```bash
python -m unittest -v test_analytics.py test_app_smoke.py
```
