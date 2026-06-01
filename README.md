# AStockAIAgent

AStockAIAgent is a research-oriented A-share market agent. It builds OHLCV features, trains proxy models for return and drawdown forecasting, turns model outputs into risk-aware candidate signals, and provides a local Flask dashboard for review. The project is intended for research, screening, and post-trade analysis, not automated investment advice.

这是一个面向 A 股历史行情的研究型 AI 代理项目。它会先用历史 OHLCV 数据训练代理模型，预测未来若干交易日的收益和回撤风险，再把模型输出转成可执行性更强的研究信号：候选股票、计划买入时间、计划买入数量、参考入场价、止损价和风控理由。

重要说明：本项目输出不是投资承诺，也不是确定性收益建议。A 股存在流动性、停牌、涨跌停、滑点、财报和政策冲击等风险，任何实盘交易前都应人工复核。

## 方法

- 数据：优先读取 `data/raw/{股票代码}.csv`；没有本地 CSV 时可用 AkShare 拉取日线历史行情。
- 特征：收益动量、均线偏离、波动率、成交量异常、ATR、RSI、MACD、换手率等。
- 代理模型：优先使用 LightGBM 回归器；如果当前环境没有 LightGBM，自动退回到 scikit-learn 的 HistGradientBoostingRegressor。
- 训练目标：分别预测未来 `horizon_days` 日收益率和区间最大回撤。
- 决策代理：按收益/回撤比、阈值、趋势和风控约束生成买入/观望/回避信号。
- 仓位：按资金、单笔风险、止损距离、最大仓位比例和 A 股 100 股一手规则计算。

## 快速开始

```powershell
cd D:\D\AStockAIAgent
python -m pip install -r requirements.txt
```

如果要自动拉取 A 股数据，额外安装：

```powershell
python -m pip install akshare
```

如果要使用更强的表格梯度提升后端，额外安装：

```powershell
python -m pip install lightgbm
```

无网络烟测可以先生成演示数据：

```powershell
python .\scripts\make_demo_data.py --out-dir .\data\raw
python -m ashare_ai_agent train --config .\configs\default.toml
python -m ashare_ai_agent recommend --config .\configs\default.toml --capital 100000
```

真实 A 股数据训练：

```powershell
python -m ashare_ai_agent fetch --config .\configs\default.toml --symbols 000001,600519,300750
python -m ashare_ai_agent train --config .\configs\default.toml
python -m ashare_ai_agent recommend --config .\configs\default.toml --symbols 000001,600519,300750 --capital 100000
```

全市场快照 + 按价格档推荐：

```powershell
python .\scripts\run_real_ashare_pipeline.py --per-bucket 20 --start-date 20210101 --capital 100000
```

这个脚本会先抓取最新 A 股实时股票池，按最新股价分为 `0-10`、`10-100`、`100+` 三档，每档优先选成交额较高的股票拉取历史数据、训练模型、做时间切分测试，并输出每档最佳候选。

启动本地网站：

```powershell
python .\scripts\run_web_app.py
```

打开 `http://127.0.0.1:7860`。页面支持输入本金、可选持仓天数、精确到秒的实时时间，并在选中推荐股票后展示实时股价和最近 K 线。

后台启动、停止和注册 Windows 开机自启：

```powershell
.\scripts\start_web_app.ps1
.\scripts\stop_web_app.ps1
.\scripts\register_windows_startup.ps1
```

网站推荐现在是后台异步分析任务：

- `已缓存股票`：只分析本地已有历史数据，速度最快。
- `全市场 300 只`：从 A 股股票池中抽样 300 只并补齐历史数据。
- `全市场 1000 只`：更大范围扫描，首次运行会更慢。
- `全部 A 股`：尽量扫描所有 A 股，包含最新上市股票；新股如果历史数据不足，会被读取但可能无法进入模型评分。

本金会影响是否买得起一手、仓位利用率和综合排序；持仓天数会影响收益/回撤阈值、技术因子权重和复核时间。
模型当前训练 `3/5/10/20/60/120` 日多周期代理；持仓几个月时会自动使用最接近的 60 或 120 日模型。

网站还支持两类补充分析：

- 指定股票代码：输入 `000001` 这类 6 位代码后，可在线抓取该股票最新历史数据并生成单股建议。
- 国际新闻影响：抓取公开国际新闻 RSS，按 AI/半导体、油气、黄金、军工、航运、新能源车、光伏储能、农业、医药、关键矿产等主题识别潜在催化剂，并给出可继续做模型评分的 A 股候选。

后台自动任务：

- `ASHARE_AUTO_NEWS=1`：自动抓取国际新闻并刷新潜力股。
- `ASHARE_NEWS_INTERVAL_SECONDS=900`：新闻刷新间隔，默认 15 分钟。
- `ASHARE_AUTO_KNOWLEDGE=1`：自动学习公开技术分析来源，更新 K 线/量价规则权重。
- `ASHARE_KNOWLEDGE_URLS=...`：可选，逗号分隔配置公开文章或博主 RSS/页面地址；系统只抽取主题权重，不复制原文。
- `ASHARE_AUTO_RETRAIN=1`：每天上海时间收盘后自动更新历史行情并重训模型。
- `ASHARE_RETRAIN_TIME=16:30`：每日重训触发时间。
- `ASHARE_RETRAIN_SYMBOL_LIMIT=1000`：可选，限制每日重训股票数。

K 线图会在真实行情右侧用低透明度蜡烛绘制未来若干交易日的模型预测路径。这个预测来自多周期收益/回撤模型、近期 ATR/波动率、均线/MACD/成交量/K 线形态规则和自动学习到的技术分析权重，不是确定价格承诺。

动态风控规则：

- `<= 5000` 元：小资金集中持仓，单股仓位上限最高 95%，单笔风险最高 5%。
- `5000-20000` 元：小资金偏集中，单股仓位上限最高 70%，单笔风险最高 3.5%。
- `20000-100000` 元：中等资金均衡，单股仓位上限最高 35%，单笔风险最高 2%。
- `100000-500000` 元：较大资金分散，单股仓位上限最高 25%，单笔风险最高 1.5%。
- `> 500000` 元：使用默认严格风控。

输出文件默认在：

- `models/proxy_model.joblib`
- `models/metrics.json`
- `reports/recommendations.csv`
- `reports/recommendations.md`
- `reports/real_data_test_by_price_bucket.csv`
- `reports/best_recommendations_by_price_bucket.csv`

## 本地 CSV 格式

支持英文列名或 AkShare 中文列名。推荐英文列名：

```text
date,symbol,open,high,low,close,volume,amount,turnover
2024-01-02,600519,1700,1725,1688,1712,123456,210000000,0.31
```

`date/open/high/low/close/volume` 是必需列；`amount/turnover` 可选。

## 常用参数

修改 `configs/default.toml`：

- `data.symbols`：股票池。
- `model.horizon_days`：预测未来多少个交易日。
- `risk.capital`：总资金。
- `risk.risk_per_trade_pct`：单笔最大风险占总资金比例。
- `risk.max_position_pct`：单只股票最大仓位比例。
- `risk.min_expected_return`：买入信号最低预测收益。
- `risk.max_expected_drawdown`：允许的预测最大回撤下限。

## 使用边界

这个项目适合做研究、筛选和复盘，不应直接无人值守地下单。实盘前至少需要补充：

- 交易日历和节假日处理。
- 停牌、涨跌停、ST、退市风险过滤。
- 交易成本、滑点和冲击成本。
- 分钟级入场确认。
- 财报、公告和行业事件风控。
