const state = {
  recommendations: [],
  selected: null,
  quoteTimer: null,
  chartTimer: null,
  progressTimer: null,
  currentJobId: null,
  chart: {
    mode: "daily",
    rows: [],
    timeline: [],
    forecast: [],
    forecastSummary: null,
    offset: 0,
    visible: 90,
    hoverIndex: null,
    dragging: false,
    dragX: 0,
    dragOffset: 0,
  },
};

const el = {
  liveClock: document.querySelector("#liveClock"),
  timeInput: document.querySelector("#timeInput"),
  statusLine: document.querySelector("#statusLine"),
  capitalInput: document.querySelector("#capitalInput"),
  holdingInput: document.querySelector("#holdingInput"),
  universeInput: document.querySelector("#universeInput"),
  symbolInput: document.querySelector("#symbolInput"),
  recommendBtn: document.querySelector("#recommendBtn"),
  singleStockBtn: document.querySelector("#singleStockBtn"),
  cancelAnalysisBtn: document.querySelector("#cancelAnalysisBtn"),
  generatedAt: document.querySelector("#generatedAt"),
  newsBtn: document.querySelector("#newsBtn"),
  retrainBtn: document.querySelector("#retrainBtn"),
  evaluationBtn: document.querySelector("#evaluationBtn"),
  automationStatus: document.querySelector("#automationStatus"),
  evaluationStatus: document.querySelector("#evaluationStatus"),
  newsGenerated: document.querySelector("#newsGenerated"),
  newsList: document.querySelector("#newsList"),
  groups: document.querySelector("#recommendationGroups"),
  progressStage: document.querySelector("#progressStage"),
  progressPct: document.querySelector("#progressPct"),
  progressFill: document.querySelector("#progressFill"),
  progressMessage: document.querySelector("#progressMessage"),
  selectedTitle: document.querySelector("#selectedTitle"),
  quoteSource: document.querySelector("#quoteSource"),
  quotePrice: document.querySelector("#quotePrice"),
  quotePct: document.querySelector("#quotePct"),
  quoteHigh: document.querySelector("#quoteHigh"),
  quoteLow: document.querySelector("#quoteLow"),
  quoteOpen: document.querySelector("#quoteOpen"),
  quoteTurnover: document.querySelector("#quoteTurnover"),
  quoteAmount: document.querySelector("#quoteAmount"),
  quoteMarketCap: document.querySelector("#quoteMarketCap"),
  klineLegend: document.querySelector("#klineLegend"),
  klineTabs: document.querySelectorAll(".kline-tabs button"),
  refreshQuoteBtn: document.querySelector("#refreshQuoteBtn"),
  selectedMeta: document.querySelector("#selectedMeta"),
  canvas: document.querySelector("#klineCanvas"),
};

function fmtNumber(value, digits = 2) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "--";
  return n.toFixed(digits);
}

function fmtPct(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "--";
  return `${(n * 100).toFixed(2)}%`;
}

function fmtPctRaw(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "--";
  return `${n.toFixed(2)}%`;
}

function fmtWan(value) {
  const n = Number(value);
  if (!Number.isFinite(n) || n === 0) return "--";
  if (Math.abs(n) >= 100000000) return `${(n / 100000000).toFixed(2)}亿`;
  if (Math.abs(n) >= 10000) return `${(n / 10000).toFixed(2)}万`;
  return n.toFixed(0);
}

function nowText() {
  const d = new Date();
  const pad = (v) => String(v).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

function tickClock() {
  const text = nowText();
  el.liveClock.textContent = text.slice(11);
  el.timeInput.value = text;
}

async function getJson(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || response.statusText);
  }
  return response.json();
}

async function loadStatus() {
  try {
    const status = await getJson("/api/status");
    const horizons = Array.isArray(status.available_horizons) ? status.available_horizons.join("/") : status.model_horizon_days;
    el.statusLine.textContent = `${status.symbols} 只基础候选 · ${status.model_backend || "模型已加载"} · ${horizons}日模型 · 最新缓存 ${status.latest_cached_date || "--"}`;
  } catch (error) {
    el.statusLine.textContent = "模型状态读取失败";
  }
}

function setProgress(job) {
  const pct = Math.max(0, Math.min(100, Number(job.progress || 0)));
  el.progressStage.textContent = job.stage || "分析中";
  el.progressPct.textContent = `${pct}%`;
  el.progressFill.style.width = `${pct}%`;
  const count = job.total ? ` (${job.processed || 0}/${job.total})` : "";
  el.progressMessage.textContent = `${job.message || ""}${count}`;
}

function setAnalyzeBusy(isBusy) {
  el.recommendBtn.disabled = isBusy;
  el.singleStockBtn.disabled = isBusy;
  el.cancelAnalysisBtn.disabled = !isBusy || !state.currentJobId;
  el.recommendBtn.textContent = isBusy ? "分析中" : "开始分析";
  el.singleStockBtn.textContent = isBusy ? "分析中" : "分析代码";
  if (!isBusy) el.cancelAnalysisBtn.textContent = "停止分析";
}

function normalizeSymbolInput(value) {
  const match = String(value || "").match(/\d{6}/);
  return match ? match[0] : "";
}

function actionClass(action) {
  return String(action || "").toLowerCase();
}

function actionLabel(action) {
  return { BUY: "买入", WATCH: "观察", AVOID: "回避" }[action] || action || "--";
}

function createMetric(label, value) {
  const box = document.createElement("div");
  box.className = "metric";
  box.innerHTML = `<span>${label}</span><strong>${value}</strong>`;
  return box;
}

function clearNode(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function renderNewsImpact(data) {
  clearNode(el.newsList);
  const rows = data.themes || [];
  el.newsGenerated.textContent = data.generated_at
    ? `生成 ${String(data.generated_at).slice(0, 19).replace("T", " ")} · ${data.article_count || 0} 条新闻 · 新闻综合推荐会结合本金和模型评分`
    : "暂无新闻数据";
  if (!rows.length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "暂无可用新闻主题";
    el.newsList.appendChild(empty);
    return;
  }

  rows.forEach((row) => {
    const card = document.createElement("article");
    card.className = "news-row";

    const main = document.createElement("div");
    main.className = "news-main";

    const title = document.createElement("strong");
    title.textContent = row.theme || row.key || "新闻主题";
    main.appendChild(title);

    const meta = document.createElement("span");
    const stockText = (row.stocks || []).slice(0, 4).map((item) => `${item.symbol} ${item.name || ""}`).join("、");
    meta.textContent = `主题分 ${fmtNumber(row.score, 2)} · ${stockText}`;
    main.appendChild(meta);

    const reason = document.createElement("p");
    reason.textContent = row.affected_logic || "";
    main.appendChild(reason);

    const evidence = document.createElement("div");
    evidence.className = "news-evidence";
    (row.evidence || []).slice(0, 3).forEach((item) => {
      const link = document.createElement("a");
      link.href = item.link || "#";
      link.target = "_blank";
      link.rel = "noreferrer";
      link.textContent = `${item.source || "news"}：${item.title || ""}`;
      evidence.appendChild(link);
    });
    main.appendChild(evidence);

    const action = document.createElement("button");
    action.type = "button";
    action.textContent = "综合";
    action.addEventListener("click", startNewsRecommendation);

    card.appendChild(main);
    card.appendChild(action);
    el.newsList.appendChild(card);
  });
}

async function loadNewsImpact() {
  el.newsBtn.disabled = true;
  el.newsBtn.textContent = "抓取中";
  el.newsGenerated.textContent = "正在抓取最新国际新闻";
  el.newsList.innerHTML = '<div class="empty">正在分析新闻催化剂...</div>';
  try {
    const data = await getJson("/api/news/impact?refresh=1&limit=12");
    renderNewsImpact(data);
  } catch (error) {
    el.newsGenerated.textContent = "新闻抓取失败";
    el.newsList.innerHTML = `<div class="empty">新闻推演失败：${error.message}</div>`;
  } finally {
    el.newsBtn.disabled = false;
    el.newsBtn.textContent = "新闻综合推荐";
  }
}

async function startNewsRecommendation() {
  const capital = Number(el.capitalInput.value || 0);
  const holding = el.holdingInput.value ? Number(el.holdingInput.value) : null;
  setAnalyzeBusy(true);
  el.newsBtn.disabled = true;
  el.newsBtn.textContent = "综合中";
  el.groups.innerHTML = '<div class="empty">正在综合近期新闻、模型评分和本金约束...</div>';
  setProgress({ progress: 1, stage: "提交新闻综合任务", message: "正在抓取新闻并生成候选股池" });
  try {
    const data = await getJson("/api/analysis/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        capital,
        holding_days: holding,
        universe_mode: "news",
        refresh_news: true,
        realtime: el.timeInput.value,
      }),
    });
    state.currentJobId = data.job_id;
    setAnalyzeBusy(true);
    if (state.progressTimer) clearInterval(state.progressTimer);
    state.progressTimer = setInterval(pollProgress, 1000);
    await pollProgress();
  } catch (error) {
    el.groups.innerHTML = `<div class="empty">新闻综合推荐启动失败：${error.message}</div>`;
    state.currentJobId = null;
    setAnalyzeBusy(false);
    el.newsBtn.disabled = false;
    el.newsBtn.textContent = "新闻综合推荐";
  }
}

function renderAutomationStatus(status) {
  const newsMode = status.auto_news_enabled ? `自动新闻 ${status.news_interval_seconds || "--"}秒/次` : "自动新闻未开启";
  const knowledgeMode = status.auto_knowledge_enabled ? "技术知识自动学习已开启" : "技术知识自动学习未开启";
  const retrainMode = status.auto_retrain_enabled ? `每日 ${status.retrain_time || "16:30"} 重训` : "自动重训未开启";
  const retrainProgress = status.retrain_progress
    ? ` · ${status.retrain_progress.stage || ""} ${status.retrain_progress.progress || 0}%`
    : "";
  const running = [
    status.news_running ? "新闻抓取中" : "",
    status.knowledge_running ? "技术知识学习中" : "",
    status.retrain_running ? "模型重训中" : "",
  ].filter(Boolean).join(" · ");
  const lastTrain = status.last_retrain && status.last_retrain.finished_at
    ? ` · 上次重训 ${String(status.last_retrain.finished_at).replace("T", " ").slice(0, 19)}`
    : "";
  const error = status.last_error ? ` · 最近错误 ${status.last_error}` : "";
  el.automationStatus.textContent = `${newsMode} · ${knowledgeMode} · ${retrainMode}${running ? " · " + running : ""}${retrainProgress}${lastTrain}${error}`;
}

function renderEvaluationStatus(data) {
  const kline = data.kline || {};
  const recommendations = data.recommendations || {};
  const klineHit = kline.direction_hit_rate === null || kline.direction_hit_rate === undefined
    ? "--"
    : fmtPct(kline.direction_hit_rate);
  const recHit = recommendations.direction_hit_rate === null || recommendations.direction_hit_rate === undefined
    ? "--"
    : fmtPct(recommendations.direction_hit_rate);
  const closeErr = kline.mean_abs_close_error_pct === null || kline.mean_abs_close_error_pct === undefined
    ? "--"
    : fmtPct(kline.mean_abs_close_error_pct);
  const returnErr = recommendations.mean_abs_return_error === null || recommendations.mean_abs_return_error === undefined
    ? "--"
    : fmtPct(recommendations.mean_abs_return_error);
  const evaluatedAt = data.evaluated_at ? String(data.evaluated_at).replace("T", " ").slice(0, 19) : "尚未回测";
  el.evaluationStatus.textContent =
    `预测回测 ${evaluatedAt} · K线 ${kline.evaluated || 0}/${kline.snapshots || 0} 方向命中 ${klineHit} 收盘误差 ${closeErr} · 推荐 ${recommendations.evaluated || 0}/${recommendations.snapshots || 0} 方向命中 ${recHit} 收益误差 ${returnErr}`;
}

async function loadAutomationStatus() {
  try {
    const status = await getJson("/api/automation/status");
    renderAutomationStatus(status);
  } catch (error) {
    el.automationStatus.textContent = `自动任务状态读取失败：${error.message}`;
  }
}

async function loadEvaluationStatus() {
  try {
    const data = await getJson("/api/evaluation/status");
    renderEvaluationStatus(data);
  } catch (error) {
    el.evaluationStatus.textContent = `预测回测状态读取失败：${error.message}`;
  }
}

async function runEvaluation() {
  el.evaluationBtn.disabled = true;
  el.evaluationBtn.textContent = "回测中";
  try {
    const data = await getJson("/api/evaluation/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refresh_history: false }),
    });
    renderEvaluationStatus(data);
  } catch (error) {
    el.evaluationStatus.textContent = `预测回测失败：${error.message}`;
  } finally {
    el.evaluationBtn.disabled = false;
    el.evaluationBtn.textContent = "回测预测";
  }
}

async function runManualRetrain() {
  el.retrainBtn.disabled = true;
  el.retrainBtn.textContent = "已提交";
  try {
    await getJson("/api/automation/retrain/run", { method: "POST" });
    await loadAutomationStatus();
  } catch (error) {
    el.automationStatus.textContent = `手动重训启动失败：${error.message}`;
  } finally {
    setTimeout(() => {
      el.retrainBtn.disabled = false;
      el.retrainBtn.textContent = "手动重训";
    }, 3000);
  }
}

function renderRecommendations(rows) {
  el.groups.innerHTML = "";
  if (!rows.length) {
    el.groups.innerHTML = '<div class="empty">暂无推荐结果</div>';
    return;
  }

  const buckets = ["0-10", "10-100", "100+"];
  buckets.forEach((bucket) => {
    const groupRows = rows.filter((row) => row.price_bucket === bucket);
    if (!groupRows.length) return;

    const section = document.createElement("section");
    const title = document.createElement("div");
    title.className = "bucket-title";
    title.innerHTML = `<strong>${bucket}</strong><span>${groupRows.length} 个候选</span>`;
    section.appendChild(title);

    const list = document.createElement("div");
    list.className = "stock-list";
    groupRows.forEach((row) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "stock-row";
      button.dataset.symbol = row.symbol;

      const symbol = document.createElement("div");
      symbol.className = "symbol-block";
      symbol.innerHTML = `<strong>${row.symbol} ${row.name || ""}</strong><span class="action ${actionClass(row.action)}">${actionLabel(row.action)}</span>`;
      button.appendChild(symbol);
      button.appendChild(createMetric("预测收益", fmtPct(row.pred_return)));
      button.appendChild(createMetric("10日涨跌", `${row.direction_prediction || "--"} ${fmtPct(row.direction_probability)}`));
      button.appendChild(createMetric("趋势概率", fmtPct(row.trend_probability)));
      button.appendChild(createMetric("预测回撤", fmtPct(row.pred_drawdown)));
      button.appendChild(createMetric("活知识", fmtNumber(row.knowledge_score, 2)));
      button.appendChild(createMetric("新闻分", fmtNumber(row.news_score, 2)));
      button.appendChild(createMetric("烂板状态", row.limit_up_state || "--"));
      button.appendChild(createMetric("买入区间", row.buy_price_range || "--"));
      button.appendChild(createMetric("建议数量", `${Number(row.shares || 0)} 股`));
      button.addEventListener("click", () => selectStock(row));
      list.appendChild(button);
    });
    section.appendChild(list);
    el.groups.appendChild(section);
  });
}

async function startAnalysis(mode = "universe") {
  const capital = Number(el.capitalInput.value || 0);
  const holding = el.holdingInput.value ? Number(el.holdingInput.value) : null;
  const universe = el.universeInput.value || "sample300";
  const symbol = normalizeSymbolInput(el.symbolInput.value);
  if (mode === "single" && !symbol) {
    setProgress({ progress: 0, stage: "等待输入", message: "请输入 6 位股票代码，例如 000001 或 600519" });
    el.groups.innerHTML = '<div class="empty">请输入 6 位股票代码后再分析。</div>';
    return;
  }

  setAnalyzeBusy(true);
  el.groups.innerHTML = mode === "single"
    ? `<div class="empty">正在在线抓取 ${symbol} 历史行情并生成建议...</div>`
    : '<div class="empty">正在分析候选股票...</div>';
  setProgress({ progress: 1, stage: "提交任务", message: "任务已提交" });

  try {
    const url = mode === "single" ? "/api/stock/analyze/start" : "/api/analysis/start";
    const body = mode === "single"
      ? { symbol, capital, holding_days: holding, realtime: el.timeInput.value }
      : { capital, holding_days: holding, universe_mode: universe, realtime: el.timeInput.value };
    const data = await getJson(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    state.currentJobId = data.job_id;
    setAnalyzeBusy(true);
    if (state.progressTimer) clearInterval(state.progressTimer);
    state.progressTimer = setInterval(pollProgress, 1000);
    await pollProgress();
  } catch (error) {
    el.groups.innerHTML = `<div class="empty">启动失败：${error.message}</div>`;
    state.currentJobId = null;
    setAnalyzeBusy(false);
  }
}

async function cancelAnalysis() {
  if (!state.currentJobId) return;
  el.cancelAnalysisBtn.disabled = true;
  el.cancelAnalysisBtn.textContent = "停止中";
  setProgress({ progress: 100, stage: "正在停止", message: "已通知后台停止当前分析任务" });
  try {
    await getJson(`/api/analysis/${state.currentJobId}/cancel`, { method: "POST" });
  } catch (error) {
    el.progressMessage.textContent = `停止失败：${error.message}`;
    el.cancelAnalysisBtn.disabled = false;
    el.cancelAnalysisBtn.textContent = "停止分析";
  }
}

async function pollProgress() {
  if (!state.currentJobId) return;
  try {
    const job = await getJson(`/api/analysis/${state.currentJobId}`);
    setProgress(job);
    if (job.status === "done") {
      clearInterval(state.progressTimer);
      setAnalyzeBusy(false);
      el.newsBtn.disabled = false;
      el.newsBtn.textContent = "新闻综合推荐";
      if (job.result && job.result.news) renderNewsImpact(job.result.news);
      state.recommendations = (job.result && job.result.best) || [];
      el.generatedAt.textContent = `生成 ${job.generated_at} · 用时 ${job.elapsed_seconds || "--"} 秒`;
      renderRecommendations(state.recommendations);
      if (state.recommendations.length) selectStock(state.recommendations[0]);
      state.currentJobId = null;
      return;
    }
    if (job.status === "error") {
      clearInterval(state.progressTimer);
      setAnalyzeBusy(false);
      el.newsBtn.disabled = false;
      el.newsBtn.textContent = "新闻综合推荐";
      el.groups.innerHTML = `<div class="empty">分析失败：${job.message || job.error}</div>`;
      state.currentJobId = null;
      return;
    }
    if (job.status === "cancelled") {
      clearInterval(state.progressTimer);
      setAnalyzeBusy(false);
      el.newsBtn.disabled = false;
      el.newsBtn.textContent = "新闻综合推荐";
      state.recommendations = [];
      el.generatedAt.textContent = "分析已停止";
      el.groups.innerHTML = '<div class="empty">本次分析已停止，没有生成新的推荐结果。</div>';
      state.currentJobId = null;
      return;
    }
  } catch (error) {
    clearInterval(state.progressTimer);
    setAnalyzeBusy(false);
    el.newsBtn.disabled = false;
    el.newsBtn.textContent = "新闻综合推荐";
    el.groups.innerHTML = `<div class="empty">进度读取失败：${error.message}</div>`;
    state.currentJobId = null;
  }
}

function updateQuoteClass(node, value) {
  node.classList.remove("up", "down");
  const n = Number(value);
  if (n > 0) node.classList.add("up");
  if (n < 0) node.classList.add("down");
}

async function loadQuote(symbol) {
  const quote = await getJson(`/api/quote/${symbol}`);
  el.quotePrice.textContent = fmtNumber(quote.price, 2);
  el.quotePct.textContent = fmtPctRaw(quote.pct_chg);
  updateQuoteClass(el.quotePrice, quote.change);
  updateQuoteClass(el.quotePct, quote.pct_chg);
  el.quoteHigh.textContent = fmtNumber(quote.high, 2);
  el.quoteLow.textContent = fmtNumber(quote.low, 2);
  el.quoteOpen.textContent = fmtNumber(quote.open, 2);
  el.quoteTurnover.textContent = quote.turnover ? fmtPctRaw(quote.turnover) : "--";
  el.quoteAmount.textContent = fmtWan(quote.amount);
  el.quoteMarketCap.textContent = fmtWan(quote.total_market_cap);
  el.quoteSource.textContent = `${quote.source} · ${quote.date || ""} ${quote.time || ""}`;
}

function setKlineMode(mode) {
  state.chart.mode = mode || "daily";
  el.klineTabs.forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === state.chart.mode);
  });
}

async function loadKline(symbol, mode = state.chart.mode) {
  setKlineMode(mode);
  if (state.chart.mode === "intraday" || state.chart.mode === "five") {
    const range = state.chart.mode === "five" ? "5d" : "1d";
    const data = await getJson(`/api/timeline/${symbol}?range=${range}`);
    state.chart.timeline = data.rows || [];
    state.chart.rows = [];
    state.chart.forecast = [];
    state.chart.forecastSummary = data;
    state.chart.offset = 0;
    state.chart.visible = Math.max(60, state.chart.timeline.length);
    state.chart.hoverIndex = null;
    drawChart();
    return;
  }
  const period = state.chart.mode === "weekly" ? "weekly" : state.chart.mode === "monthly" ? "monthly" : "daily";
  const forecastDays = period === "daily" ? 8 : 0;
  const data = await getJson(`/api/kline/${symbol}?limit=260&forecast_days=${forecastDays}&period=${period}`);
  state.chart.rows = data.rows || [];
  state.chart.timeline = [];
  state.chart.forecast = data.forecast || [];
  state.chart.forecastSummary = data.forecast_summary || null;
  state.chart.offset = 0;
  state.chart.visible = Math.min(110, Math.max(45, state.chart.rows.length + state.chart.forecast.length));
  state.chart.hoverIndex = null;
  drawChart();
}

function setSelectedRow(symbol) {
  document.querySelectorAll(".stock-row").forEach((node) => {
    node.classList.toggle("selected", node.dataset.symbol === symbol);
  });
}

async function selectStock(row) {
  state.selected = row;
  setSelectedRow(row.symbol);
  el.selectedTitle.textContent = `${row.symbol} ${row.name || ""}`;
  el.refreshQuoteBtn.disabled = false;
  el.selectedMeta.innerHTML = `
    <div><span>推荐买入区间</span><strong>${row.buy_price_range || "--"}</strong></div>
    <div><span>买入时间</span><strong>${row.planned_buy_time || "--"}</strong></div>
    <div><span>入场规则</span><strong>${row.entry_rule || "--"}</strong></div>
    <div><span>参考价</span><strong>${fmtNumber(row.reference_entry, 2)}</strong></div>
    <div><span>止损价</span><strong>${fmtNumber(row.stop_loss, 2)}</strong></div>
    <div><span>建议数量</span><strong>${Number(row.shares || 0)} 股</strong></div>
    <div><span>计划金额</span><strong>${fmtNumber(row.cash, 2)}</strong></div>
    <div><span>风控模式</span><strong>${row.sizing_mode || "--"}</strong></div>
    <div><span>单股上限</span><strong>${fmtPct(row.effective_max_position_pct)}</strong></div>
    <div><span>单笔风险</span><strong>${fmtPct(row.effective_risk_per_trade_pct)}</strong></div>
    <div><span>持仓复核</span><strong>${row.planned_review_time || "--"}</strong></div>
    <div><span>10日涨跌预测</span><strong>${row.direction_prediction || "--"} ${fmtPct(row.direction_probability)}</strong></div>
    <div><span>趋势预测</span><strong>${row.trend_prediction || "--"} ${fmtPct(row.trend_probability)}</strong></div>
    <div><span>方向置信度</span><strong>${fmtPct(row.direction_confidence)}</strong></div>
    <div><span>趋势置信度</span><strong>${fmtPct(row.trend_confidence)}</strong></div>
    <div><span>模型评分</span><strong>${fmtNumber(row.score, 4)}</strong></div>
    <div><span>活知识分</span><strong>${fmtNumber(row.knowledge_score, 2)}</strong></div>
    <div><span>知识标签</span><strong>${row.knowledge_themes || "--"}</strong></div>
    <div><span>涨停/烂板</span><strong>${row.limit_up_state || "--"}</strong></div>
    <div><span>新闻主题</span><strong>${row.news_themes || "--"}</strong></div>
    <div><span>新闻分</span><strong>${fmtNumber(row.news_score, 2)}</strong></div>
    <div><span>K线信号</span><strong>${row.technical_signals || "--"}</strong></div>
    <div><span>理由</span><strong>${row.reason || "--"}</strong></div>
  `;
  if (state.quoteTimer) clearInterval(state.quoteTimer);
  if (state.chartTimer) clearInterval(state.chartTimer);
  try {
    await Promise.all([loadQuote(row.symbol), loadKline(row.symbol)]);
  } catch (error) {
    el.quoteSource.textContent = `加载失败：${error.message}`;
  }
  state.quoteTimer = setInterval(() => {
    if (state.selected) loadQuote(state.selected.symbol).catch(() => {});
  }, 10000);
  state.chartTimer = setInterval(() => {
    if (state.selected && (state.chart.mode === "intraday" || state.chart.mode === "five")) {
      loadKline(state.selected.symbol, state.chart.mode).catch(() => {});
    }
  }, 15000);
}

function movingAverage(rows, window) {
  const out = new Array(rows.length).fill(null);
  let sum = 0;
  for (let i = 0; i < rows.length; i += 1) {
    sum += Number(rows[i].close);
    if (i >= window) sum -= Number(rows[i - window].close);
    if (i >= window - 1) out[i] = sum / window;
  }
  return out;
}

function ema(values, span) {
  const out = new Array(values.length).fill(null);
  const alpha = 2 / (span + 1);
  let prev = null;
  values.forEach((value, i) => {
    const n = Number(value);
    prev = prev === null ? n : alpha * n + (1 - alpha) * prev;
    if (i >= span - 1) out[i] = prev;
  });
  return out;
}

function macdRows(rows) {
  const closes = rows.map((r) => Number(r.close));
  const ema12 = ema(closes, 12);
  const ema26 = ema(closes, 26);
  const dif = closes.map((_, i) => (ema12[i] === null || ema26[i] === null ? null : ema12[i] - ema26[i]));
  const dea = ema(dif.map((v) => v ?? 0), 9);
  return dif.map((value, i) => {
    if (value === null || dea[i] === null || i < 33) return { dif: null, dea: null, hist: null };
    return { dif: value, dea: dea[i], hist: (value - dea[i]) * 2 };
  });
}

function drawLine(ctx, values, startIndex, endIndex, xAt, yAt, color) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  let started = false;
  for (let i = startIndex; i < endIndex; i += 1) {
    const value = values[i];
    if (value === null || !Number.isFinite(value)) continue;
    const x = xAt(i - startIndex);
    const y = yAt(value);
    if (!started) {
      ctx.moveTo(x, y);
      started = true;
    } else {
      ctx.lineTo(x, y);
    }
  }
  if (started) ctx.stroke();
}

function drawChart() {
  if (state.chart.mode === "intraday" || state.chart.mode === "five") {
    drawTimeline();
    return;
  }
  drawKline();
}

function drawTimeline() {
  const canvas = el.canvas;
  const ctx = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(320, Math.round(rect.width * dpr));
  canvas.height = Math.max(220, Math.round(rect.height * dpr));
  ctx.scale(dpr, dpr);

  const w = rect.width;
  const h = rect.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, w, h);

  const rows = state.chart.timeline || [];
  if (!rows.length) {
    ctx.fillStyle = "#627183";
    ctx.font = "14px Microsoft YaHei, Segoe UI, Arial";
    ctx.fillText("暂无实时分时数据", 18, 28);
    if (el.klineLegend) el.klineLegend.textContent = "分时 --";
    return;
  }

  const pad = { left: 58, right: 64, top: 22, bottom: 24 };
  const priceTop = pad.top;
  const priceH = Math.max(220, h * 0.66);
  const volumeTop = priceTop + priceH + 20;
  const volumeH = Math.max(78, h - volumeTop - pad.bottom);
  const chartW = w - pad.left - pad.right;
  const prices = rows.map((r) => Number(r.price)).filter(Number.isFinite);
  const avgPrices = rows.map((r) => Number(r.avg_price)).filter(Number.isFinite);
  const allPrices = prices.concat(avgPrices);
  const maxPrice = Math.max(...allPrices);
  const minPrice = Math.min(...allPrices);
  const span = Math.max(0.01, maxPrice - minPrice);
  const y = (price) => priceTop + (maxPrice - price) / span * priceH;
  const xAt = (i) => pad.left + (rows.length <= 1 ? 0 : (chartW * i) / (rows.length - 1));

  ctx.strokeStyle = "#e5ebf1";
  ctx.lineWidth = 1;
  ctx.fillStyle = "#627183";
  ctx.font = "12px Microsoft YaHei, Segoe UI, Arial";
  for (let i = 0; i <= 4; i += 1) {
    const yy = priceTop + priceH * i / 4;
    const price = maxPrice - span * i / 4;
    ctx.beginPath();
    ctx.moveTo(pad.left, yy);
    ctx.lineTo(w - pad.right, yy);
    ctx.stroke();
    ctx.fillText(price.toFixed(2), w - pad.right + 8, yy + 4);
  }

  const drawSeries = (field, color, width = 1.4) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.beginPath();
    let started = false;
    rows.forEach((row, i) => {
      const value = Number(row[field]);
      if (!Number.isFinite(value)) return;
      const x = xAt(i);
      const yy = y(value);
      if (!started) {
        ctx.moveTo(x, yy);
        started = true;
      } else {
        ctx.lineTo(x, yy);
      }
    });
    if (started) ctx.stroke();
  };
  drawSeries("price", "#d3352a", 1.7);
  drawSeries("avg_price", "#d68b00", 1.1);

  const maxVolume = Math.max(...rows.map((r) => Number(r.volume) || 0), 1);
  rows.forEach((row, i) => {
    const volume = Number(row.volume) || 0;
    const prev = i > 0 ? Number(rows[i - 1].price) : Number(row.price);
    const price = Number(row.price);
    const barH = volume / maxVolume * volumeH;
    const x = xAt(i);
    const barW = Math.max(1, chartW / Math.max(rows.length, 1) * 0.65);
    ctx.fillStyle = price >= prev ? "rgba(192,54,44,0.62)" : "rgba(23,138,85,0.62)";
    ctx.fillRect(x - barW / 2, volumeTop + volumeH - barH, barW, Math.max(1, barH));
  });

  ctx.strokeStyle = "#e5ebf1";
  ctx.beginPath();
  ctx.moveTo(pad.left, volumeTop + volumeH);
  ctx.lineTo(w - pad.right, volumeTop + volumeH);
  ctx.stroke();
  ctx.fillStyle = "#627183";
  ctx.fillText("成交量", 10, volumeTop + 12);

  const first = rows[0];
  const last = rows[rows.length - 1];
  ctx.fillStyle = "#627183";
  ctx.fillText(state.chart.mode === "five" ? first.date.replaceAll("-", "") : first.time, pad.left, h - 8);
  ctx.fillText(state.chart.mode === "five" ? last.date.replaceAll("-", "") : last.time, w - pad.right - 72, h - 8);
  if (el.klineLegend) {
    el.klineLegend.textContent = `${state.chart.mode === "five" ? "五日分时" : "分时"}  最新:${fmtNumber(last.price, 2)}  均价:${fmtNumber(last.avg_price, 2)}  数据源:${state.chart.forecastSummary?.source || "--"}`;
  }

  if (state.chart.hoverIndex !== null) {
    const local = Math.max(0, Math.min(rows.length - 1, state.chart.hoverIndex));
    const row = rows[local];
    const x = xAt(local);
    ctx.strokeStyle = "rgba(23,32,42,0.45)";
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(x, priceTop);
    ctx.lineTo(x, volumeTop + volumeH);
    ctx.stroke();
    ctx.setLineDash([]);
    const tip = `${row.date} ${row.time}  价:${fmtNumber(row.price)} 均:${fmtNumber(row.avg_price)} 量:${Math.round(Number(row.volume) / 100)}`;
    ctx.fillStyle = "rgba(23,32,42,0.92)";
    const tipW = Math.min(w - 24, Math.max(300, ctx.measureText(tip).width + 20));
    const tipX = Math.min(w - tipW - 12, Math.max(12, x - tipW / 2));
    ctx.fillRect(tipX, priceTop + 8, tipW, 28);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(tip, tipX + 10, priceTop + 27);
  }
}

function drawKline() {
  const canvas = el.canvas;
  const ctx = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(320, Math.round(rect.width * dpr));
  canvas.height = Math.max(220, Math.round(rect.height * dpr));
  ctx.scale(dpr, dpr);

  const w = rect.width;
  const h = rect.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, w, h);

  const rows = state.chart.rows;
  const forecastRows = state.chart.forecast || [];
  const combinedRows = rows.concat(forecastRows);
  if (!combinedRows.length) {
    ctx.fillStyle = "#627183";
    ctx.font = "14px Microsoft YaHei, Segoe UI, Arial";
    ctx.fillText("暂无 K 线数据", 18, 28);
    return;
  }

  const visible = Math.max(25, Math.min(state.chart.visible, combinedRows.length));
  const start = Math.max(0, combinedRows.length - visible - state.chart.offset);
  const end = Math.min(combinedRows.length, start + visible);
  const view = combinedRows.slice(start, end);
  const maBase = rows.concat(forecastRows.map((item) => ({ ...item, close: item.close })));
  const ma5 = movingAverage(maBase, 5);
  const ma10 = movingAverage(maBase, 10);
  const ma20 = movingAverage(maBase, 20);
  const macd = macdRows(maBase);
  const latestMa = (values) => {
    const value = [...values].reverse().find((item) => Number.isFinite(item));
    return fmtNumber(value, 2);
  };
  if (el.klineLegend) {
    el.klineLegend.textContent = `MA5:${latestMa(ma5)}  MA10:${latestMa(ma10)}  MA20:${latestMa(ma20)}  预测:${forecastRows.length}日`;
  }

  const pad = { left: 58, right: 64, top: 24, bottom: 24 };
  const priceTop = pad.top;
  const priceH = Math.max(180, h * 0.58);
  const volumeTop = priceTop + priceH + 18;
  const volumeH = Math.max(60, h * 0.16);
  const macdTop = volumeTop + volumeH + 16;
  const macdH = Math.max(58, h - macdTop - pad.bottom);
  const chartW = w - pad.left - pad.right;
  const highs = view.map((r) => Number(r.high));
  const lows = view.map((r) => Number(r.low));
  const maxPrice = Math.max(...highs);
  const minPrice = Math.min(...lows);
  const span = Math.max(0.01, maxPrice - minPrice);
  const y = (price) => priceTop + (maxPrice - price) / span * priceH;
  const step = chartW / view.length;
  const bodyW = Math.max(3, Math.min(10, step * 0.62));
  const xAt = (localIndex) => pad.left + step * localIndex + step / 2;

  ctx.strokeStyle = "#e5ebf1";
  ctx.lineWidth = 1;
  ctx.fillStyle = "#627183";
  ctx.font = "12px Microsoft YaHei, Segoe UI, Arial";
  for (let i = 0; i <= 4; i += 1) {
    const yy = priceTop + priceH * i / 4;
    const price = maxPrice - span * i / 4;
    ctx.beginPath();
    ctx.moveTo(pad.left, yy);
    ctx.lineTo(w - pad.right, yy);
    ctx.stroke();
    ctx.fillText(price.toFixed(2), w - pad.right + 8, yy + 4);
  }

  view.forEach((r, i) => {
    const open = Number(r.open);
    const close = Number(r.close);
    const high = Number(r.high);
    const low = Number(r.low);
    const x = xAt(i);
    const up = close >= open;
    const alpha = r.forecast ? 0.28 : 1;
    ctx.strokeStyle = up ? `rgba(192,54,44,${alpha})` : `rgba(23,138,85,${alpha})`;
    ctx.fillStyle = up ? `rgba(192,54,44,${alpha})` : `rgba(23,138,85,${alpha})`;
    ctx.beginPath();
    ctx.moveTo(x, y(high));
    ctx.lineTo(x, y(low));
    ctx.stroke();
    const top = y(Math.max(open, close));
    const bottom = y(Math.min(open, close));
    ctx.fillRect(x - bodyW / 2, top, bodyW, Math.max(1, bottom - top));
  });

  drawLine(ctx, ma5, start, end, xAt, y, "#d68b00");
  drawLine(ctx, ma10, start, end, xAt, y, "#111827");
  drawLine(ctx, ma20, start, end, xAt, y, "#cc2b8a");
  ctx.fillStyle = "#d68b00";
  ctx.fillText("MA5", pad.left, 14);
  ctx.fillStyle = "#111827";
  ctx.fillText("MA10", pad.left + 38, 14);
  ctx.fillStyle = "#cc2b8a";
  ctx.fillText("MA20", pad.left + 84, 14);
  if (forecastRows.length) {
    ctx.fillStyle = "rgba(98,113,131,0.90)";
    const summary = state.chart.forecastSummary || {};
    const text = `半透明为未来${forecastRows.length}个交易日预测 · 目标收益 ${fmtPct(summary.target_return)} · 建议窗口 ${summary.suggested_future_entry_window || "--"}`;
    ctx.fillText(text, pad.left + 130, 14);
  }

  const maxVolume = Math.max(...view.map((r) => Number(r.volume) || 0), 1);
  ctx.strokeStyle = "#e5ebf1";
  ctx.beginPath();
  ctx.moveTo(pad.left, volumeTop + volumeH);
  ctx.lineTo(w - pad.right, volumeTop + volumeH);
  ctx.stroke();
  view.forEach((r, i) => {
    const volume = Number(r.volume) || 0;
    const open = Number(r.open);
    const close = Number(r.close);
    const barH = volume / maxVolume * volumeH;
    const alpha = r.forecast ? 0.18 : 0.45;
    ctx.fillStyle = close >= open ? `rgba(192,54,44,${alpha})` : `rgba(23,138,85,${alpha})`;
    ctx.fillRect(xAt(i) - bodyW / 2, volumeTop + volumeH - barH, bodyW, Math.max(1, barH));
  });
  ctx.fillStyle = "#627183";
  ctx.fillText("VOL", 10, volumeTop + 12);

  const macdView = macd.slice(start, end);
  const macdValues = macdView.flatMap((m) => [m.dif, m.dea, m.hist]).filter((v) => Number.isFinite(v));
  const macdAbs = Math.max(...macdValues.map((v) => Math.abs(v)), 0.01);
  const macdY = (value) => macdTop + macdH / 2 - value / macdAbs * (macdH / 2 - 6);
  ctx.strokeStyle = "#e5ebf1";
  ctx.beginPath();
  ctx.moveTo(pad.left, macdY(0));
  ctx.lineTo(w - pad.right, macdY(0));
  ctx.stroke();
  macdView.forEach((m, i) => {
    if (m.hist === null) return;
    const top = macdY(Math.max(m.hist, 0));
    const bottom = macdY(Math.min(m.hist, 0));
    const alpha = view[i]?.forecast ? 0.18 : 0.45;
    ctx.fillStyle = m.hist >= 0 ? `rgba(192,54,44,${alpha})` : `rgba(23,138,85,${alpha})`;
    ctx.fillRect(xAt(i) - bodyW / 2, top, bodyW, Math.max(1, bottom - top));
  });
  drawLine(ctx, macd.map((m) => m.dif), start, end, xAt, macdY, "#d68b00");
  drawLine(ctx, macd.map((m) => m.dea), start, end, xAt, macdY, "#126a8a");
  ctx.fillStyle = "#627183";
  ctx.fillText("MACD", 10, macdTop + 12);

  ctx.fillStyle = "#627183";
  const first = view[0].date.slice(5);
  const last = view[view.length - 1].date.slice(5);
  ctx.fillText(first, pad.left, h - 10);
  ctx.fillText(last, w - pad.right - 42, h - 10);

  const firstForecastLocal = view.findIndex((r) => r.forecast);
  if (firstForecastLocal >= 0) {
    const x = xAt(firstForecastLocal) - step / 2;
    ctx.strokeStyle = "rgba(18,106,138,0.45)";
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(x, priceTop);
    ctx.lineTo(x, macdTop + macdH);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "rgba(18,106,138,0.85)";
    ctx.fillText("预测", x + 6, priceTop + 14);
  }

  if (state.chart.hoverIndex !== null) {
    const local = Math.max(0, Math.min(view.length - 1, state.chart.hoverIndex));
    const row = view[local];
    const x = xAt(local);
    ctx.strokeStyle = "rgba(23,32,42,0.45)";
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(x, priceTop);
    ctx.lineTo(x, macdTop + macdH);
    ctx.stroke();
    const closeY = y(Number(row.close));
    ctx.beginPath();
    ctx.moveTo(pad.left, closeY);
    ctx.lineTo(w - pad.right, closeY);
    ctx.stroke();
    ctx.setLineDash([]);
    const prefix = row.forecast ? "预测 " : "";
    const tip = `${prefix}${row.date}  开:${fmtNumber(row.open)} 高:${fmtNumber(row.high)} 低:${fmtNumber(row.low)} 收:${fmtNumber(row.close)} 量:${Math.round(Number(row.volume) / 10000)}万`;
    ctx.fillStyle = "rgba(23,32,42,0.92)";
    const tipW = Math.min(w - 24, Math.max(320, ctx.measureText(tip).width + 20));
    const tipX = Math.min(w - tipW - 12, Math.max(12, x - tipW / 2));
    ctx.fillRect(tipX, priceTop + 8, tipW, 28);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(tip, tipX + 10, priceTop + 27);
  }
}

function chartLocalIndex(event) {
  const rect = el.canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const chartW = rect.width - 58 - 64;
  const count = state.chart.mode === "intraday" || state.chart.mode === "five"
    ? Math.max(1, state.chart.timeline.length)
    : Math.max(1, state.chart.visible);
  const step = chartW / count;
  return Math.round((x - 58 - step / 2) / step);
}

function clampChartOffset() {
  const maxOffset = Math.max(0, state.chart.rows.length + state.chart.forecast.length - state.chart.visible);
  state.chart.offset = Math.max(0, Math.min(maxOffset, state.chart.offset));
}

el.recommendBtn.addEventListener("click", () => startAnalysis("universe"));
el.singleStockBtn.addEventListener("click", () => startAnalysis("single"));
el.cancelAnalysisBtn.addEventListener("click", cancelAnalysis);
el.newsBtn.addEventListener("click", startNewsRecommendation);
el.retrainBtn.addEventListener("click", runManualRetrain);
el.evaluationBtn.addEventListener("click", runEvaluation);
el.klineTabs.forEach((button) => {
  button.addEventListener("click", () => {
    const mode = button.dataset.mode || "daily";
    if (mode === "more") return;
    setKlineMode(mode);
    if (state.selected) loadKline(state.selected.symbol, mode).catch((error) => {
      el.quoteSource.textContent = `图表加载失败：${error.message}`;
    });
  });
});
el.refreshQuoteBtn.addEventListener("click", () => {
  if (state.selected) loadQuote(state.selected.symbol).catch(() => {});
});
el.canvas.addEventListener("mousemove", (event) => {
  const hasData = state.chart.mode === "intraday" || state.chart.mode === "five"
    ? state.chart.timeline.length
    : state.chart.rows.length;
  if (!hasData) return;
  if (state.chart.dragging && state.chart.mode !== "intraday" && state.chart.mode !== "five") {
    const rect = el.canvas.getBoundingClientRect();
    const step = Math.max(1, (rect.width - 58 - 64) / Math.max(1, state.chart.visible));
    const delta = Math.round((event.clientX - state.chart.dragX) / step);
    state.chart.offset = state.chart.dragOffset + delta;
    clampChartOffset();
  }
  state.chart.hoverIndex = chartLocalIndex(event);
  drawChart();
});
el.canvas.addEventListener("mouseleave", () => {
  state.chart.hoverIndex = null;
  state.chart.dragging = false;
  drawChart();
});
el.canvas.addEventListener("mousedown", (event) => {
  if (state.chart.mode === "intraday" || state.chart.mode === "five") return;
  state.chart.dragging = true;
  state.chart.dragX = event.clientX;
  state.chart.dragOffset = state.chart.offset;
});
window.addEventListener("mouseup", () => {
  state.chart.dragging = false;
});
el.canvas.addEventListener("wheel", (event) => {
  if (!state.chart.rows.length || state.chart.mode === "intraday" || state.chart.mode === "five") return;
  event.preventDefault();
  const delta = event.deltaY > 0 ? 10 : -10;
  state.chart.visible = Math.max(30, Math.min(state.chart.rows.length, state.chart.visible + delta));
  clampChartOffset();
  drawChart();
}, { passive: false });
window.addEventListener("resize", () => {
  if (state.selected) drawChart();
});

tickClock();
setInterval(tickClock, 1000);
setInterval(loadAutomationStatus, 10000);
loadStatus();
loadAutomationStatus();
loadEvaluationStatus();
loadNewsImpact();
