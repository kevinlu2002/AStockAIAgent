# 云服务器部署说明

目标：让网站可从任意联网设备访问。

## 服务器要求

- Linux 云服务器，建议 Ubuntu 22.04/24.04。
- 至少 2 核 4GB 内存；如果要经常全 A 扫描，建议 4 核 8GB。
- 开放入站端口 `7860`，或使用 Nginx/Caddy 反向代理到 HTTPS。
- 已安装 Docker 和 Docker Compose。

## 上传项目

把整个 `D:\D\AStockAIAgent` 上传到服务器，例如放到：

```bash
/opt/AStockAIAgent
```

## 配置访问密码

在服务器项目目录创建 `.env`：

```bash
ASHARE_WEB_PASSWORD=换成你的强密码
ASHARE_SECRET_KEY=换成一串随机长字符串
ASHARE_AUTO_NEWS=1
ASHARE_NEWS_INTERVAL_SECONDS=900
ASHARE_AUTO_KNOWLEDGE=1
ASHARE_KNOWLEDGE_INTERVAL_SECONDS=86400
ASHARE_AUTO_RETRAIN=1
ASHARE_RETRAIN_TIME=16:30
ASHARE_FETCH_WORKERS=8
```

## 启动

```bash
cd /opt/AStockAIAgent
docker compose up -d --build
```

访问：

```text
http://服务器公网IP:7860
```

## 停止、重启、查看日志

```bash
cd /opt/AStockAIAgent
docker compose down
docker compose up -d
docker compose logs -f --tail=100
```

`docker-compose.yml` 已配置 `restart: unless-stopped`。只要 Docker 服务随系统启动，云服务器重启后网站会自动恢复。

## 更新模型或代码

```bash
cd /opt/AStockAIAgent
docker compose up -d --build
```

## 自动新闻和每日重训

- `ASHARE_AUTO_NEWS=1`：后台自动抓取国际新闻并刷新潜力股，默认每 `900` 秒一次。
- `ASHARE_AUTO_KNOWLEDGE=1`：后台自动抓取公开技术分析来源，更新 K 线/量价规则权重，默认每天一次。
- `ASHARE_KNOWLEDGE_URLS`：可选，逗号分隔的公开文章或博主 RSS/页面地址；系统只抽取主题权重，不复制原文。
- `ASHARE_AUTO_RETRAIN=1`：每天上海时间 `16:30` 后自动更新已有股票池历史行情并重新训练模型。
- `ASHARE_RETRAIN_SYMBOL_LIMIT`：可选，限制每日重训股票数，适合小服务器先设为 `300` 或 `1000`。

每日重训会提高模型对最新行情的适应性，但不保证性能一定变好。它更容易改善“数据新鲜度”和“新趋势识别”，也可能因为样本噪声、接口失败或短期异常行情导致验证指标波动。建议观察 `reports/daily_update_train_status.json` 和 `models/metrics.json`。

## 生产建议

公网长期使用建议加域名和 HTTPS。最简单方式是在服务器安装 Caddy，然后把域名反向代理到 `127.0.0.1:7860`。

`Caddyfile` 示例：

```text
your-domain.com {
    reverse_proxy 127.0.0.1:7860
}
```

## 注意

- 如果云服务器关机，网站仍然无法访问；但云服务器一般 24 小时运行。
- 全 A 扫描会访问外部行情接口，首次运行可能很慢。
- 这是研究辅助系统，不应直接无人值守自动交易。
