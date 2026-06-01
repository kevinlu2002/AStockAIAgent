from __future__ import annotations

from ashare_ai_agent.webapp import JOBS, JOBS_LOCK, _parse_tencent_minute_rows, _parse_tencent_quote, _set_job, create_app


def test_web_index_loads() -> None:
    app = create_app()
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert "A 股 AI 推荐" in response.get_data(as_text=True)


def test_time_api_loads() -> None:
    app = create_app()
    client = app.test_client()
    response = client.get("/api/time")
    assert response.status_code == 200
    data = response.get_json()
    assert data["timezone"] == "Asia/Shanghai"


def test_single_stock_analysis_rejects_invalid_code() -> None:
    app = create_app()
    client = app.test_client()
    response = client.post("/api/stock/analyze/start", json={"symbol": "abc"})
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_cancel_missing_analysis_job_returns_404() -> None:
    app = create_app()
    client = app.test_client()
    response = client.post("/api/analysis/not-found/cancel")
    assert response.status_code == 404
    assert response.get_json()["status"] == "missing"


def test_cancel_existing_analysis_job_marks_cancel_requested() -> None:
    app = create_app()
    client = app.test_client()
    job_id = "test-cancel-existing"
    _set_job(job_id, status="running", stage="test", progress=10, message="test")
    try:
        response = client.post(f"/api/analysis/{job_id}/cancel")
        assert response.status_code == 200
        assert response.get_json()["status"] == "running"
        with JOBS_LOCK:
            assert JOBS[job_id]["cancel_requested"] is True
    finally:
        with JOBS_LOCK:
            JOBS.pop(job_id, None)


def test_automation_status_loads() -> None:
    app = create_app()
    client = app.test_client()
    response = client.get("/api/automation/status")
    assert response.status_code == 200
    data = response.get_json()
    assert "auto_news_enabled" in data
    assert "auto_retrain_enabled" in data


def test_parse_tencent_quote() -> None:
    text = 'v_sh600028="1~中国石化~600028~4.86~4.83~4.78~1853378~1112853~737304~4.85~16272~4.84~26893~4.83~29239~4.82~26134~4.81~19933~4.86~15508~4.87~14231~4.88~30325~4.89~39363~4.90~70581~~20260528130504~0.03~0.62~4.90~4.78~4.86/1853378/900498851~1853378~90050~0.20~16.53~~4.90~4.78~2.48~4604.97~5876.98~0.71~5.31~4.35~1.25~-51537~4.86~8.64~18.48~~~0.11~90049.8851~0.0000~0~ ~GP-A~-21.36~-3.38~4.70~4.27~1.79~8.11~4.73~-5.63~-8.99~-24.77~94752475375~120925514222~-17.87~-15.51~94752475375~~~-12.62~0.00~~CNY~0~___D";'
    quote = _parse_tencent_quote(text, "600028")
    assert quote is not None
    assert quote["source"] == "tencent_realtime"
    assert quote["price"] == 4.86
    assert quote["date"] == "2026-05-28"


def test_parse_tencent_minute_rows() -> None:
    rows = _parse_tencent_minute_rows(
        ["0930 4.78 41668 19917304.00", "0931 4.79 105715 50604899.00"],
        "2026-05-28",
    )
    assert rows[0]["datetime"] == "2026-05-28 09:30:00"
    assert rows[1]["volume"] > 0
    assert rows[1]["price"] == 4.79
