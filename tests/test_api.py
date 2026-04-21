from __future__ import annotations

import json
from pathlib import Path

FIXTURE_DIR = Path(__file__).parent / "fixtures"
SAMPLE = json.loads((FIXTURE_DIR / "sample_request.json").read_text())
BATCH = json.loads((FIXTURE_DIR / "sample_batch.json").read_text())


def test_root(api_client):
    r = api_client.get("/")
    assert r.status_code == 200
    assert "message" in r.json()


def test_healthz(api_client):
    r = api_client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_valid(api_client):
    r = api_client.post("/predict", json=SAMPLE)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["anomaly_score"] <= 1.0
    assert r.headers.get("x-request-id")


def test_predict_invalid_returns_422(api_client):
    bad = dict(SAMPLE)
    bad["src_port"] = 999_999
    r = api_client.post("/predict", json=bad)
    assert r.status_code == 422
    body = r.json()
    assert body["detail"].startswith("Invalid")
    assert body["errors"]


def test_predict_missing_field_returns_422(api_client):
    bad = dict(SAMPLE)
    bad.pop("duration")
    r = api_client.post("/predict", json=bad)
    assert r.status_code == 422


def test_batch_predict(api_client):
    r = api_client.post("/batch_predict", json=BATCH)
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["predictions"]) == len(BATCH["items"])
    for p in body["predictions"]:
        assert p["prediction"] in (0, 1)
        assert 0.0 <= p["anomaly_score"] <= 1.0


def test_batch_predict_empty_rejected(api_client):
    r = api_client.post("/batch_predict", json={"items": []})
    assert r.status_code == 422


def test_metrics_endpoint_has_custom_counter(api_client):
    api_client.post("/predict", json=SAMPLE)
    r = api_client.get("/metrics")
    assert r.status_code == 200
    body = r.text
    assert "predictions_total" in body
    assert "inference_latency_seconds" in body


def test_request_id_echoed_when_provided(api_client):
    r = api_client.post(
        "/predict", json=SAMPLE, headers={"x-request-id": "fixed-id-123"}
    )
    assert r.status_code == 200
    assert r.headers["x-request-id"] == "fixed-id-123"
