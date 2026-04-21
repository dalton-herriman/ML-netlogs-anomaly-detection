# Anomaly Detection in Network Logs

[![CI](https://github.com/dalton-herriman/ML-netlogs-anomaly-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/dalton-herriman/ML-netlogs-anomaly-detection/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)

Production-style ML service for SOC anomaly detection on network flow logs. Trains an XGBoost classifier on CICIDS-2017-shaped flow features, serves predictions behind a FastAPI endpoint, and ships with MLflow experiment tracking plus a Prometheus + Grafana observability stack.

## Quickstart

```bash
git clone https://github.com/dalton-herriman/ML-netlogs-anomaly-detection.git && cd ML-netlogs-anomaly-detection
python scripts/generate_sample.py && python -m src.preprocess --input data/raw/sample.csv --output data/processed/ && python -m src.train --train data/processed/train.csv --test data/processed/test.csv
docker compose up --build -d && curl -s -X POST http://localhost:8000/predict -H 'content-type: application/json' -d @tests/fixtures/sample_request.json
```

That trains a toy model, starts the API + Prometheus + Grafana, and gets a prediction back — no external dataset required.

## What's actually in here

| Component | Path | Notes |
| --- | --- | --- |
| FastAPI service | `src/api.py` | `/predict`, `/batch_predict`, `/healthz`, `/metrics`, `/`. JSON structured logs, request-ID middleware, 422 on validation errors, 500 handler that never leaks internals, 503 when no model is loaded. |
| Custom metrics | `src/api.py` | `predictions_total{label,endpoint}` counter + `inference_latency_seconds{endpoint}` histogram, on top of `prometheus-fastapi-instrumentator`'s default HTTP metrics. |
| Config loader | `src/config.py` | Pydantic Settings, reads `.env` + environment. Example values in [`.env.example`](./.env.example). |
| Structured logging | `src/logging_setup.py` | JSON formatter with request-ID context, stdlib-only. |
| Training | `src/train.py` | XGBoost classifier; logs params, precision/recall/F1/ROC-AUC and the model artifact to MLflow (local `./mlruns` by default). |
| Preprocessing | `src/preprocess.py` | Normalises CICIDS-style CSVs into a train/test split. |
| Batch CSV inference | `src/inference.py` | Offline scoring using the saved model + scaler. |
| Sample data generator | `scripts/generate_sample.py` | Emits a tiny CICIDS-shaped CSV so anyone who clones the repo can train end-to-end in <60s. |
| Docker image | `Dockerfile` | `python:3.12-slim`, non-root user, `curl`-based healthcheck, runs `uvicorn src.api:app`. |
| Compose stack | `docker-compose.yml` | `api` + `prometheus` (:9090) + `grafana` (:3000). Grafana is pre-provisioned with a Prometheus datasource and an "Anomaly Detection API" dashboard. |
| Prometheus config | `ops/prometheus/prometheus.yml` | Scrapes `api:8000/metrics` every 15s. |
| Grafana provisioning | `ops/grafana/` | Datasource + dashboard provisioning, dashboard JSON under `ops/grafana/dashboards/api.json`. |
| Tests | `tests/` | Pytest suite — unit tests per `src/` module + FastAPI `TestClient` tests for every endpoint. Toy model is trained on the fly; no real dataset is downloaded. |
| CI | `.github/workflows/ci.yml` | Python 3.12 on Ubuntu — ruff, mypy (non-blocking), pytest with coverage gate at 70%. |

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET`  | `/`              | Liveness sanity check. |
| `GET`  | `/healthz`       | Returns `{status, model_loaded}` — model-independent. |
| `POST` | `/predict`       | Score a single flow (see example below). |
| `POST` | `/batch_predict` | Score up to 1000 flows per request. |
| `GET`  | `/metrics`       | Prometheus exposition format. |

### Example request

```json
POST /predict
{
  "duration": 3.4,
  "protocol": 6,
  "src_port": 51515,
  "dst_port": 443,
  "packet_count": 12,
  "byte_count": 1024
}
```

### Example response

```json
{
  "prediction": 0,
  "anomaly_score": 0.12
}
```

## Architecture

```text
                  ┌──────────────────────┐
                  │ scripts/             │
                  │ generate_sample.py   │  ← synthetic CICIDS-shaped data
                  └──────────┬───────────┘
                             │
                             ▼
┌──────────────┐   ┌─────────────────────┐    ┌──────────────┐
│ data/raw/    │→→ │ src/preprocess.py   │→→  │ data/        │
│              │   │ (clean, split)      │    │ processed/   │
└──────────────┘   └─────────────────────┘    └──────┬───────┘
                                                     │
                                                     ▼
                                           ┌─────────────────────┐
                                           │ src/train.py        │
                                           │ XGBoost + MLflow    │
                                           └──────────┬──────────┘
                                                      │
                              ┌───────────────────────┴───────────────────────┐
                              ▼                                               ▼
                    ┌───────────────────┐                          ┌──────────────────┐
                    │ models/           │                          │ ./mlruns/        │
                    │ xgb_model.joblib  │                          │ (MLflow store)   │
                    │ scaler.joblib     │                          └──────────────────┘
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────────┐          ┌───────────────┐
                    │ src/api.py (FastAPI)  │ /metrics │ Prometheus    │
                    │ /predict              │────────▶│ (ops/         │
                    │ /batch_predict        │          │ prometheus/)  │
                    │ /healthz /metrics     │          └──────┬────────┘
                    └───────────────────────┘                 │
                                                              ▼
                                                      ┌───────────────┐
                                                      │ Grafana       │
                                                      │ (provisioned) │
                                                      └───────────────┘
```

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest --cov=src
ruff check .
```

### Training on real CICIDS 2017 data (optional)

Register at <https://www.unb.ca/cic/datasets/ids-2017.html>, place CSVs in `data/raw/`, then:

```bash
python -m src.preprocess --input data/raw/your_file.csv --output data/processed/
python -m src.train --train data/processed/train.csv --test data/processed/test.csv
```

MLflow runs land in `./mlruns/` by default; point `MLFLOW_TRACKING_URI` at a remote server to change that.

## Observability

After `docker compose up`:

- **API:**        <http://localhost:8000/healthz>
- **Metrics:**    <http://localhost:8000/metrics>
- **Prometheus:** <http://localhost:9090/targets>
- **Grafana:**    <http://localhost:3000> (anonymous Viewer role is enabled; admin login is `admin` / `admin`)

The pre-provisioned "Anomaly Detection API" dashboard shows predictions-per-second by label, p95 inference latency, and HTTP request rate.

## Configuration

All runtime settings come from environment variables (see `.env.example`). The only ones you'll normally change:

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_PATH` | `models/xgb_model.joblib` | Where the API loads the classifier from. |
| `SCALER_PATH` | `models/scaler.joblib` | Where the API loads the feature scaler from. |
| `LOG_LEVEL` | `INFO` | Root log level for the JSON-formatted logger. |
| `MLFLOW_TRACKING_URI` | `file:./mlruns` | Local file store by default; point at an MLflow server to centralise runs. |
| `MLFLOW_EXPERIMENT_NAME` | `anomaly-detection` | MLflow experiment name. |

## Security

See [SECURITY.md](./SECURITY.md) for responsible disclosure. Input is validated via Pydantic; no auth / rate-limiting is configured in this reference build — put the API behind a reverse proxy or API gateway for any real deployment.

## License

[MIT](./LICENSE) © Dalton Herriman

## Acknowledgments

- CICIDS 2017 dataset — Canadian Institute for Cybersecurity
- FastAPI, scikit-learn, XGBoost, MLflow, Prometheus, Grafana
