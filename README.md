## VietCMS — Vietnamese Content Moderation Service
  An AI-powered, microservice-based content moderation platform built for Vietnamese-language content.
  Detect toxicity, hate speech, harassment, spam, PII, and more — in text, images, and audio.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Environment Configuration](#environment-configuration)
- [Downloading Data & Models](#downloading-data--models)
- [Deployment](#deployment)
- [Services Reference](#services-reference)
- [Client SDKs](#client-sdks)
- [Monitoring](#monitoring)
- [Makefile Commands](#makefile-commands)
- [Exposing to the Internet](#exposing-to-the-internet)
- [Project Structure](#project-structure)
- [References](#references)

---

## Overview

**VietCMS** (Vietnamese Content Moderation Service) is a production-ready platform that automatically moderates user-generated content in Vietnamese. It provides:

- **Multi-modal moderation** — Text, image (NSFW detection + OCR), and audio (speech-to-text) analysis.
- **AI-powered NLP pipeline** — Built on [PhoBERT](https://github.com/VinAIResearch/PhoBERT) (Vietnamese BERT) with multi-task classification for toxicity, hate speech, harassment, threats, sexual content, spam, and PII detection.
- **Rule-based + ML ensemble** — Combines deep-learning inference with rule-based checks, toxic word dictionaries, and variant detection for robust moderation.
- **Async job processing** — RabbitMQ-based message queue for high-throughput, scalable content processing (1,000+ jobs/minute).
- **Webhook notifications** — Real-time moderation results delivered to client applications via webhooks with HMAC signature verification.
- **Admin dashboard** — A modern web UI for managing clients, reviewing moderation results, and monitoring system health.
- **Client SDKs** — Ready-to-use SDKs for Node.js and Python.

---

## Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌────────────────────┐
│  Client App  │────▶│   Nginx Proxy    │────▶│  Moderation API    │
│  (SDK/REST)  │     │   (port 80)      │     │  (FastAPI)         │
└──────────────┘     └──────────────────┘     └────────┬───────────┘
                              │                        │
                              │                        ▼
                     ┌────────▼───────┐       ┌────────────────────┐
                     │   Admin UI     │       │     RabbitMQ       │
                     │   (Vite+React) │       │  (Message Broker)  │
                     └────────────────┘       └────────┬───────────┘
                                                       │
                                              ┌────────▼───────────┐
                                              │ Moderation Worker  │
                                              │ (PhoBERT + Rules)  │
                                              └────────┬───────────┘
                                                       │
                              ┌─────────────────┬──────┴──────┐
                              ▼                 ▼             ▼
                     ┌────────────────┐ ┌─────────────┐ ┌───────────┐
                     │   PostgreSQL   │ │    Redis     │ │  Webhook  │
                     │   (Database)   │ │   (Cache)    │ │Dispatcher │
                     └────────────────┘ └─────────────┘ └───────────┘
```

---

## Tech Stack

| Layer               | Technology                                                    |
| ------------------- | ------------------------------------------------------------- |
| **API**             | Python 3.11, FastAPI, SQLAlchemy, Alembic                     |
| **NLP / AI**        | PhoBERT (vinai/phobert-base-v2), PyTorch, ONNX Runtime, Underthesea |
| **Image**           | Falconsai NSFW detection, OpenRouter Vision OCR               |
| **Message Queue**   | RabbitMQ 3.12                                                 |
| **Database**        | PostgreSQL 15                                                 |
| **Cache**           | Redis 7                                                       |
| **Admin UI**        | Vite, React, Tailwind CSS                                     |
| **Reverse Proxy**   | Nginx, Cloudflared                                                       |
| **Monitoring**      | Prometheus, Grafana                                           |
| **Containerization**| Docker, Docker Compose                                        |

---

## Prerequisites

Before you begin, ensure you have the following installed:

- **Docker Desktop** ≥ 4.x (includes Docker Compose v2)
- **Git**
- **PowerShell** (Windows) or **Bash** (Linux/macOS) for helper scripts
- At least **8 GB RAM** allocated to Docker (the moderation worker needs 2–4 GB)
- At least **10 GB free disk space** for Docker images and model files

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/khangnguyenthe18/VietCMS.git
cd VietCMS
```

### 2. Configure Environment Variables

```bash
cp env-example.txt .env
```

Edit `.env` to set your own secrets (see [Environment Configuration](#environment-configuration)).

### 3. Build & Start All Services

```bash
# Build Docker 
docker-compose build

# Start all services in detached mode
docker-compose up -d
```

### 4. Run Database Migrations

```bash
docker-compose exec moderation-api alembic upgrade head
```

### 5. Create the Default Admin User

```bash
# Windows (PowerShell)
docker-compose exec postgres psql -U vietcms -d vietcms_moderation -f /scripts/create-admin.sql
```

Default admin credentials: `admin@vietcms.com` / `admin123`

### 6. (Optional) Set Up the Demo Client

```bash
# Windows
powershell -ExecutionPolicy Bypass -File ./scripts/setup-demo-client.ps1

# Linux / macOS
bash ./scripts/setup-demo-client.sh
```

### 7. Access the Services

| Service              | URL                              |
| -------------------- | -------------------------------- |
| **Admin Dashboard**  | http://localhost:80              |
| **API Health Check** | http://localhost/api/v1/health   |
| **Demo Website**     | http://localhost:5000            |
| **RabbitMQ Console** | http://localhost:15672           |

---

## Environment Configuration

Copy `env-example.txt` to `.env` in the project root. Key variables:

| Variable                | Description                                  | Default                      |
| ----------------------- | -------------------------------------------- | ---------------------------- |
| `POSTGRES_DB`           | Database name                                | `vietcms_moderation`         |
| `POSTGRES_USER`         | Database user                                | `vietcms`                    |
| `POSTGRES_PASSWORD`     | Database password                            | `vietcms_password_123`       |
| `RABBITMQ_DEFAULT_USER` | RabbitMQ user                                | `admin`                      |
| `RABBITMQ_DEFAULT_PASS` | RabbitMQ password                            | `rabbitmq_password_456`      |
| `API_SECRET_KEY`        | Secret key for API (min 32 chars)            | *(change in production!)*    |
| `API_RATE_LIMIT`        | Max requests per minute                      | `1000`                       |
| `WORKER_CONCURRENCY`    | Parallel jobs per worker replica              | `20`                         |
| `WORKER_REPLICAS`       | Number of worker containers                  | `2`                          |
| `MODEL_PATH`            | Path to PhoBERT model inside the container   | `/app/models/phobert-onnx`   |
| `MODEL_DEVICE`          | Inference device (`cpu` or `cuda`)           | `cpu`                        |
| `CONFIDENCE_THRESHOLD`  | Min confidence to flag content (0.0–1.0)     | `0.7`                        |
| `VITE_API_BASE_URL`     | API base URL for Admin UI                    | `http://localhost/api/v1`    |
| `LOG_LEVEL`             | Logging level                                | `INFO`                       |

> ⚠️ **Important:** Always change `POSTGRES_PASSWORD`, `RABBITMQ_DEFAULT_PASS`, `API_SECRET_KEY`, and `JWT_SECRET_KEY` for production deployments.

---

## Downloading Data & Models

### AI Models (PhoBERT)

The moderation worker uses **PhoBERT** (`vinai/phobert-base-v2`) for Vietnamese NLP. Models are downloaded automatically on first run, but you can pre-download them:

**Option A — Automated setup script (recommended):**

```bash
# Inside the moderation-worker directory
cd services/moderation-worker

# Windows
powershell -ExecutionPolicy Bypass -File setup.ps1

# Linux / macOS
bash setup.sh
```

This script will:
1. Install Python dependencies
2. Download PhoBERT from HuggingFace to `services/moderation-worker/models/phobert-base-v2/`
3. Download Vietnamese moderation datasets
4. Create quick-start training & testing scripts

**Option B — Manual download via Python:**

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
model = AutoModel.from_pretrained("vinai/phobert-base-v2")

tokenizer.save_pretrained("./services/moderation-worker/models/phobert-base-v2")
model.save_pretrained("./services/moderation-worker/models/phobert-base-v2")
```

**Option C — Docker volume (auto-download):**

When running via Docker Compose, the worker container maps `./services/moderation-worker/models` to `/app/models`. The HuggingFace cache (`HF_HOME=/app/models`) ensures models persist across container restarts.

### Image Moderation Model

The NSFW image detection model (`Falconsai/nsfw_image_detection`) is **downloaded automatically** by HuggingFace Transformers on first use.

### Training Datasets

Vietnamese moderation datasets can be downloaded using the included script:

```bash
cd services/moderation-worker
python data/download_datasets.py --data-dir ./datasets --dataset all
```

Available datasets:

| Dataset        | Description                               | Source                                              |
| -------------- | -------------------------------------     | --------------------------------------------------- |
| **ViHSD**      | Vietnamese Hate Speech Detection          | https://github.com/ongocthanhvan/ViHSD              |
| **ViHOS**      | Vietnamese Hate & Offensive Spans         | https://github.com/tarudesu/ViHOS                   |
| **UIT-ViCTSD** | Vietnamese Constructive/Toxic Speech      | Contact `nlp@uit.edu.vn`                            |
| **UIT-VSMEC**  | Vietnamese Constructive/Multi-emotion     | Contact `nlp@uit.edu.vn`                            |
| **UIT-VSFC**   | Vietnamese Constructive/Customer Feedback | Contact `nlp@uit.edu.vn`                            |


### Training a Custom Model

After downloading datasets, you can train your own model:

```bash
cd services/moderation-worker

# Quick training with recommended settings
bash quick_train.sh

# Or custom training
python training/train_full.py \
    --data-dir ./datasets \
    --output-dir ./checkpoints \
    --batch-size 16 \
    --epochs 10 \
    --learning-rate 2e-5 \
    --use-focal-loss
```

---

## Deployment

### Development (Local)

```bash
docker-compose up -d          # Start all services
docker-compose logs -f        # Follow logs
docker-compose down           # Stop all services
```

### Production

For production deployments, ensure you:

1. **Change all default passwords** in `.env` (database, RabbitMQ, API keys, JWT secrets).
2. **Set `ENVIRONMENT=production`** in `.env`.
3. **Configure CORS** — Set `API_CORS_ORIGINS` to your allowed domains instead of `*`.
4. **Scale workers** — Increase `WORKER_REPLICAS` based on expected load:
   ```
   # Example: 4 replicas × 20 concurrency = 80 parallel jobs
   WORKER_REPLICAS=4
   ```
5. **Enable HTTPS** — Configure SSL certificates in the Nginx reverse proxy.
6. **Enable monitoring** — Deploy Prometheus + Grafana (see [Monitoring](#monitoring)).

```bash
# Production build & deploy
docker-compose build --no-cache
docker-compose up -d --scale moderation-worker=4
```

### Resource Requirements

| Service             | CPU     | Memory  |
| ------------------- | ------- | ------- |
| Moderation API      | 0.25–1  | 256M–1G |
| Moderation Worker   | 1–2     | 2G–4G   |
| Webhook Dispatcher  | 0.1–0.5 | 128M–512M |
| PostgreSQL          | 0.5     | 512M    |
| RabbitMQ            | 0.25    | 256M    |
| Redis               | 0.1     | 512M    |
| Nginx               | 0.1     | 64M     |
| Admin UI            | 0.1     | 64M     |

---

## Services Reference

### Moderation API (`services/moderation-api`)
- **Framework:** FastAPI + Uvicorn
- **Port:** 8000 (internal), exposed via Nginx at `/api`
- **Features:** Client registration, content submission, moderation results, rate limiting, JWT authentication
- **Health check:** `GET /api/v1/health`

### Moderation Worker (`services/moderation-worker`)
- **Purpose:** Async content analysis using AI models
- **Pipeline:** Text preprocessing → PhoBERT inference → Rule-based checks → Ensemble scoring
- **Modalities:** Text, Image (NSFW + OCR), Audio (Whisper STT)
- **Labels:** `toxicity`, `hate`, `harassment`, `threat`, `sexual`, `spam`, `pii`
- **Actions:** `allowed`, `review`, `reject`

### Webhook Dispatcher (`services/webhook-dispatcher`)
- **Purpose:** Delivers moderation results to client webhook URLs
- **Features:** HMAC signature verification, retry logic (configurable max retries)

### Admin UI (`services/admin-ui`)
- **Framework:** Vite + React + Tailwind CSS
- **Access:** http://localhost80 (via Nginx)
- **Features:** Dashboard, client management, moderation review queue

### Demo Client Website (`demo-client-website`)
- **Port:** 5000
- **Purpose:** Example integration showing how to submit content and receive webhook results

---

## Client SDKs

Ready-to-use SDKs are provided in the `client-sdk/` directory:

### Node.js

```bash
cd client-sdk/nodejs
npm install
```

### Python

```bash
cd client-sdk/python
pip install -r requirements.txt
```

---

## Monitoring

Optional Prometheus + Grafana monitoring is available in the `monitoring/` directory.

| Service        | URL                        | Credentials          |
| -------------- | -------------------------- | -------------------- |
| **Prometheus** | http://localhost:9090       | —                    |
| **Grafana**    | http://localhost:3001       | `admin` / `admin123` |

The Moderation API exposes Prometheus metrics via `prometheus-fastapi-instrumentator`.

---

## Makefile Commands

| Command              | Description                         |
| -------------------- | ----------------------------------- |
| `make build`         | Build all Docker images             |
| `make up`            | Start all services (detached)       |
| `make down`          | Stop all services                   |
| `make logs`          | Follow logs for all services        |
| `make migrate`       | Run Alembic database migrations     |
| `make test`          | Run test suite                      |
| `make clean`         | Stop services and remove volumes    |
| `make create-admin`  | Create default admin user           |
| `make setup-demo`    | Set up the demo client application  |

---

## Exposing to the Internet

### Ngrok (for Admin UI / API)

```bash
ngrok http 80 --config ngrok.yml
```

### Cloudflared (for Demo Website)

```bash
cloudflared tunnel --url http://localhost:5000
```

## Project Structure

```
VietCMS/
├── docker-compose.yml          # Orchestrates all services
├── .env                        # Environment configuration (from env-example.txt)
├── env-example.txt             # Example environment variables
├── Makefile                    # Development shortcuts
├── ngrok.yml                   # Ngrok tunnel configuration
│
├── nginx/                      # Reverse proxy config
│   └── default.conf
│
├── services/
│   ├── moderation-api/         # REST API (FastAPI)
│   │   ├── app/                # Application source code
│   │   ├── alembic/            # Database migrations
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── moderation-worker/      # AI moderation worker
│   │   ├── nlp/                # NLP pipeline (PhoBERT, rules, preprocessing)
│   │   ├── models/             # Model architectures & cached weights
│   │   ├── data/               # Dataset loaders & downloaders
│   │   ├── training/           # Training scripts
│   │   ├── image/              # Image moderation module
│   │   ├── audio/              # Audio moderation module
│   │   ├── worker.py           # Main worker entry point
│   │   ├── config.py           # Worker configuration
│   │   ├── setup.sh / setup.ps1# Automated setup scripts
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── webhook-dispatcher/     # Webhook delivery service
│   │   └── Dockerfile
│   │
│   └── admin-ui/               # Admin dashboard (Vite + React)
│       ├── src/
│       ├── Dockerfile
│       └── package.json
│
├── demo-client-website/        # Demo integration example
│   ├── src/                    # Frontend (Vite + React)
│   ├── backend/                # Backend server
│   └── Dockerfile
│
├── client-sdk/                 # Client integration SDKs
│   ├── nodejs/
│   └── python/
│
├── scripts/                    # Helper scripts
│   ├── init-db.sql             # Database initialization
│   ├── create-admin.sql        # Admin user creation
│   ├── register-demo.py        # Demo client registration
│   └── setup-demo-client.*     # Demo setup automation
│
├── monitoring/                 # Observability stack
│   ├── prometheus.yml
│   └── grafana/
│
└── tests/                      # Integration tests
```

---

## References
1. Nguyen Dat Quoc, Nguyen Anh Tu, "PhoBERT: Pre-trained language models
for Vietnamese", Findings of EMNLP 2020.
2. Nguyen Huy Thang et al., "ViHSD: A Vietnamese Hate Speech Detection
Dataset", NAACL 2022.
3. Nguyen Luong Viet et al., "UIT-ViCTSD: A Vietnamese Constructive and Toxic
Speech Detection Dataset", PACLIC 2021.
4. Devlin, J., et al. "BERT: Pre-training of Deep Bidirectional Transformers for
Language Understanding." NAACL 2019.
5. Vaswani, A., et al. "Attention Is All You Need." NeurIPS 2017.
6. Wolf, T., et al. "Transformers: State-of-the-Art Natural Language Processing."
EMNLP 2020.
7. Newman, S. "Building Microservices." O'Reilly Media, 2021.
8. Lin, T.Y., et al. "Focal Loss for Dense Object Detection." ICCV 2017.
9. FastAPI Documentation, https://fastapi.tiangolo.com/
10. PyTorch Documentation, https://pytorch.org/docs/
11. Hugging Face Transformers, https://huggingface.co/docs/transformers/
12. RabbitMQ Documentation, https://www.rabbitmq.com/documentation.html
13. Docker Documentation, https://docs.docker.com/
14. PostgreSQL Documentation, https://www.postgresql.org/docs/
