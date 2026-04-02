# RAG Layer for SEBI-Compliant Knowledge Retrieval

A FastAPI + Streamlit Retrieval-Augmented Generation (RAG) project for answering SEBI regulation questions with evidence-backed responses from indexed documents.

## Features

- FastAPI backend for health checks and query answering
- Streamlit chat UI
- PDF ingestion and corpus chunking pipeline
- FAISS vector index using LangChain + HuggingFace embeddings
- Hybrid retrieval with reranking and fallback behavior
- Docker support + GitHub Actions Docker build workflow

## Tech Stack

- Python
- FastAPI
- Streamlit
- LangChain
- FAISS
- Hugging Face Inference API

## Project Structure

```text
app/
  api/api_server.py          # FastAPI app
  ui/ui.py                   # Streamlit frontend
  ingestion/                 # PDF parse + corpus build
  retrieval/                 # Embedding + vector retrieval
  generation/                # LLM generation and fallback logic
data/
  data_raw/                  # Input PDFs
  data_processed/            # Parsed JSON + corpus.jsonl
faiss_index/
  langchain_index/           # FAISS artifacts
.github/workflows/main.yml   # Docker CI workflow
Dockerfile
```

## Prerequisites

- Python 3.10+ (3.11 recommended)
- `pip`
- Docker (optional, for containerized run)

## Environment Variables

Create a `.env` file in the project root:

```env
HF_API_KEY=your_huggingface_api_key
HF_MODEL=google/flan-t5-large
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

Notes:
- `HF_API_KEY` is required for Hugging Face generation calls.
- Defaults are used for `HF_MODEL` and `EMBED_MODEL` if not set.

## Local Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation Pipeline

1. Put SEBI PDFs in `data/data_raw/`
2. Parse PDFs to JSON:

```bash
python -m app.ingestion.parse_pdfs
```

3. Build chunked corpus:

```bash
python -m app.ingestion.build_corpus
```

4. Build FAISS index:

```bash
python -m app.retrieval.lc_embed_index
```

## Run the API

```bash
uvicorn app.api.api_server:app --host 0.0.0.0 --port 8000
```

API endpoints:

- `GET /`
- `GET /health`
- `POST /query`

### Example Query Request

```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"What are disclosure requirements for listed entities?\",\"history\":[]}"
```

## Run the Streamlit UI

```bash
streamlit run app/ui/ui.py
```

By default, UI calls the API at `http://127.0.0.1:8000`.

## Docker

Build image:

```bash
docker build -t rag-sebi .
```

Run container:

```bash
docker run --rm -p 8000:8000 --env-file .env rag-sebi
```

Then open API docs:

- `http://127.0.0.1:8000/docs`

## Deployment

This project is easiest to deploy as a Dockerized FastAPI service.

### Option 1: Render (Docker Web Service)

1. Push this repository to GitHub.
2. In Render, create a new **Web Service** from that repository.
3. Use Docker runtime (Render will auto-detect `Dockerfile`).
4. Set environment variables in Render dashboard:
   - `HF_API_KEY`
   - `HF_MODEL` (optional)
   - `EMBED_MODEL` (optional)
5. Set port to `8000` (or let Render provide `PORT` and update start command if needed).
6. Deploy.

Note: If your retrieval index/model download is heavy, choose an instance size with enough RAM.

### Option 2: Railway (Docker Deployment)

1. Push the repo to GitHub.
2. Create a new Railway project and link the repo.
3. Railway will build using your `Dockerfile`.
4. Add environment variables:
   - `HF_API_KEY`
   - `HF_MODEL` (optional)
   - `EMBED_MODEL` (optional)
5. Expose port `8000` and deploy.

### Option 3: AWS EC2 (Docker + Nginx)

1. Launch an Ubuntu EC2 instance.
2. Install Docker:

```bash
sudo apt-get update
sudo apt-get install -y docker.io
sudo usermod -aG docker $USER
```

3. Clone your repo and create `.env`.
4. Build and run:

```bash
docker build -t rag-sebi .
docker run -d --name rag-sebi-api --restart unless-stopped -p 8000:8000 --env-file .env rag-sebi
```

5. (Recommended) Put Nginx in front of `:8000` and add HTTPS via Let's Encrypt.

### Deployment Checklist

- Ensure `data/data_processed/corpus.jsonl` and `faiss_index/langchain_index/` are present if you expect retrieval to work immediately on first boot.
- Confirm `HF_API_KEY` is set in platform environment variables.
- Verify `/health` responds with `200`.
- Verify `/query` works with a sample SEBI prompt.

## GitHub Actions

Workflow file: `.github/workflows/main.yml`

- Triggers on push (`main`/`master`) and pull requests
- Builds Docker image from project `Dockerfile`

## Troubleshooting

- If index is missing, rebuild with:
  - `python -m app.retrieval.lc_embed_index`
- If model download fails, verify network access and `EMBED_MODEL` value.
- If API cannot call Hugging Face, verify `HF_API_KEY`.

## License

Add your preferred license here.
