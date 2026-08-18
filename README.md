# Smart Retail & Customer Intelligence Platform

A FastAPI backend that bundles four ML services for a retail use case — face-based repeat-customer recognition, product image classification, review sentiment analysis, and a hybrid rule/ML FAQ chatbot — behind one API, with a Streamlit dashboard for viewing aggregate stats.

I built this to practice wiring several independent ML models into a single deployable service, rather than training a model in a notebook and stopping there.

## What it does

- **Face recognition** — recognizes a returning customer from a photo and logs the visit.
- **Product classification** — classifies a product image into one of 5 categories (shoes, bags, electronics, clothing, groceries).
- **Sentiment analysis** — scores customer review text as positive/negative using a TF-IDF + logistic regression pipeline.
- **FAQ chatbot** — answers common retail questions using a rule-based layer backed by an ML intent classifier for anything the rules don't cover.
- **Dashboard** — a Streamlit view of aggregate visit and sentiment stats pulled from the same API.

## Architecture

All four models are loaded once at startup (`pipeline.py`) and served through a single FastAPI app (`app/main.py`), with each capability split into its own router and service module:

```
app/
├── main.py                  # FastAPI entrypoint
├── routers/                 # vision.py, nlp.py, chatbot.py — one router per capability
├── services/                 # cv_service.py, nlp_service.py, chatbot_service.py
└── models/                   # serialized model files (.h5 / .pkl)
cv_utils.py                   # OpenCV preprocessing helpers
face_recognition_module.py    # face encoding + visit logging
pipeline.py                   # loads all models once, shared across requests
dashboard.py                  # Streamlit dashboard
notebooks/                    # training notebooks for each model
train_all_models.py           # trains and serializes all four models
tests/test_endpoints.py       # endpoint tests (pytest)
```

## API

| Method | Endpoint | What it does |
|---|---|---|
| POST | `/recognize-face` | Identify a returning customer from an image |
| POST | `/classify-product` | Classify a product image (5 categories) |
| POST | `/analyze-sentiment` | Score review text as positive/negative |
| POST | `/chatbot` | Get a response from the hybrid FAQ chatbot |
| GET | `/dashboard/stats` | Aggregate visit + sentiment stats |

## Running it

```bash
git clone https://github.com/<your-username>/smart-retail-ai.git
cd smart-retail-ai
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

python train_all_models.py          # train + serialize the 4 models
uvicorn app.main:app --reload --port 8000   # API + Swagger docs at /docs
streamlit run dashboard.py          # dashboard at localhost:8501
pytest tests/test_endpoints.py -v   # run the test suite
```

Also includes a `Dockerfile` for containerized deployment:
```bash
docker build -t smart-retail-ai .
docker run -d -p 8000:8000 smart-retail-ai
```

## A note on face recognition

Face recognition raises real privacy questions, so worth stating plainly: this is a learning project, not a production system, and the current implementation stores face encodings locally with no consent flow, encryption, or bias auditing built in. A real deployment would need explicit opt-in consent, encrypted storage of encodings (never raw images), a deletion mechanism, and regular fairness testing across demographics before it could be used on real customers.

## What I'd improve next

- Add authentication to the API endpoints (currently open)
- Replace the on-disk pickle/h5 model storage with a proper model registry
- Add CI test coverage beyond the current endpoint smoke tests
- Move face encodings to encrypted storage with a consent + deletion flow

## Tech stack

Python, FastAPI, OpenCV, scikit-learn, Streamlit, Docker, pytest
