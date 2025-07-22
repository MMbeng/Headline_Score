from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import logging
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

app = FastAPI()

try:
    logging.info("Loading model---...")
    CLASSIFIER = joblib.load("svm.joblib")
    EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    logging.info("Models loaded successfully.")
except Exception as e:
    logging.critical("Failed to load models: %s", e)
    raise RuntimeError("Model loading failed.") from e

class HeadlineRequest(BaseModel):
    headlines: List[str]

LABELS = {
    0: "Pessimistic",
    1: "Neutral",
    2: "Optimistic"
}

@app.get("/status")
def get_status():
    """
    Returns service health status.
    """
    logging.info("GET /status request received.")
    return {"status": "OK"}

@app.post("/score_headlines")
def score_headlines(request: HeadlineRequest):
    """
    Accepts a list of headlines and returns corresponding sentiment labels.
    """
    headlines = request.headlines
    logging.info("POST /score_headlines request received. Number of headlines: %d", len(headlines))

    if not headlines:
        logging.warning("Received empty headline list.")
        raise HTTPException(status_code=400, detail="No headlines provided.")

    try:
        embeddings = EMBEDDER.encode(headlines)
        predictions = CLASSIFIER.predict(embeddings)
        labels = [LABELS.get(label, "Unknown") for label in predictions]
        return {"labels": labels}
    except Exception as exc:
        logging.error("Error during prediction: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error.")
