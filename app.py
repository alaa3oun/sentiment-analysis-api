from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Initialize FastAPI app
app = FastAPI()

# Load the sentiment analysis model
sentiment_model = pipeline("sentiment-analysis")

# Define request body structure
class TextInput(BaseModel):
    text: str

@app.post("/predict/")
def predict_sentiment(input_data: TextInput):
    result = sentiment_model(input_data.text)
    return {"label": result[0]['label'], "score": result[0]['score']}


@app.get("/model-info/")
def model_info():
    return {"model": "distilbert-base-uncased-finetuned-sst-2-english"}

def get_confidence_level(score):
    if score >= 0.9:
        return "High"
    elif score >= 0.7:
        return "Medium"
    else:
        return "Low"

@app.post("/predict/")
def predict_sentiment(input_data: TextInput):
    result = sentiment_model(input_data.text)
    confidence = get_confidence_level(result[0]['score'])
    return {"label": result[0]['label'], "score": result[0]['score'], "confidence": confidence}

#uvicorn app:app --reload