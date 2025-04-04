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

# comparing sentiments
class TextPair(BaseModel):
    text1: str
    text2: str

@app.post("/compare/")
def compare_sentiments(data: TextPair):
    results = sentiment_model([data.text1, data.text2])
    return {
        "text1": {"label": results[0]["label"], "score": results[0]["score"]},
        "text2": {"label": results[1]["label"], "score": results[1]["score"]}
    }


#uvicorn app:app --reload