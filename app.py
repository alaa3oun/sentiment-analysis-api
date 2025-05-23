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

# user feedback - allowing
class Feedback(BaseModel):
    text: str
    predicted_label: str
    user_feedback: str  # e.g., 'correct' or 'incorrect'

@app.post("/feedback/")
def collect_feedback(feedback: Feedback):
    # Save to a file or database in real implementation
    print("Feedback received:", feedback)
    return {"message": "Thank you for your feedback!"}

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
# barch sentiment analysis
class BatchTextInput(BaseModel):
    texts: list[str]

@app.post("/predict/batch/")
def predict_sentiment_batch(batch_data: BatchTextInput):
    results = sentiment_model(batch_data.texts)
    return [
        {"text": text, "label": res["label"], "score": res["score"]}
        for text, res in zip(batch_data.texts, results)
    ]

@app.get("/model-info/")
def model_info():
    return {
        "model": "distilbert-base-uncased-finetuned-sst-2-english",
        "task": "sentiment-analysis",
        "description": "Binary sentiment classification (positive/negative)",
    }


#uvicorn app:app --reload