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


#uvicorn app:app --reload