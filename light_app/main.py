from typing import Union
from pydantic import BaseModel
from fastapi import FastAPI
from light_app.model import predict_sentiment

# Create the FastAPI instance
app = FastAPI()

# Create a Pydantic model to validate the request body
class TextData(BaseModel):
    text: str

# Define a GET endpoint for the root URL
@app.get("/")
def read_root() -> Union[dict, str]:
    return {"message": "Welcome to the Sentiment Analysis API!"}

# Define a POST endpoint for the /predict URL
@app.post("/predict")
def predict(data: TextData) -> dict:
    text = data.text
    sentiment = predict_sentiment(text)
    return {"text": text, "sentiment": sentiment}