from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
from model import set_model, custom_standardization

# Load the model
model = set_model() # Assign the function model_path argument with the another model if needed

# Create the FastAPI instance
app = FastAPI()

# Create the TextData class with Pydantic BaseModel for POST request
class TextData(BaseModel):
    text: str

# Define the GET request
@app.get("/")
def read_root() -> Union[dict, str]:
    return {"message": "Welcome to the Sentiment Analysis API!"}

@app.post("/predict")
def predict(data: TextData) -> dict:
    text = data.text
    # Make the prediction
    probability = model.predict(tf.constant([text]))
    sentiment = "positive" if probability[0][0] > 0.5 \
        else ("negative" if probability[0][0] < 0.5 else "neutral")
    return {"text": text, "sentiment": sentiment} # "probability": float(prediction[0][0])