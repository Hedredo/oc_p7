from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
from tf_app.model import set_model

@tf.keras.utils.register_keras_serializable(package="custom_text_func", name="custom_standardization")
def custom_standardization(tensor):
    tensor = tf.strings.lower(tensor)  # lowercase
    tensor = tf.strings.regex_replace(tensor, r"@\w+", " ")  # strip mentions
    tensor = tf.strings.regex_replace(tensor, r"http\S+|www\S+", " ")  # strip urls
    tensor = tf.strings.regex_replace(tensor, r"[^\w\s\d]", " ")  # strip punctuation
    tensor = tf.strings.regex_replace(tensor, r"\s{2,}", " ")  # strip multiple spaces
    return tf.strings.strip(tensor)  # strip leading and trailing spaces

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