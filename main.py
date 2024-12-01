from typing import Union, Optional
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.tensorflow
import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="custom_text_func", name="custom_standardization")
def custom_standardization(tensor):
    tensor = tf.strings.lower(tensor)  # lowercase
    tensor = tf.strings.regex_replace(tensor, r"@\w+", " ")  # strip mentions
    tensor = tf.strings.regex_replace(tensor, r"http\S+|www\S+", " ")  # strip urls
    tensor = tf.strings.regex_replace(tensor, r"[^\w\s\d]", " ")  # strip punctuation
    tensor = tf.strings.regex_replace(tensor, r"\s{2,}", " ")  # strip multiple spaces
    return tf.strings.strip(tensor)  # strip leading and trailing spaces

# Load the model
model_path = "./mlflow/689416981458083287/b5efda2cee954ff1a88923f357bc0525/artifacts/model"
model = mlflow.tensorflow.load_model(model_path) # Load the model with the MLflow Keras API

# Create the FastAPI instance
app = FastAPI()

# Create the TextData class with Pydantic BaseModel for POST request
class TextData(BaseModel):
    text: str


# Define the GET request
@app.get("/")
def read_root() -> Union[str, dict]:
    return {"Hello": "World"}

@app.post("/predict")
def predict(data: TextData) -> dict:
    # Make the prediction
    prediction = model.predict(tf.constant([data.text]))
    label = "positive" if prediction[0][0] > 0.5 else "negative"
    return {"prediction": label, "probability": float(prediction[0][0])}