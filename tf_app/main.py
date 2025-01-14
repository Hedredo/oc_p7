from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from model import set_model, SpacyTokenizer, custom_standardization_punct

# Load the model & tokenizer
model = set_model("./mlflow/15c36d2a681f47219f7b77774ce43838/artifacts/model") # Assign the function model_path argument with the another model if needed
tokenizer = SpacyTokenizer("./mlflow/5ce076a511c44429b2bcfbf7a02d0779/artifacts/spacy_en_core_web_sm")

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
    probability = model.predict(tokenizer.tokenize([text]))
    sentiment = "positive" if probability[0][0] > 0.5 \
        else ("negative" if probability[0][0] < 0.5 else "neutral")
    return {"text": text, "sentiment": sentiment} # "probability": float(prediction[0][0])