import streamlit as st
import requests
import time
import logging
from dotenv import load_dotenv
import os
from enum import Enum

# pip install opencensus opencensus-ext-logging opencensus-ext-azure
from opencensus.ext.azure.log_exporter import AzureLogHandler

# Configure la classe Step pour le changement dynamique des boutons
class Step(Enum):
    INITIAL = 1
    PREDICT = 2
    FEEDBACK = 3

    @classmethod
    def next_value(cls, current_value):
        next_value = (current_value.value % len(cls)) + 1
        return Step(next_value)

# Récupère les variables de session
load_dotenv()
api_url = os.getenv("API_URL")
instrumentation_key = os.getenv("INSTRUMENTATION_KEY")

# Configure le journal pour envoyer des traces à Application Insights
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(AzureLogHandler(connection_string=f"InstrumentationKey={instrumentation_key}"))

# Streamlit : Configure l'interface utilisateur affiché par défaut
st.title("Live Tweet Sentiment Analysis")


if "step" not in st.session_state:
    st.session_state.step = Step.INITIAL
if "prediction_data" not in st.session_state:
    st.session_state.prediction_data = {}
if "text" not in st.session_state:
    st.session_state.text = ""

def handle_on_click():
    if st.session_state.step == Step.FEEDBACK:
        st.session_state.step = Step.INITIAL
    else:
        st.session_state.step = Step.next_value(st.session_state.step)

# Ajout d'un bouton pour effectuer l'action de prédiction du sentiment
if st.session_state.step.value == Step.INITIAL.value:
    st.session_state.text = ""
    text = st.text_input("Enter a tweet please:")
    st.session_state.text = text
    st.button(
        "Predict the sentiment",
        on_click=handle_on_click,
    )

elif st.session_state.step.value == Step.PREDICT.value:
    url = f"{api_url}/predict"
    data = {"text": st.session_state.text}
    response = requests.post(url, json=data)

    if response.status_code != 200:
        st.error(
            "Error occurring the connection. We will fix the problem ASAP. Please try again later."
        )
        logger.error(
            "Error occurring the API call.", extra={
                "custom_dimensions": {
                    "status_code": response.status_code,
                    "request": "predict",
                }
            })
    else:
        prediction = response.json()["sentiment"]
    
    # Display the result
    st.write(f"Tweet entered: {st.session_state.text}")
    st.write("Model computing the sentiment of the tweet...")
    time.sleep(2)
    # Display the result
    match prediction:
        case "positive":
            st.success(f"Sentiment predicted: {prediction}")
        case "negative":
            st.error(f"Sentiment predicted: {prediction}")
        case _:
            st.write(f"Sentiment predicted: {prediction}")
        
    # Stocker les données de prédiction dans la session
    st.session_state.prediction_data = {
        "text": st.session_state.text,
        "prediction": prediction,
    }

    st.write("Was the prediction correct?")
    # Affiche les boutons côté à côte
    leftcol, midcol, rightcol = st.columns(3)
    with leftcol:
        st.button("Yes", on_click=handle_on_click)
        logger.info("Good prediction", extra={
            "custom_dimensions": {
                "text": st.session_state.prediction_data["text"],
                "prediction": st.session_state.prediction_data["prediction"],
            }
        })
    
    with midcol:
        st.button("No", on_click=handle_on_click)
        logger.warning("Bad prediction", extra={
            "custom_dimensions": {
                "text": st.session_state.prediction_data["text"],
                "prediction": st.session_state.prediction_data["prediction"],
            }
        })

    with rightcol:
        st.button("I don't know", on_click=handle_on_click)

else:
    st.write("Thank you for your feedback!")
    st.button("Try another tweet !", on_click=handle_on_click)