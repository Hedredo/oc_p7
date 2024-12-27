import streamlit as st
import requests
import time
import logging
from dotenv import load_dotenv
import os
from enum import Enum
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

# Création de la fonction logger avec le résultat mis en cache pour éviter d'ajouter plusieurs fois le même handler
@st.cache_resource
def get_logger():
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        logger.addHandler(AzureLogHandler(connection_string=f"InstrumentationKey={instrumentation_key}"))
    return logger

# Récupère les variables de session
load_dotenv()
api_url = os.getenv("API_URL")
instrumentation_key = os.getenv("INSTRUMENTATION_KEY")

# Récupère ou initialise le logger
logger = get_logger()

# Streamlit : Configure l'interface utilisateur affiché par défaut
st.title("Live Tweet Sentiment Analysis")

# Initialise les session state lors du premier run de script
if "step" not in st.session_state:
    st.session_state.step = Step.INITIAL
if "prediction_data" not in st.session_state:
    st.session_state.prediction_data = {}
if "text" not in st.session_state:
    st.session_state.text = ""

# Fonction handle_on_click : gère les actions de boutons en fonction des états de session
def handle_on_click(message=None):
    match message:
        case "Yes":
            logger.info("Good prediction", extra={
                "custom_dimensions": {
                    "text": st.session_state.prediction_data["text"],
                    "prediction": st.session_state.prediction_data["prediction"],
                }
            })
        case "No":
            logger.warning("Bad prediction", extra={
                "custom_dimensions": {
                    "text": st.session_state.prediction_data["text"],
                    "prediction": st.session_state.prediction_data["prediction"],
                }
            })
        case _:
            pass
        
    if st.session_state.step == Step.FEEDBACK:
        st.session_state.step = Step.INITIAL
    else:
        st.session_state.step = Step.next_value(st.session_state.step)

# session_state INITIAL
if st.session_state.step.value == Step.INITIAL.value:
    st.session_state.text = ""
    text = st.text_input("Enter a tweet please:")
    st.session_state.text = text
    st.button(
        "Predict the sentiment",
        on_click=handle_on_click,
    )

# session_state PREDICT
elif st.session_state.step.value == Step.PREDICT.value:
    url = f"{api_url}/predict"
    data = {"text": st.session_state.text}
    response = requests.post(url, json=data)

    if response.status_code != 200:
        st.error(
            "Unable to establish the connection. We will fix the problem ASAP. Please reload the page to try again later."
        )
        logger.error(
            "Error : API call", extra={
                "custom_dimensions": {
                    "status_code": response.status_code,
                    "request": "predict",
                }
            })
        st.stop()
    else:
        prediction = response.json()["sentiment"]
    
    # Affiche du texte
    st.write(f"Tweet entered: {st.session_state.text}")
    st.write("Model computing the sentiment of the tweet...")
    time.sleep(2) # Simulation d'un temps de calcul

    # Affiche le résultat de la prédiction
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

    # Affiche la question de feedback
    st.write("Was the prediction correct?")
    # Affiche les boutons de feedback côté à côte
    leftcol, midcol, rightcol = st.columns(3)
    # Affiche le bouton de gauche
    with leftcol:
        st.button("Yes", on_click=handle_on_click, args=("Yes",), icon="\U0001F44D", use_container_width=True)
    # Affiche le bouton du milieu
    with midcol:
        st.button("No", on_click=handle_on_click, args=("No",), icon="\U0001F44E", use_container_width=True)
    # Affiche le bouton de droite
    with rightcol:
        st.button("I don't know", on_click=handle_on_click, use_container_width=True)

# session_state FEEDBACK
else:
    st.write("Thank you for your feedback!")
    st.button("Try another tweet !", on_click=handle_on_click)