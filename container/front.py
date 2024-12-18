import streamlit as st
import requests
import time
import os

port = os.environ.get("PORT", "8000")

st.title("Sentiment Analysis Service")
text = st.text_input("Enter text")

if st.button("Predict the sentiment"):
    url = f"http://localhost:{port}/predict"
    data = {"text": text}
    response = requests.post(url, json=data)

    if response.status_code != 200:
        st.error(
            "Error occurred during prediction. Please verify your input as a text and try again."
        )

    else:
        prediction = response.json()["sentiment"]
        # Display the result
        st.write(f"Text entered: {text}")
        st.write("Model computing the sentiment of the text...")
        time.sleep(1)
        # Display the result
        match prediction:
            case "positive":
                st.success(f"Sentiment predicted: {prediction}")
            case "negative":
                st.error(f"Sentiment predicted: {prediction}")
            case _:
                st.write(f"Sentiment predicted: {prediction}")