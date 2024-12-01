import streamlit as st
import requests

# Titre de l'application
st.title('Interface utilisateur de l\'API')

# Saisie du texte à prédire
text = st.text_input('Entrez le texte à prédire')

# Bouton pour effectuer la prédiction
if st.button('Prédire'):
    # URL de votre API locale
    api_url = 'http://127.0.0.1:8000/predict'

    # Données à envoyer à l'API
    data = {'text': text}

    # Envoi de la requête à l'API
    response = requests.post(api_url, json=data)

    # Affichage de la prédiction
    if response.status_code == 200:
        prediction = response.json()['prediction']
        st.success(f'Prédiction : {prediction}, Probabilité : {response.json()["probability"]:.4f}')
    else:
        st.error('Une erreur s\'est produite lors de la prédiction')
        # Pour lancer l'application Streamlit, utilisez la commande suivante dans votre terminal :
        # streamlit run /home/hedredo/github/dagshub_p7/st_app.py