# README de l'application déployée en production sur le cloud
#
# ## Description
#
# Cette application est déployée sur le cloud et permet de faire des prédictions de sentiment sur des tweets en anglais.
# Les classes de sentiments sont les suivantes:
# - 0: négatif
# - 1: positif
#
#
# ## Structure des dossiers
# Le projet est organisé comme suit :
#
# tf_app/
# ├── __init__.py # Script python d'initialisation
# ├── main.py # Script python de l'application FastAPI pour effectuer les prédictions
# ├── model.py # Script python qui contient le code pour charger le modèle avec MLFlow et effectuer une prédiction
# ├── requirements.txt # Fichier texte contenant les dépendances du projet
# ├── Dockerfile # Fichier pour construire l'image Docker
# ├── .dockerignore # Fichier pour ignorer les fichiers lors de la construction de l'image Docker
# ├── mlflow/ # Dossier contenant les artefacts du modèle MLFLOW
#
# ## Utilisation
#
# Pour utiliser l'application, il suffit de se rendre sur la page web suivante: hedredo-sentiment-b4a8fabydkh4hge5.westeurope-01.azurewebsites.net
#
# ## Comment effectuer une prédiction
#
# Soit en effectuant une requête POST avec un script BASH:
# ```bash
# curl -X POST http://127.0.0.1:8000/predict \
#     -H "Content-Type: application/json" \
#     -d "{\"text\": \"i don't like this, this is bad\"}" \
#     -w "\n"
# ```
#
# Soit en utilisant l'applications Swagger UI en se rendant sur la page web suivante: http://hedredo-sentiment-b4a8fabydkh4hge5.westeurope-01.azurewebsites.net/docs
#
# Soit en utilisant l'interface utilisateur Streamlit (dossier ui, script python front_tf.py) dont le code se trouve sur le dépôt GitHub suivant: Hedredo/oc_p7
#
