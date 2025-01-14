# README de l'application en phase de test sur le cloud
#
# ## Description
#
# Cette application a été déployée sur le cloud en phase de test pour vérifier rapidement le bon fonctionnement de l'API. Elle permets de faire des prédictions de sentiment sur des tweets en anglais.
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
# ├── model.py # Script python qui contient un sricpt simple avec des opération regex
# ├── requirements.txt # Fichier texte contenant les dépendances du projet
# ├── Dockerfile # Fichier pour construire l'image Docker
# ├── .dockerignore # Fichier pour ignorer les fichiers lors de la construction de l'image Docker
# 

