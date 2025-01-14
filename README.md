# oc_p7 : Prédiction de sentiment de tweets via un modèle de Deep Learning avec une approche orientée MLOps

## Introduction
Ce projet a pour but de déployer une API de prédiction de sentiment de tweets. L'API est basée sur un modèle de Deep Learning entraîné sur un jeu de données de tweets annotés. L'API est déployée sur une instance AWS EC2 et est accessible via une URL. 
Ce projet a été réalisé en plusieurs étapes pour démontrer les avantages d'une démarche orientée MLOps :
1. Téléchargement des données open-source depuis Kaggle : https://www.kaggle.com/kazanova/sentiment140. Le fichier étant volumineux, il n'est pas inclus dans le dépôt.
2. Création de notebooks et scripts python pour l'analyse exploratoire des données, le prétraitement des données, l'entraînement du modèle et l'évaluation du modèle.
3. Une base de données MLflow a été utilisée pour suivre les expériences et les métriques de chaque modèle testé.
4. Les dossiers MLFlow contenant les artifacts des expériences servant au déploiement de l'API.
5. Création d'une API avec FastAPI pour la prédiction de sentiment de tweets.
6. Création d'une interface utilisateur pour tester l'API avec Streamlit dans le dossier ui pour chaque application depuis le cloud et en local.
7. Un workflow GitHub Actions a été créé pour automatiser le déploiement continue de l'API avec conteneurisation avec Docker à chaque mise à jour des dossiers tf_app et light_app.
8. Un dossier tests contenant des scripts pour réaliser des tests unitaires avant chaque déploiement du nouveau conteneur de l'API.
9. Un script avec une commande CURL pour tester l'API.


## Déploiement du projet sur le cloud
L'API de production est déployée sur une instance AWS EC2. Pour tester l'API de production, il suffit de se rendre sur l'URL suivante : hedredo-sentiment-b4a8fabydkh4hge5.westeurope-01.azurewebsites.net

## Notes concernant les librairies utilisées pour le projet
Il a été installé la version tensorflow 2.10 compatible avec DirectML pour pouvoir utiliser le GPU lors de l'entraînement.
Si vous utilisez une autre carte graphique de type NVIDIA, il est recommandé d'installer la version tensorflow gpu et les librairies CUDA et cuDNN compatibles avec votre carte graphique.

## Structure des dossiers
Le projet est organisé comme suit :

oc_p7/ 
├── .github/ # Workflows GitHub Actions pour déploiement continu du conteneur de l'API de test (light_app) et de production (tf_app) 
├── mlflow/ # Artifacts du modèle entrainé et déployé en production
├── notebooks/ # Contient les différents notebooks pour l'analyse exploratoire des données, le prétraitement des données, l'entraînement du modèle et l'évaluation du modèle
|   ├── exploration.ipynb # Notebook pour l'analyse exploratoire des données et la préparation des données de test et d'entraînement
|   ├── ml_model.ipynb # Notebook de modélisation pour l'entraînement du modèle de Machine Learning
|   ├── nn_model.ipynb # Notebook de modélisation pour l'entraînement du modèle de Deep Learning
|   ├── adv_model.ipynb # Notebook pour le fine-tuning du modèle d'un modèle BERT
|   ├── register_serializable.ipynb # Notebook pour automatiser l'enregistrement des fonctions sérialisables pour KERAS
|   ├── func.py # Script python qui contient les fonctions sérialisables pour KERAS identique au notebook register_serializable.ipynb
|   ├── dl.py # Script python pour les fonctions relatives aux modèles de Deep Learning
|   ├── ml.py # Script python pour les fonctions relatives aux modèles de Machine Learning
|   ├── utils.py # Script python pour les fonctions d'utilité générale
|   ├── .env # Fichier contenant les variables d'environnement pour tester le modèle à déployer
|   ├── dataset/ # dossier intermédiaire pour stocker des .csv pour l'enregistrement des datasets sur mlflow
├── ui/ # Contient les interfaces utilisateurs pour tester l'API avec Streamlit
|   ├── front_light.py/ # Script python contenant l'interface utilisateur pour tester l'API de test (light_app) testée sur le cloud avec Streamlit
|   ├── front_tf.py/ # Script python contenant l'interface utilisateur pour tester l'API de production (tf_app) déployée sur le cloud avec Streamlit
|   ├── front_local_test_tf.py # Script python contenant l'interface utilisateur pour tester l'API de production (tf_app) en local avec Streamlit
├── tf_app/ # Dossier contenant l'API de prédiction de sentiment de tweets avec FastAPI et déployé sur AWS EC2
├── light_app/ # Dossier contenant une application de test de l'API avec FastAPI et déployé sur AWS EC2 
├── tests/ # Contient des scripts pour réaliser des tests unitaires avant chaque déploiement du nouveau conteneur de l'API
|   ├── light_app/ # Dossier contenant les tests unitaires pour l'API de test (light_app)
|   ├── tf_app/ # Dossier contenant les tests unitaires pour l'API de production (tf_app)
├── README.md # Ce fichier 
├── mlflow.db # La base de données MLflow pour suivre les expériences et les métriques de chaque modèle testé
├── requirements.txt # Fichier contenant les dépendances du projet utilisés pour l'entraînement avec le GPU
├── request.sh # Script avec une commande CURL pour tester l'API