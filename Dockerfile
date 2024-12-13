# Partir d'une image Python 3.10 slim
FROM python:3.10-slim

# Installation des packages système éventuels (ex: lib required)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Définition du répertoire de travail
WORKDIR /app

# Copier les dépendances
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste de l'application (code + répertoires)
COPY . .

# Exposer le port sur lequel l’app écoute
EXPOSE 80

# Lancer l’application avec Uvicorn (modifier main:app si votre fichier s’appelle différemment)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
