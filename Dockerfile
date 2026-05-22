#Utiliser une image slégère
FROM python:3.14-slim

#Éviter que python génère des fichier .pyc
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

#Installer les dépendances
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

#Copier et installer les dépendances python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#Copier tout le projet
COPY . .

#Exposition du port FastAPI
EXPOSE 8000

#Commande pour lancer l'API
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]