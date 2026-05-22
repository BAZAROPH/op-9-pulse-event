# Pulse Events RAG API 🎯📅

Un système de questions/réponses intelligent (RAG) basé sur Mistral AI et FAISS, accessible via une API REST avec FastAPI. Ce projet permet d'interroger dynamiquement une base de données d'événements culturels ou professionnels provenant d'OpenAgenda, et d'obtenir des réponses précises en langage naturel.

## 🚀 Fonctionnalités Principales

- **Moteur RAG (Retrieval-Augmented Generation)** : Combine la recherche sémantique (FAISS) et la génération de texte (Mistral AI) pour répondre aux questions sur les événements.
- **API REST (FastAPI)** : Endpoints complets pour discuter avec l'IA et pour reconstruire l'index dynamique à la volée.
- **Pipeline d'Ingestion robuste** : Nettoyage massif de la donnée avec Pandas, chunking intelligent et vectorisation via Mistral Embeddings.
- **Évaluation RAG (DeepEval)** : Script d'évaluation complet (`evaluation.py`) mesurant les performances du RAG via 4 métriques : Faithfulness, Answer Relevancy, Contextual Precision et Contextual Recall. Prise en compte automatique des Rate Limits du tier gratuit de Mistral.

## 🛠️ Stack Technique

- **Langage** : Python 3.10+
- **Framework Web** : FastAPI
- **LLM & Embeddings** : Mistral AI (`mistral-small-latest`, `mistral-embed`)
- **Orchestration LLM** : Langchain
- **Base de données Vectorielle** : FAISS (Local)
- **Data Engineering** : Pandas, Requests
- **Évaluation** : DeepEval

## 📂 Architecture du Projet

```text
op-9-pulse-event/
├── faiss_index_events/     # Dossier généré contenant l'index vectoriel FAISS local
├── src/
│   ├── main.py             # Application FastAPI (Endpoints /ask et /rebuild)
│   ├── ingestion.py        # Script de récupération, nettoyage et vectorisation des événements (OpenAgenda)
│   ├── rag_manager.py      # Cœur du RAG (Retriever, Prompt, Chains Mistral/Langchain)
│   └── evaluation.py       # Script d'évaluation des performances du RAG avec DeepEval
├── .env                    # Fichier des variables d'environnement (API Keys)
├── requirements.txt        # Dépendances Python
└── README.md               # Documentation
```

## ⚙️ Installation & Configuration

1. **Cloner le dépôt et se placer dans le dossier**
2. **Créer et activer un environnement virtuel**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Sur Mac/Linux
   # .venv\Scripts\activate   # Sur Windows
   ```
3. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```
4. **Configurer les variables d'environnement**
   Créez un fichier `.env` à la racine du projet et ajoutez vos clés :
   ```env
   MISTRAL_API=votre_cle_mistral
   MISTRAL_API_KEY=votre_cle_mistral
   OPEN_AGENGA_ENDPOINT=https://openagenda.com/agendas/.../events.json
   HF_TOKEN=votre_token_huggingface  # (Si nécessaire)
   ```

## 🏃‍♂️ Utilisation

### 1. Ingestion initiale des événements
Avant de lancer l'API, vous devez générer la base vectorielle locale :
```bash
python src/ingestion.py
```
> Cela va créer le dossier `faiss_index_events/`.

### 2. Démarrer le serveur API
```bash
uvicorn src.main:app --reload
```
L'API sera disponible sur : [http://127.0.0.1:8000](http://127.0.0.1:8000)
La documentation interactive Swagger est accessible sur : [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 3. Endpoints de l'API

- **`GET /`** : Statut de l'API.
- **`POST /ask`** : Poser une question au RAG.
  ```json
  {
    "question": "Quels sont les évènements à Bordeaux pour 6 euros ?"
  }
  ```
- **`POST /rebuild`** : Forcer la reconstruction de l'index FAISS selon une ville et une période.
  ```json
  {
    "confirm": "rebuild",
    "city": "Bordeaux",
    "days_past": 30,
    "days_future": 60,
    "limit": 100
  }
  ```

### 4. Évaluation du RAG
Pour lancer l'évaluation des performances du RAG avec DeepEval (génère un fichier `evaluation_results_deepeval.csv`) :
```bash
python src/evaluation.py
```
> *Note : Le script inclut des pauses (`time.sleep`) pour respecter les Rate Limits très stricts de l'API Mistral gratuite (environ 1 req/s).*

## 🐳 Utilisation avec Docker

Le projet est packagé avec Docker pour faciliter son déploiement. Vous pouvez construire et lancer l'API dans un conteneur isolé.

### 1. Construire l'image Docker
Depuis la racine du projet (où se trouve le `Dockerfile`) :
```bash
docker build -t pulse-event-rag:v1 .
```

### 2. Lancer le conteneur
Assurez-vous que votre fichier `.env` est bien configuré avec vos clés API. Vous pouvez ensuite lancer le conteneur en liant le port 8000 et en passant le fichier d'environnement :
```bash
docker run -p 8000:8000 --env-file .env pulse-event-rag:v1
```

L'API sera instantanément accessible sur [http://localhost:8000](http://localhost:8000) et la documentation Swagger sur [http://localhost:8000/docs](http://localhost:8000/docs).

> **Important** : Si vous n'avez pas généré l'index `faiss_index_events/` localement avant de builder l'image, vous devrez appeler l'endpoint `/rebuild` via l'API (ou Swagger) une fois le conteneur lancé pour créer l'index vectoriel.
