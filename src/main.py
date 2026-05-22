import os
#Correctif indispensable pour la libomp sur mon Mac
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import FastAPI, HTTPException, status, Request
import logging
import time
from pydantic import BaseModel, Field
from src.rag_manager import RAGManager
from src import ingestion
from typing import Optional
from datetime import datetime, timedelta

#Format des requêtes entrantes
class QueryRequest(BaseModel):
    question: str = Field(
        ..., 
        description="La question en langage naturel sur les événements",
        examples=["Tu as des évènements à bordeaux ?"]
    )

class RebuildRequest(BaseModel):
    confirm: str
    city: Optional[str] = "Bordeaux" #Par défaut Bordeaux
    days_past: Optional[int] = 365 #Par défaut 1 an en arrière
    days_future: Optional[int] = 365 #Par défaut 1 an en avant
    limit: Optional[int] = 100 #Par défaut 100 events


#Configurer le formatt des log
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger_api = logging.getLogger("PULSE-API")

#Lancement de l'app et chargement initial du moteur RAG
app = FastAPI(title="Pulse Events RAG API")

#Middleware qui va logger chaque appel d'api automatiquement
@app.middleware("http")
async def log_requestst(request:Request, call_next):
    start_time = time.time()

    #log à l'arriée
    logger_api.info(f"➡ Requête: {request.method} {request.url.path}")

    response = await call_next(request)

    #Calcul di temps de reéponse
    duration = time.time() - start_time
    logger_api.info(f"⬅ Terminé | Status: {response.status_code} | Durée: {duration:.2f}s")

    return response

#On essaie de charger, mais on ne crash pas si l'index n'est pas là
try:
    rag = RAGManager(index_path="faiss_index_events")
except FileNotFoundError:
    rag = None
    print("Warning: Index non trouvé au démarrage. En attente d'un rebuild.")

@app.get("/", status_code=status.HTTP_200_OK)
def read_root():
    return {
        "status": "online",
        "message": "Puls-Events API is running"
    }

@app.post(
    "/ask",
    tags=["Moteur RAG"],
    summary="Poser une question à l'IA",
    response_description="La réponse générée par Mistral AI",
    status_code=status.HTTP_200_OK
)
def ask_question(request: QueryRequest):
    """
        Route principale : elle reçoit ma question et interroge le RAG 
        pour obtenir une réponse basée sur les événements indexés.
    """

    if rag is None:
        raise HTTPException(status_code=503, detail="Le moteur RAG n'est pas encore prêt. Veuillez reconstruire l'index.")
    
    try:
        if not request.question.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail="La question ne peut pas être vide"
            )
        
        #Appler le moteur RAG pour générer la réponse
        answer = rag.ask_question(request.question)

        return {
            "question": request.question,
            "answer": answer
        }
    except HTTPException as he:
        #Laisser remonter les erreurs 400 proprement
        raise he
    except Exception as e:
        #En cas de grosse ereeur technique, renvoie une 500
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=str(e)
        )

@app.post("/rebuild")
def rebuild_index(request: RebuildRequest):
    """
        Route de reconstruction filtrée : permet de choisir la ville et la période.
    """
    try:
        if request.confirm != "rebuild":
            return {"message": "Action annulée : mot-clé de confirmation incorrect."}

        #1 Calcul des dates dynamiques
        start_date = (datetime.now() - timedelta(days=request.days_past)).strftime("%Y-%m-%d")
        end_date = (datetime.now() + timedelta(days=request.days_future)).strftime("%Y-%m-%d")
        
        #2 Construction de la requête filtrée
        query_params = {
            "where": f"firstdate_begin >= '{start_date}' AND firstdate_begin <= '{end_date}' AND location_city='{request.city}'",
            "limit": request.limit
        }

        #3 Exécution de l'ingestion
        print(f"Rebuild lancé pour {request.city} (du {start_date} au {end_date})")
        event_results = ingestion.get_events(params=query_params)

        if not event_results:
            return {"message": f"Aucun événement trouvé pour {request.city} sur cette période."}

        #4 Sauvegarde et rechargement
        ingestion.process_and_save_to_faiss(events=event_results)
        rag.vector_store = rag._load_index()

        return {
            "message": "Rebuilding effectué avec succès !",
            "details": {
                "city": request.city,
                "events_count": len(event_results),
                "period": f"{start_date} to {end_date}"
            }
        }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))