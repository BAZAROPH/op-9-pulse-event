import os
#Correctif indispensable pour la libomp sur mon Mac
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from src.rag_manager import RAGManager
from src import ingestion

#Format des requêtes entrantes
class QueryRequest(BaseModel):
    question: str

class RebuildRequest(BaseModel):
    confirm: str

#Lancement de l'app et chargement initial du moteur RAG
app = FastAPI(title="Pulse Events RAG API")
rag = RAGManager(index_path="faiss_index_events")

@app.get("/", status_code=status.HTTP_200_OK)
def read_root():
    return {
        "status": "online",
        "message": "Puls-Events API is running"
    }

@app.post("/ask", status_code=status.HTTP_200_OK)
def ask_question(request: QueryRequest):
    """
        Route principale : elle reçoit ma question et interroge le RAG 
        pour obtenir une réponse basée sur les événements indexés.
    """
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

@app.post("/rebuild", status_code=status.HTTP_200_OK)
def rebuild_index(request: RebuildRequest):
    """
        Route de maintenance : elle me permet de reconstruire l'index FAISS 
        en récupérant les derniers événements et de mettre à jour le moteur en direct.
    """
    try:
        if request.confirm == "rebuild":
            #Lancer la récupération et l'indexation
            events = ingestion.get_events()
            ingestion.process_and_save_to_faiss(events)

            #Important : recharger l'index dans l'objet pour que l'API soit à jour
            rag.vector_store = rag._load_index()

            return {"message": "Rebuilding effectué avec succès !"}
        
        else:
            #Si le mot-clé n'est pas bon, ne pas autoriser le rebuild
            return {"message": "Action annulée : mot-clé de confirmation incorrect."}
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=str(e)
        )