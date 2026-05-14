import os
# Correctif pour Mac (OpenMP)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.rag_manager import RAGManager
from src import ingestion

#Définir le format sous lequel le user doit envoyer sa requête
class QueryRequest(BaseModel):
    question: str

class RebuildRequest(BaseModel):
    confirm: str

#Initialiser l'API et le RAG
app = FastAPI(title="Pulse Events RAG API")
rag = RAGManager(index_path="faiss_index_events")

@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Puls-Events API is running"
    }

#Le point d'entrée proncipal pour le RAG
@app.post("/ask")
def ask_question(request: QueryRequest):
    """
        Route qui prend un question en paramètre et renvoi la réponse générée à partir du RAG
    """
    try:
        if not request.question:
            raise HTTPException(status_code=400, detail="La question ne peut pas être vide")
        
        #on appelle le moteur RAG
        answer = rag.ask_question(request.question)

        return {
            "question": request.question,
            "answer": answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rebuild")
def rebuild_index(request: RebuildRequest):
    """
        Route qui prend une confirmation de rebuild puis rebuild l'index faiss
    """
    try:
        if not request.confirm:
            raise HTTPException(status_code=400, detail="Vous devez confirmer le rebuild")
        
        if request.confirm == "rebuild":
            events = ingestion.get_events()
            ingestion.process_and_save_to_faiss(events)

            #Recharger l'index
            rag._load_index()

            return "Rebuilding effectué avec sucès !"
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))