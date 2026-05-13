import os
from src.ingestion import get_events, process_and_save_to_faiss

def test_real_api_ingestion():
    # 1 Vérifier que l'URL est bien configurée
    endpoint = os.getenv("OPEN_AGENGA_ENDPOINT")
    assert endpoint is not None, "L'URL de l'API n'est pas configurée dans le .env"

    #2 Test de l'appel réel
    params = {"limit": 2}
    events = get_events(params=params)
    
    assert isinstance(events, list)
    assert len(events) > 0, "L'API a renvoyé une liste vide"
    assert "title_fr" in events[0], "La structure du JSON ne contient pas 'title_fr'"

def test_real_faiss_creation():
    #1 On récupère 2 événements réels
    events = get_events(params={"limit": 2})
    
    #2 On lance la création réelle de l'index
    process_and_save_to_faiss(events)
    
    #3 On vérifie que les fichiers ont été créés physiquement
    assert os.path.exists("faiss_index_events/index.faiss")
    assert os.path.exists("faiss_index_events/index.pkl")