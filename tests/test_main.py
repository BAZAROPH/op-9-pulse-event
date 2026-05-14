import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_read_root():
    """
        Vérifie que l'API répond au ping initial
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

def test_ask_question_success():
    """
        Vérifie que le RAG répond correctement à une question valide
    """
    payload = {"question": "Quels sont les événements à Paris ?"}
    response = client.post("/ask", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert data["question"] == payload["question"]

def test_ask_question_empty():
    """
        Vérifie que l'API renvoie bien une erreur 400 si la question est vide
    """
    payload = {"question": "   "}
    response = client.post("/ask", json=payload)
    
    # Ici on attend bien une 400 car on a corrigé le bloc try/except
    assert response.status_code == 400
    assert "vide" in response.json()["detail"]

def test_rebuild_index_success():
    """
        Vérifie que la reconstruction se lance avec le bon mot-clé
    """
    payload = {"confirm": "rebuild"}
    response = client.post("/rebuild", json=payload)
    
    assert response.status_code == 200
    assert response.json()["message"] == "Rebuilding effectué avec succès !"

def test_rebuild_index_wrong_key():
    """
        Vérifie que l'action est annulée si le mot-clé est faux
    """
    payload = {"confirm": "invalid_key"}
    response = client.post("/rebuild", json=payload)
    
    assert response.status_code == 200
    assert "annulée" in response.json()["message"]