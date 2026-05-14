import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from src.ingestion import process_and_save_to_faiss
from src.main import app

client = TestClient(app)

import shutil

#Dossier temporaire pour  ne pas polluer
TEMP_INDEX_PATH = Path(__file__).resolve().parents[1] / "faiss_index_test"

@pytest.fixture(scope="module", autouse=True)
def setup_test_index():
    """
        Crée un petit index FAISS de test avant de lancer les test du module.
        Supprime le dossier après tests.
    """
    #On prépare les faux events
    fake_events = [
        {
            "uid": "43692627",
            "slug": "dedicace-de-peyo-lizarazu",
            "canonicalurl": "https://openagenda.com/librairie-mollat/events/dedicace-de-peyo-lizarazu",
            "title_fr": "D\u00e9dicace de Peyo Lizarazu",
            "description_fr": "Venez rencontrer Peyo Lizarazu lors d'une s\u00e9ance de d\u00e9dicace de son livre \"Vies de surf\" aux \u00e9ditions La Martini\u00e8re.",
            "longdescription_fr": "<p>\"Une d\u00e9couverte des vagues les plus majestueuses de la plan\u00e8te, accompagn\u00e9s de portraits des personnalit\u00e9s marquantes du monde du surf, de documents d'archives et de photographies.\" \u00a9Electre 2022</p>\n<p></p>\n<p>Rendez-vous d\u00e8s 15 heures \u00e0 la Librairie Mollat.</p>",
            "conditions_fr": None,
            "keywords_fr": None,
            "image": "https://cibul.s3.amazonaws.com/d0962598cf834ec58e32a23ccc3422a5.base.image.jpg",
            "imagecredits": None,
            "thumbnail": "https://cibul.s3.amazonaws.com/d0962598cf834ec58e32a23ccc3422a5.thumb.image.jpg",
            "originalimage": "https://cibul.s3.amazonaws.com/d0962598cf834ec58e32a23ccc3422a5.full.image.jpg",
            "updatedat": "2022-10-25T09:00:46+00:00",
            "daterange_fr": "Mercredi 16 novembre, 15h00",
            "firstdate_begin": "2022-11-16T14:00:00+00:00",
            "firstdate_end": "2022-11-16T17:00:00+00:00",
            "lastdate_begin": "2022-11-16T14:00:00+00:00",
            "lastdate_end": "2022-11-16T17:00:00+00:00",
            "timings": "[{\"begin\": \"2022-11-16T15:00:00+01:00\", \"end\": \"2022-11-16T18:00:00+01:00\"}]",
            "accessibility": None,
            "accessibility_label_fr": None,
            "location_uid": "65949775",
            "location_coordinates": {
                "lon": -0.578647,
                "lat": 44.840868
            },
            "location_name": "Librairie Mollat",
            "location_address": "15 rue Vital Carles, 33000 Bordeaux",
            "location_district": "Triangle d'Or",
            "location_insee": "33063",
            "location_postalcode": "33000",
            "location_city": "Bordeaux",
            "location_department": "Gironde",
            "location_region": "Nouvelle-Aquitaine",
            "location_countrycode": "FR",
            "location_image": None,
            "location_imagecredits": None,
            "location_phone": None,
            "location_website": None,
            "location_links": None,
            "location_tags": None,
            "location_description_fr": None,
            "location_access_fr": None,
            "attendancemode": "{\"id\": 1, \"label\": {\"fr\": \"Sur place\", \"en\": \"Offline\", \"it\": \"In presenza\", \"es\": \"Desconnectad\", \"de\": \"Offline\", \"br\": \"War al lec\\u2019h\", \"io\": \"crwdns14266:0crwdne14266:0\"}}",
            "onlineaccesslink": None,
            "status": "{\"id\": 1, \"label\": {\"fr\": \"Programm\\u00e9\", \"en\": \"Scheduled\", \"io\": \"crwdns16100:0crwdne16100:0\"}}",
            "age_min": None,
            "age_max": None,
            "originagenda_title": "Librairie Mollat",
            "originagenda_uid": "30224219",
            "contributor_email": None,
            "contributor_contactnumber": None,
            "contributor_contactname": None,
            "contributor_contactposition": None,
            "contributor_organization": None,
            "category": None,
            "country_fr": "France (M\u00e9tropole)",
            "registration": None,
            "links": None
        },
        {
            "uid": "61763667",
            "slug": "la-peur-en-voyage",
            "canonicalurl": "https://openagenda.com/reseau-des-mediatheques-du-mans/events/la-peur-en-voyage",
            "title_fr": "La peur en voyage",
            "description_fr": "Venez frissonner \u00e0 l\u2019international !\nLectures r\u00e9alis\u00e9es en partenariat avec l\u2019association AFaLaC.",
            "longdescription_fr": "<p>Venez frissonner \u00e0 l\u2019international !<br />Lectures r\u00e9alis\u00e9es en partenariat avec l\u2019association AFaLaC.<br /><em>\u00c0 partir de 4 ans.</em></p>",
            "conditions_fr": None,
            "keywords_fr": None,
            "image": "https://cibul.s3.amazonaws.com/fe5e41c82f1242268a426d8f2820e36e.base.image.jpg",
            "imagecredits": None,
            "thumbnail": "https://cibul.s3.amazonaws.com/fe5e41c82f1242268a426d8f2820e36e.thumb.image.jpg",
            "originalimage": "https://cibul.s3.amazonaws.com/fe5e41c82f1242268a426d8f2820e36e.full.image.jpg",
            "updatedat": "2023-01-03T16:01:24+00:00",
            "daterange_fr": "Samedi 21 janvier, 19h00",
            "firstdate_begin": "2023-01-21T18:00:00+00:00",
            "firstdate_end": "2023-01-21T18:30:00+00:00",
            "lastdate_begin": "2023-01-21T18:00:00+00:00",
            "lastdate_end": "2023-01-21T18:30:00+00:00",
            "timings": "[{\"begin\": \"2023-01-21T19:00:00+01:00\", \"end\": \"2023-01-21T19:30:00+01:00\"}]",
            "accessibility": None,
            "accessibility_label_fr": None,
            "location_uid": "87034922",
            "location_coordinates": {
                "lon": 0.192803,
                "lat": 48.00265
            },
            "location_name": "M\u00e9diath\u00e8que Aragon",
            "location_address": "54 rue du Port, 72 000 Le Mans",
            "location_district": None,
            "location_insee": "72181",
            "location_postalcode": "72000",
            "location_city": "Le Mans",
            "location_department": "Sarthe",
            "location_region": "Pays de la Loire",
            "location_countrycode": "FR",
            "location_image": None,
            "location_imagecredits": None,
            "location_phone": None,
            "location_website": None,
            "location_links": None,
            "location_tags": None,
            "location_description_fr": None,
            "location_access_fr": None,
            "attendancemode": "{\"id\": 1, \"label\": {\"fr\": \"Sur place\", \"en\": \"Offline\", \"it\": \"In presenza\", \"es\": \"Desconnectad\", \"de\": \"Offline\", \"br\": \"War al lec\\u2019h\", \"io\": \"crwdns14266:0crwdne14266:0\"}}",
            "onlineaccesslink": None,
            "status": "{\"id\": 1, \"label\": {\"fr\": \"Programm\\u00e9\", \"en\": \"Scheduled\", \"io\": \"crwdns16100:0crwdne16100:0\"}}",
            "age_min": 4,
            "age_max": 99,
            "originagenda_title": "R\u00e9seau des m\u00e9diath\u00e8ques du Mans",
            "originagenda_uid": "48454528",
            "contributor_email": None,
            "contributor_contactnumber": None,
            "contributor_contactname": None,
            "contributor_contactposition": None,
            "contributor_organization": None,
            "category": None,
            "country_fr": "France (M\u00e9tropole)",
            "registration": None,
            "links": None
        },
    ]

    #Générer l'index
    process_and_save_to_faiss(fake_events)

    # On force le rechargement dans l'app si nécessaire
    from src.main import rag as main_rag
    import src.main as main_module
    if main_module.rag is None:
        from src.rag_manager import RAGManager
        main_module.rag = RAGManager(index_path=TEMP_INDEX_PATH)
    
    yield
    
    #Nettoyage après tests
    if os.path.exists(TEMP_INDEX_PATH):
        shutil.rmtree(TEMP_INDEX_PATH)

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
    
    #Ici on attend bien une 400 car on a corrigé le bloc try/except
    assert response.status_code == 400
    assert "vide" in response.json()["detail"]

def test_rebuild_index_success():
    """
        Vérifie le rebuild avec filtres par défaut
    """
    payload = {"confirm": "rebuild", "city": "Bordeaux"}
    response = client.post("/rebuild", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "succès" in data["message"]
    assert data["details"]["city"] == "Bordeaux"

def test_rebuild_index_wrong_key():
    """
        Vérifie que l'action est annulée si le mot-clé est faux
    """
    payload = {"confirm": "invalid_key"}
    response = client.post("/rebuild", json=payload)
    
    assert response.status_code == 200
    assert "annulée" in response.json()["message"]