import os
from dotenv import load_dotenv
import requests
import re
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime, timedelta

load_dotenv()

def get_events(params={}):
    """
        Retourne les évènements de open agenda en fonction des paramètres de filtre qui lui sont fourni
    """
    endpoint_url = os.getenv("OPEN_AGENGA_ENDPOINT")
    print(f"Connexion à : {endpoint_url}")
    
    response = requests.get(endpoint_url, params=params)

    if response.status_code != 200:
        print(f"Erreur code : {response.status_code}")
        print(f"Erreur API : {response.text}")
        response.raise_for_status()
    
    return response.json().get("results")


def process_and_save_to_faiss(events):
    """
        Transforme les JSON d'Open Agenda en vecteurs et les sauvegarde localement
    """
    documents_list = []

    #Configurer le splitter pour les chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, 
        chunk_overlap=200, 
        separators=["\n\n", "\n", ".", "?", "!", " "]
    )

    for event in events:
        #1 Préparation du texte (Fusion titre + description)
        event_title = event.get("title_fr", "Évènement sans titre")
        event_description = event.get("description_fr", "")
        event_long_desc = event.get("longdescription_fr", "")
        event_city = event.get("location_city", "Lieu non précisé")
        event_address = event.get("location_address", "")
        event_pricing = event.get("conditions_fr", "Non précisé")

        #Nettoyage des balises HTML dans la description longue
        clean_details = re.sub('<[^<]+?>', '', event_long_desc) if event_long_desc else ""

        #2 Construction du texte pour la recherche sémantique
        #On met les informations les plus riche ici
        semantic_content = (
            f"Title: {event_title}\n"
            f"City: {event_city}\n"
            f"Description: {event_description}\n"
            f"Details: {clean_details}"
        )

        #Découper le full_text en chunk
        text_chunks = text_splitter.split_text(semantic_content)

        #3 Stocker les infos pratiques dans les métadatas
        metadata_payload = {
            "uid": event.get("uid"),
            "url": event.get("canonicalurl"),
            "image_url": event.get("image"),
            "address": event_address,
            "price": event_pricing,
            "schedule": event.get("daterange_fr"),
            "venue_name": event.get("location_name"),
            "city": event_city
        }

        #Transformer chaque chunk en Document LangChain
        for chunk in text_chunks:
            new_doc = Document(
                page_content=chunk,
                metadata=metadata_payload
            )
            #On l'ajoute à la liste de docs
            documents_list.append(new_doc)
        

    #4 Initialisation du modèle d'embedding
    #On transforme les phrases en vecteurs sentence avec sentence-trasnformer qui est local et gratuit. Idéal pour un POC
    model_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    #5 Création de l'index FAISS à partir de tous les documents
    print(f"Vectorisation de {len(documents_list)} chunks...")
    vector_store = FAISS.from_documents(documents_list, model_embeddings)

    #6 Enregistrer l'index en local
    #Cela créer un dossier "faiss_index_events" contenant l'index et les métadonnées
    output_directory = "faiss_index_events"
    vector_store.save_local(output_directory)

    print(f"Indexation terminée et sauvegardée dans '{output_directory}'")


#----- Test
if __name__ == "__main__":
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    end_date = (datetime.now() + timedelta(days=365)).strftime("%Y-%m-%d")
    
    query_params = {
        "where": f"firstdate_begin >= '{start_date}' AND firstdate_begin <= '{end_date}'",
        "limit": 100
    }

    print(f"Requête filtrée du {start_date} au {end_date}")
    event_results = get_events(params=query_params)

    if event_results:
        print(f"Succès : {len(event_results)} évènements récupérés.")
        process_and_save_to_faiss(events=event_results)
    else:
        print("Aucun évènement trouvé.")