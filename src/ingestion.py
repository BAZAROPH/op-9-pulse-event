import os
from pathlib import Path
from dotenv import load_dotenv
import requests
import re
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime, timedelta

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def get_events(params={}):
    """
        Retourne les évènements de open agenda en fonction des paramètres de filtre qui lui sont fourni
    """
    endpoint_url = os.getenv("OPEN_AGENGA_ENDPOINT", "Pas durl")
    print(f"Connexion à : {endpoint_url}")
    
    response = requests.get(endpoint_url, params=params)

    if response.status_code != 200:
        print(f"Erreur code : {response.status_code}")
        print(f"Erreur API : {response.text}")
        response.raise_for_status()
    
    return response.json().get("results")


def process_and_save_to_faiss(events, output_path=PROJECT_ROOT / "faiss_index_events"):
    """
        Transforme les JSON d'Open Agenda en vecteurs et les sauvegarde localement
        Nettoyage Pandas et enrichissement des métadonnées inclus
    """
    import pandas as pd
    import re

    #On transforme la liste d'évènements en DataFrame pour un nettoyage massif
    df = pd.DataFrame(events)

    #1 Nettoyage avec Pandas
    #Remplacer les None par des chaînes vides ou des valeurs par défaut
    df['title_fr'] = df['title_fr'].fillna("Évènement sans titre")
    df['description_fr'] = df['description_fr'].fillna("Description en français non pécisée")
    df['longdescription_fr'] = df['longdescription_fr'].fillna("")
    df['conditions_fr'] = df['conditions_fr'].fillna("Non précisé")
    df['location_name'] = df['location_name'].fillna("Lieu non précisé")
    df['location_city'] = df['location_city'].fillna("Ville non précisée")
    df['location_address'] = df['location_address'].fillna("")
    df['daterange_fr'] = df['daterange_fr'].fillna("Date non communiquée")
    
    #Nettoyage HTML sur toute la colonne longdescription
    df['clean_long_desc'] = df['longdescription_fr'].apply(lambda x: re.sub('<[^<]+?>', '', str(x)) if x else "")

    #Extraction propre des âges (on garde des entiers pour les filtres futurs)
    df['age_min'] = pd.to_numeric(df['age_min'], errors='coerce').fillna(0).astype(int)
    df['age_max'] = pd.to_numeric(df['age_max'], errors='coerce').fillna(99).astype(int)

    documents_list = []

    #Configurer le splitter pour les chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024, 
        chunk_overlap=250, 
        separators=["\n\n", "\n", ".", "?", "!", " "]
    )

    #On itère sur les lignes nettoyées du DataFrame
    for _, row in df.iterrows():
        
        #2 Construction du texte pour la recherche sémantique (Le "Cerveau")
        #On inclut l'adresse et le prix car l'utilisateur peut chercher "gratuit à bordeaux"
        semantic_content = (
            f"Titre: {row['title_fr']}\n"
            f"Lieu: {row['location_name']}, {row['location_city']} ({row['location_address']})\n"
            f"Quand: {row['daterange_fr']}\n"
            f"Prix/Conditions: {row['conditions_fr']}\n"
            f"Public: De {row['age_min']} à {row['age_max']} ans\n"
            f"Résumé: {row['description_fr']}\n"
            f"Détails: {row['clean_long_desc']}"
        )

        #Découper le full_text en chunk
        text_chunks = text_splitter.split_text(semantic_content)

        #3 Stocker le maximum d'infos pertinentes dans les métadatas (Le "Filtre")
        metadata_payload = {
            "uid": str(row.get("uid")),
            "url": row.get("canonicalurl"),
            "image_url": row.get("image"),
            "title": row['title_fr'],
            "address": row['location_address'],
            "price": row['conditions_fr'],
            "schedule": row['daterange_fr'],
            "venue_name": row['location_name'],
            "city": row['location_city'],
            "postal_code": row.get("location_postalcode"),
            "region": row.get("location_region"),
            "age_min": row['age_min'],
            "age_max": row['age_max'],
            "latitude": row.get("location_coordinates", {}).get("lat") if isinstance(row.get("location_coordinates"), dict) else None,
            "longitude": row.get("location_coordinates", {}).get("lon") if isinstance(row.get("location_coordinates"), dict) else None,
            "updated_at": row.get("updatedat")
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
    model_embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        api_key=os.getenv("MISTRAL_API")
    )

    #5 Création de l'index FAISS à partir de tous les documents
    print(f"Vectorisation de {len(documents_list)} chunks...")
    vector_store = FAISS.from_documents(documents_list, model_embeddings)

    #6 Enregistrer l'index en local
    #Cela créer un dossier "faiss_index_events" contenant l'index et les métadonnées
    vector_store.save_local(output_path)

    print(f"Indexation terminée et sauvegardée dans '{output_path}'")


#----- Test
if __name__ == "__main__":
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    end_date = (datetime.now() + timedelta(days=365)).strftime("%Y-%m-%d")
    location_city="Bordeaux"
    
    query_params = {
        "where": f"firstdate_begin >= '{start_date}' AND firstdate_begin <= '{end_date}' AND location_city='{location_city}'",
        "limit": 100
    }

    print(f"Requête filtrée du {start_date} au {end_date}")
    event_results = get_events(params=query_params)

    if event_results:
        print(f"Succès : {len(event_results)} évènements récupérés.")
        process_and_save_to_faiss(events=event_results)
    else:
        print("Aucun évènement trouvé.")