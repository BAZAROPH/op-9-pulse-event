import os
from dotenv import load_dotenv
import requests

load_dotenv()

def get_events(params={}):
    """
        Retourne les évènements de open agenda en fonction des paramètres de filtre qui lui sont fourni
    """
    url = os.getenv("OPEN_AGENGA_ENDPOINT")
    print(url)
    response = requests.get(url, params=params)

    if response.status_code != 200:
        print("Erreur code :", response.status_code)
        print("Erreur API :", response.text)
        response.raise_for_status()
    
    return response.json().get("results")

