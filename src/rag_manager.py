import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate 
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from pathlib import Path

load_dotenv()

class RAGManager:
    """
        Classe responsable de la rtecherche sémantique et de la génération de réponses via Mistral
    """

    def __init__(self, index_path=None):

        #On définit la racine du projet (un niveau au-dessus de 'src')
        #__file__ est le chemin de rag_manager.py
        #.resolve().parents[1] remonte de deux crans pour arriver à la racine
        self.project_root = Path(__file__).resolve().parents[1]
        
        #Si aucun chemin n'est fourni, on cible par défaut la racine
        if index_path is None:
            self.index_path = str(self.project_root / "faiss_index_events")
        else:
            self.index_path = index_path

        #On utilise le même modèle d'embedding que pour l'ingestion
        self.embeddings = MistralAIEmbeddings(
            model="mistral-embed",
            mistral_api_key=os.getenv("MISTRAL_API")
        )

        #Initialisation du modèle de chat Mistral
        self.llm = ChatMistralAI(
            model="mistral-large-latest", #Modèle pas trop généraliste et nopas treès gourmand, efficace pour le POC
            temperature=0.2, #Basse pour évirter les hallucinations
            api_key=os.getenv("MISTRAL_API")
        )
        self.vector_store = self._load_index()

    def _load_index(self):
        """
            Méthode qui charge l'index FAISS local s'il existe
        """
        if os.path.exists(self.index_path):
            print(f"Chargement de l'index depuis {self.index_path}...")

            return FAISS.load_local(
                folder_path=self.index_path,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            raise FileNotFoundError(f"Index introuvable dans le répertoire {self.index_path}")
        
    def ask_question_with_context(self, user_query):
        #Similaire à ask_question mais retourne le dictionnaire complet de la chain
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        
        #On récupère les docs manuellement pour Ragas
        docs = retriever.invoke(user_query)
        context_strings = [doc.page_content for doc in docs]
        
        #On génère la réponse
        answer = self.ask_question(user_query)
        
        return {
            "answer": answer,
            "contexts": context_strings
        }

    def ask_question(self, user_query):
        """
            Méthode qui prend une question, cherche danas les docs et répond via Mistral
        """

        #1 Définir le prompt system
        system_prompt = (
            "Tu es l'assistant de Puls-Events. Réponds aux questions en utilisant UNIQUEMENT "
            "le contexte fourni ci-dessous. Si tu ne trouves pas la réponse, dis que tu ne sais pas poliment."
            "\n\n"
            "Contexte : {context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{input}")
        ])

        #2 Créer  la chaine de récupéraation (Retrieval Chain)
        #Elle  cherche les 6 documents les plus proches avec un seuil de 0.8
        retriever = self.vector_store.as_retriever(search_kwargs={"k":3, "score_threshold": 0.8})
        document_chain = create_stuff_documents_chain(llm=self.llm, prompt=prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        #3 Exécuter la recherche et la génération
        response = retrieval_chain.invoke({"input": user_query})
        return response["answer"]

#Test rapide
if __name__ == "__main__":
    rag = RAGManager()
    question = "Quels sont les évènements culturels prévus à Bordeaux ?"
    answer = rag.ask_question(question),
    print(f"\nQuestion : {question}")
    print(f"\nRéponse : {answer}")