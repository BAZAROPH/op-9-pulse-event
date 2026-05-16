import os
import time
import pandas as pd
from dotenv import load_dotenv

#Importation de DeepEval
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM 
from langchain_mistralai import ChatMistralAI
from src.rag_manager import RAGManager

#On crée un petit wrapper pour que DeepEval accepte Mistral
class MistralDeepEval(DeepEvalBaseLLM):
    def __init__(self, model):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        return self.model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        res = await self.model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return "Mistral Large"

def main():
    load_dotenv()
    api_key = os.getenv("MISTRAL_API")

    #1 Configurer le modèle juge (Mistral)
    base_mistral = ChatMistralAI(model="mistral-small-latest", mistral_api_key=api_key)
    mistral_model = MistralDeepEval(base_mistral)

    #2 Préparation du jeu de données (Ground Truth)
    test_questions = [
        {
            "question": "Quel est le prix de l'évènement qui à lieu le Mercredi 2 juillet au 15 rue causserouge ?",
            "ground_truth": "10€ sur inscription"
        },
        {
            "question": "Quand a lieu l'évènement sur les stages professionnels en Europe ?",
            "ground_truth": "Mercredi 10 septembre 2025, 09h30"
        },
        {
            "question": "Quels sont les évènements qui ont lieu à la Place Pey-Berland, 33000 Bordeaux ? Pour 6 euros",
            "ground_truth": "Visite guidée : Le gascon, la nature et le paysage à Bordeaux"
        }
    ]

    #3 Collecte des résultats de ton RAG
    print("Interrogation du RAG local...")
    rag = RAGManager()
    
    #Initialisation des métriques
    metric_faith = FaithfulnessMetric(threshold=0.5, model=mistral_model)
    metric_relevancy = AnswerRelevancyMetric(threshold=0.5, model=mistral_model)
    metric_precision = ContextualPrecisionMetric(threshold=0.5, model=mistral_model)
    metric_recall = ContextualRecallMetric(threshold=0.5, model=mistral_model)

    results_list = []

    for item in test_questions:
        #On interroge ton RAG
        result = rag.ask_question_with_context(item["question"])
        
        #Correction : On gère si le contexte est déjà du texte ou un objet Document
        contexts = []
        for doc in result["contexts"]:
            if hasattr(doc, 'page_content'):
                contexts.append(doc.page_content)
            else:
                contexts.append(str(doc))
        
        #On fusionne les contextes en une seule chaîne pour le CSV
        context_str = "\n---\n".join(contexts)
        
        #Création du cas de test
        test_case = LLMTestCase(
            input=item["question"],
            actual_output=result["answer"],
            retrieval_context=contexts,
            expected_output=item["ground_truth"]
        )

        #Calcul des scores
        print(f"\nÉvaluation pour : {item['question'][:50]}...")
        metric_faith.measure(test_case)
        metric_relevancy.measure(test_case)
        metric_precision.measure(test_case)
        metric_recall.measure(test_case)

        #Ajout des données dans la liste (avec context et réponse)
        results_list.append({
            "question": item["question"],
            "retrieved_context": context_str, #Colonne Contexte
            "rag_answer": result["answer"],    #Colonne Réponse
            "ground_truth": item["ground_truth"],
            "faithfulness": metric_faith.score,
            "answer_relevancy": metric_relevancy.score,
            "context_precision": metric_precision.score,
            "context_recall": metric_recall.score
        })
        
        #Pause pour éviter le rate limit du plan Free Mistral
        time.sleep(2)

    #4 Export des résultats
    print("\nRésultats de l'évaluation :")
    df = pd.DataFrame(results_list)
    print(df)

    df.to_csv("evaluation_results_deepeval.csv", index=False)
    print("\nRapport généré : evaluation_results_deepeval.csv")

if __name__ == "__main__":
    main()