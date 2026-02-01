import requests
import os
from dotenv import load_dotenv

load_dotenv()

LANGUAGE_KEY = os.getenv('AZURE_LANGUAGE_KEY')
LANGUAGE_ENDPOINT = os.getenv('AZURE_LANGUAGE_ENDPOINT')

# Teksty do analizy
texts = [
    "Jestem bardzo zadowolony z tego produktu! Świetna jakość.",
    "To był okropny dzień, wszystko poszło nie tak.",
    "Spotkanie odbędzie się w piątek o 15:00."
]

# === ANALIZA SENTYMENTU ===
url = f"{LANGUAGE_ENDPOINT}language/:analyze-text?api-version=2023-04-01"

body = {
    "kind": "SentimentAnalysis",
    "parameters": {"modelVersion": "latest"},
    "analysisInput": {
        "documents": [
            {"id": str(i), "language": "pl", "text": text}
            for i, text in enumerate(texts)
        ]
    }
}

response = requests.post(
    url,
    headers={
        "Ocp-Apim-Subscription-Key": LANGUAGE_KEY,
        "Content-Type": "application/json"
    },
    json=body
)

result = response.json()

print("=== ANALIZA SENTYMENTU ===\n")
for doc in result["results"]["documents"]:
    text = texts[int(doc["id"])]
    sentiment = doc["sentiment"]
    scores = doc["confidenceScores"]
    print(f"Tekst: {text[:50]}...")
    print(f"Sentyment: {sentiment}")
    print(f"Pewność: pozytywny={scores['positive']:.0%}, negatywny={scores['negative']:.0%}, neutralny={scores['neutral']:.0%}\n")
