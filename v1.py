import requests
import os
from dotenv import load_dotenv

load_dotenv()

VISION_KEY = os.getenv('AZURE_VISION_KEY')
VISION_ENDPOINT = os.getenv('AZURE_VISION_ENDPOINT')

# URL obrazka do analizy (możesz też użyć lokalnego pliku)
image_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSQIc45SyCcYJGm3KMXfaBD1spdLlZ1L4QL4Q&s"

# Endpoint API
analyze_url = f"{VISION_ENDPOINT}vision/v3.2/analyze"

# Co chcemy wykryć
params = {
    'visualFeatures': 'Categories,Description,Tags,Objects,Faces'
}

# Wyślij request
response = requests.post(
    analyze_url,
    headers={
        'Ocp-Apim-Subscription-Key': VISION_KEY,
        'Content-Type': 'application/json'
    },
    params=params,
    json={'url': image_url}
)

result = response.json()
# print(result)

# Wyświetl wyniki
print("=== OPIS ===")
print(result['description']['captions'][0]['text'])

print("\n=== TAGI ===")
for tag in result['tags'][:5]:
    print(f"- {tag['name']} ({tag['confidence']:.0%})")

print("\n=== OBIEKTY ===")
for obj in result.get('objects', []):
    print(f"- {obj['object']}")