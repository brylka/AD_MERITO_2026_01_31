from flask import Flask, render_template, request, jsonify
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

LANGUAGE_KEY = os.getenv('AZURE_LANGUAGE_KEY')
LANGUAGE_ENDPOINT = os.getenv('AZURE_LANGUAGE_ENDPOINT')


@app.route('/')
def index():
    return render_template('f4.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form.get('text')

    url = f"{LANGUAGE_ENDPOINT}language/:analyze-text?api-version=2023-04-01"

    body = {
        "kind": "SentimentAnalysis",
        "parameters": {"modelVersion": "latest"},
        "analysisInput": {
            "documents": [
                {"id": "1", "language": "pl", "text": text}
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
    doc = result["results"]["documents"][0]

    return jsonify({
        "sentiment": doc["sentiment"],
        "positive": f"{doc['confidenceScores']['positive']:.0%}",
        "negative": f"{doc['confidenceScores']['negative']:.0%}",
        "neutral": f"{doc['confidenceScores']['neutral']:.0%}"
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
