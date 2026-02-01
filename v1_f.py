from flask import Flask, render_template, request, jsonify
import requests
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

VISION_KEY = os.getenv('AZURE_VISION_KEY')
VISION_ENDPOINT = os.getenv('AZURE_VISION_ENDPOINT')


@app.route('/')
def index():
    return render_template('f3.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    # Pobierz obraz
    image_file = request.files['image']
    image_data = image_file.read()

    # Endpoint API
    analyze_url = f"{VISION_ENDPOINT}vision/v3.2/analyze"

    # Co chcemy wykryć
    params = {
        'visualFeatures': 'Categories,Description,Tags,Objects'
    }

    # Wyślij request z binarnym obrazem
    response = requests.post(
        analyze_url,
        headers={
            'Ocp-Apim-Subscription-Key': VISION_KEY,
            'Content-Type': 'application/octet-stream'
        },
        params=params,
        data=image_data
    )

    result = response.json()

    # Sprawdź błędy
    if 'error' in result:
        return jsonify({
            'success': False,
            'error': result['error']['message']
        })

    # Przygotuj odpowiedź
    return jsonify({
        'success': True,
        'description': result['description']['captions'][0]['text'] if result['description']['captions'] else 'Brak opisu',
        'tags': [{'name': tag['name'], 'confidence': f"{tag['confidence']:.0%}"} for tag in result['tags'][:10]],
        'objects': [obj['object'] for obj in result.get('objects', [])]
    })


if __name__ == '__main__':
    app.run(debug=True, port=8000)