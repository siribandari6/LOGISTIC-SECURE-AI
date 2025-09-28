import json
import os       # Import the 'os' library to access environment variables
import requests # Import the 'requests' library
from flask import Flask, render_template, jsonify

app = Flask(__name__)

# --- DATA LOADING ---
# Load the PRE-CALCULATED ML results at startup
try:
    with open('ml_evaluation_results.json', 'r') as f:
        ML_RESULTS = json.load(f)
    print("✅ Pre-calculated ML evaluation results loaded.")
except Exception as e:
    print(f"❌ Error loading ML results file 'ml_evaluation_results.json': {e}")
    print("   Please run 'refined.py' first.")
    ML_RESULTS = {}

# --- API ROUTES ---
@app.route('/')
def index():
    """Serves the main, all-in-one HTML page."""
    return render_template('index.html')

# --- MODIFIED NEWS ROUTE ---
@app.route('/api/news')
def get_news():
    """Provides live news data from NewsAPI.org via API."""
    
    # Safely get the API key from the environment variable
    api_key = "bb5983b69437478395d0af26d5dfc6dd"

    if not api_key:
        print("❌ NEWS_API_KEY environment variable not set.")
        # Return a helpful error message if the key is missing
        return jsonify([{"title": "News API Key Not Configured", "description": "Please set the NEWS_API_KEY environment variable.", "link": "#", "imgSrc": ""}])

    # Construct the URL to search for relevant topics
    url = (
        'https://newsapi.org/v2/everything?'
        'q=(logistics AND fraud) OR ("supply chain" AND AI) OR ("shipping technology")&'
        'language=en&'
        'sortBy=publishedAt&'
        'pageSize=10&'
        f'apiKey={api_key}'
    )

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        news_data = response.json()
        
        # Format the articles from NewsAPI to match what the frontend expects
        formatted_articles = []
        for article in news_data.get('articles', []):
            formatted_articles.append({
                "title": article.get('title'),
                "description": article.get('description'),
                "link": article.get('url'),
                "imgSrc": article.get('urlToImage') or 'https://images.unsplash.com/photo-1504711434969-e33886168f5c?q=80&w=2070&auto=format&fit=crop' # Provide a fallback image
            })
        return jsonify(formatted_articles)

    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching news from NewsAPI: {e}")
        return jsonify([{"title": "Error Fetching News", "description": "Could not connect to the news service.", "link": "#", "imgSrc": ""}])


@app.route('/api/ml_metrics/classification/<dataset_name>')
def get_ml_classification(dataset_name):
    """Provides the pre-calculated classification metrics from the JSON file."""
    results = ML_RESULTS.get(dataset_name)
    if results:
        # The frontend expects percentages, let's format them
        response_data = results.copy()
        response_data['accuracy'] = f"{response_data.get('accuracy', 0) * 100:.2f}%"
        response_data['precision'] = f"{response_data.get('precision', 0) * 100:.2f}%"
        response_data['recall'] = f"{response_data.get('recall', 0) * 100:.2f}%"
        response_data['f1Score'] = f"{response_data.get('f1Score', 0) * 100:.2f}%"
        return jsonify(response_data)
    else:
        return jsonify({"error": f"Results for dataset '{dataset_name}' not found."}), 404

if __name__ == '__main__':
    app.run(debug=True)