import pickle
import flask
from flask_cors import CORS
from flask import Flask, request, jsonify
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Create flask app
app = Flask(__name__)
CORS(app)

# Print Flask version
print("flask vers", flask.__version__) 

# Load pre-trained KNN model from pickle file
with open("iris.pkl", "rb") as file:
    knn = pickle.load(file)

# Load pre-trained sentiment analysis model and tokenizer
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
sentiment_analysis = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Root route for the Flask app
@app.route("/")
def root():
    return jsonify({"message": "flask worked correctly!"})

# Iris prediction route
@app.route("/iris", methods=["POST"])
def iris():
    # Get new data from requests
    new_data = request.get_json()

    # Change the string data type to float
    new_data_float = {k: float(v) for k, v in new_data.items()}

    # Predicting iris species
    prediction = knn.predict([list(new_data_float.values())])

    # Return the prediction as a JSON response
    return jsonify({'species': prediction[0]})

# Sentiment analysis route
@app.route('/sentence', methods=['POST'])
def sentence():
    # Get data from the request
    data = request.json
    # Extract the sentence from the incoming data
    sentence = data.get('sentence')
    # Check if a sentence is provided
    if not sentence:
        # Return an error if no sentence is provided
        return jsonify({'error': 'No sentence provided'}), 400
    # Perform sentiment analysis on the sentence
    result = sentiment_analysis(sentence)[0]
    # Return the sentiment label and score as a JSON response
    return jsonify({
        'label': result['label'],
        'score': result['score']
    })

# Main entry point to run the Flask app
if __name__ == "__main__":
    app.run(debug=True)