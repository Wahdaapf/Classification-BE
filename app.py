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

print("flask vers", flask.__version__) 

with open("iris.pkl", "rb") as file:
    knn = pickle.load(file)

model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
sentiment_analysis = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

@app.route("/")
def root():
    return jsonify({"message": "flask worked correctly!"})

@app.route("/iris", methods=["POST"])
def iris():
    # Get new data from requests
    new_data = request.get_json()

    # Change the string data type to float
    new_data_float = {k: float(v) for k, v in new_data.items()}

    # Predicting iris species
    prediction = knn.predict([list(new_data_float.values())])

    return jsonify({'species': prediction[0]})

@app.route('/sentence', methods=['POST'])
def sentence():
    data = request.json
    sentence = data.get('sentence')
    if not sentence:
        return jsonify({'error': 'No sentence provided'}), 400
    
    result = sentiment_analysis(sentence)[0]
    return jsonify({
        'label': result['label'],
        'score': result['score']
    })

if __name__ == "__main__":
    app.run(debug=True)