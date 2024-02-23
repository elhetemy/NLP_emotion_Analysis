

from flask import Flask, render_template, request
import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

def load_tokenizer():
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer
# Load the tokenizer
tokenizer = load_tokenizer()
# Load the saved  model
try:
    with open('lstm_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
except EOFError as e:
    print("Error loading the model:", e)


# Define the labels
labels = ['joy', 'sadness', 'anger', 'fear', 'love', 'surprise']

# Load stopwords and initialize WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')

STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function to process the input text
def process_input(text):
    clean_text = [word.lower() for word in text if word.lower() not in STOPWORDS]
    clean_text = [lemmatizer.lemmatize(word) for word in clean_text]
    sequences = tokenizer.texts_to_sequences([clean_text])
    padded_sequences = pad_sequences(sequences, maxlen=200)  # Assuming max_length is 200
    return padded_sequences

# Function to make predictions
def predict_emotion(text):
    processed_text = process_input(text)
    pred = loaded_model.predict(processed_text)
    predicted_label_index = np.argmax(pred)
    predicted_label = labels[predicted_label_index-1]
    return predicted_label

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        predicted_emotion = predict_emotion(text)
        return render_template('index.html', prediction=predicted_emotion)

if __name__ == '__main__':
    app.run(debug=True)


