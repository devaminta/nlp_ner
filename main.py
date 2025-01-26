from fastapi import FastAPI, Request
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import uvicorn

app = FastAPI()

# Load the model
model = tf.keras.models.load_model("amharic_ner_model.h5")

# Load word2idx, tag2idx, unique_words, unique_tags
word2idx = {"your": "word2idx", "dictionary": "here"}  # Replace with your word2idx
tag2idx = {"your": "tag2idx", "dictionary": "here"}  # Replace with your tag2idx
unique_tags = ["your", "unique", "tags", "here"]  # Replace with your unique_tags

max_len = 50  # Same as during training

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    sentence = data.get("sentence", "")

    # Preprocess the sentence
    sentence_words = sentence.split()
    sentence_indices = [word2idx.get(word, word2idx["<UNK>"]) for word in sentence_words]
    padded_sentence = pad_sequences([sentence_indices], maxlen=max_len, padding="post")

    # Predict the tags
    p = model.predict(padded_sentence)
    p = np.argmax