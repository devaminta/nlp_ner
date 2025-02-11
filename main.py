from fastapi import FastAPI, Request
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from fastapi.middleware.cors import CORSMiddleware  #
import json

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (replace with your frontend URL in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)
# Load the model
model = tf.keras.models.load_model("amharic_ner_model.h5")

# Load word2idx, tag2idx, unique_tags
with open("word2idx.json", "r") as f:
    word2idx = json.load(f)
with open("tag2idx.json", "r") as f:
    tag2idx = json.load(f)
with open("unique_tags.json", "r") as f:
    unique_tags = json.load(f)

max_len = 50  # Same as during training

# GET method to check API status
@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Amharic NER API!",
        "status": "API is live and running.",
        "usage": "Send a POST request to /predict with a JSON body containing a 'sentence' key."
    }

# POST method for prediction
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
    p = np.argmax(p, axis=-1)

    # Convert predictions to tag names
    predicted_tags = [unique_tags[tag] for tag in p[0]]

    # Map each word to its corresponding tag
    word_tag_pairs = [{"word": word, "tag": tag} for word, tag in zip(sentence_words, predicted_tags[:len(sentence_words)])]

    # Return results
    return {"sentence": sentence, "word_tag_pairs": word_tag_pairs}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)