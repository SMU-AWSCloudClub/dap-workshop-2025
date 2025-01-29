import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import uvicorn
import spacy

PICKLE_NAME = "model.pkl"
SPACY_WEB_CORE = "en_core_web_sm"
nlp = spacy.load(SPACY_WEB_CORE)

with open(PICKLE_NAME, "rb") as f:
    data = pickle.load(f)

model = data["model"]
model.eval()  # set to evaluation mode

# Check if it's a Hugging Face model
huggingface_mode = "tokenizer" in data
tokenizer = data["tokenizer"] if huggingface_mode else None
word_to_idx = data.get("word_to_idx", None)

def spacy_tokenize(text: str, word_to_idx: dict):
    """
    Tokenize text with spaCy. Return list of integer IDs.
    """
    doc = nlp(text.lower())
    tokens = [token.text for token in doc if not token.is_space and not token.is_punct]
    unk_idx = word_to_idx.get("[UNK]", 0)
    return [word_to_idx.get(t, unk_idx) for t in tokens]

app = FastAPI()

class TextPayload(BaseModel):
    text: str

@app.post("/predict")
def predict(payload: TextPayload):
    if huggingface_mode:
        # Hugging Face-style tokenization & inference
        inputs = tokenizer(payload.text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)  # e.g. a transformers model
            predicted_class = torch.argmax(outputs.logits, dim=-1).item()
    else:
        # SpaCy-based tokenization for custom RNN/LSTM
        token_ids = spacy_tokenize(payload.text, word_to_idx)
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        lengths = torch.tensor([len(token_ids)])

        with torch.no_grad():
            # For custom LSTM or RNN expecting (input_ids, lengths)
            outputs = model(input_ids, lengths)
            predicted_class = torch.argmax(outputs, dim=-1).item()

    return {"prediction": predicted_class}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
