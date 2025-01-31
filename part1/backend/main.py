import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

PICKLE_NAME = "model.pkl"

# Load model and associated data from pickle
with open(PICKLE_NAME, "rb") as f:
    data = pickle.load(f)

model = data["model"]
model.eval()  # set to evaluation mode

# Check if it's a Hugging Face model
huggingface_mode = "tokenizer" in data
tokenizer = data["tokenizer"] if huggingface_mode else None
word_to_idx = data.get("word_to_idx", None)

def simple_tokenize(text: str, word_to_idx: dict):
    """
    Tokenise text by lowercasing and splitting on whitespace.
    Map each token to its ID in word_to_idx; use [UNK] if not found.
    """
    tokens = text.lower().split()
    unk_idx = word_to_idx.get("[UNK]", 0)
    return [word_to_idx.get(t, unk_idx) for t in tokens]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

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
        # Simple whitespace-based tokenization for custom RNN/LSTM
        token_ids = simple_tokenize(payload.text, word_to_idx)
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        lengths = torch.tensor([len(token_ids)])
        with torch.no_grad():
            # For custom LSTM or RNN expecting (input_ids, lengths)
            outputs = model(input_ids, lengths)
            predicted_class = torch.argmax(outputs, dim=-1).item()

    return {"prediction": predicted_class}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
