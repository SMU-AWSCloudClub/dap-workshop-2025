import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import uvicorn

PICKLE_NAME = "model.pkl"
with open("model.pkl", "rb") as f:
    data = pickle.load(f)
tokenizer = data["tokenizer"]
model = data["model"]

app = FastAPI()

class TextPayload(BaseModel):
    text: str

@app.post("/predict")
def predict(payload: TextPayload):
    inputs = tokenizer(payload.text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=-1).item()
    return {"prediction": predicted_class}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
