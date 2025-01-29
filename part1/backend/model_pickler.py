import os
import pandas as pd
import torch
import pickle
import gensim.downloader as api

from models import (
    TextDataset,
    CustomRNN,
    CustomLSTM,
    train_model,
    load_huggingface_model,
    build_vocab_and_matrix
)

PICKLE_NAME = "model.pkl"
LSTM_WEIGHTS = "lstm_model.pt"
RNN_WEIGHTS = "rnn_model.pt"
TRAINING_DATA = "training_data.csv"


def main():
    model_type = input(
        "Enter model type (transformer/rnn/lstm): ").strip().lower()

    if model_type == "transformer":
        # Load pre-trained HF tokenizer/model
        tokenizer, model = load_huggingface_model()
        with open(PICKLE_NAME, "wb") as f:
            pickle.dump({"tokenizer": tokenizer, "model": model}, f)
        print("Hugging Face model pickled successfully.")

    elif model_type == "rnn":
        # Step 1: Load Data and Tokenizer
        df = pd.read_csv(TRAINING_DATA)
        texts = df["text"].tolist()
        w2v = api.load("word2vec-google-news-300")

        word_to_idx, _, emb_matrix = build_vocab_and_matrix(
            texts, w2v, emb_dim=300)

        model = CustomRNN(emb_matrix=emb_matrix, hidden_dim=64, num_classes=5)

        # Step 2: Load Custom Model Weights / Train Custom Model
        if os.path.exists(RNN_WEIGHTS):
            print("rnn weights found. Loading and pickling...")
            model.load_state_dict(torch.load(RNN_WEIGHTS))

        else:
            print("No rnn weights found. Training from CSV...")
            labels = df["label"].tolist()

            dataset = TextDataset(texts, labels, word_to_idx)
            model = train_model(model, dataset, epochs=4, batch_size=4)
            torch.save(model.state_dict(), RNN_WEIGHTS)

            print('Custom RNN model trained successfully.')

        # Step 3: Save Model to Pickle
        with open(PICKLE_NAME, "wb") as f:
            pickle.dump({
                "model": model,
                "word_to_idx": word_to_idx,
                "emb_matrix": emb_matrix
            }, f)
        print("Custom RNN model pickled successfully.")

    elif model_type == "lstm":
        # Step 1: Load Data & Tokenizer
        df = pd.read_csv(TRAINING_DATA)
        texts = df["text"].tolist()
        w2v = api.load("word2vec-google-news-300")

        word_to_idx, _, emb_matrix = build_vocab_and_matrix(
            texts, w2v, emb_dim=300)

        # Step 2: Load Custom Model Weights / Train Custom Model
        if os.path.exists(LSTM_WEIGHTS):
            print("LSTM weights found. Loading and pickling...")
            
            model = CustomLSTM(emb_matrix=emb_matrix,
                               hidden_dim=64, num_classes=5)
            model.load_state_dict(torch.load(LSTM_WEIGHTS))

        else:
            print("No LSTM weights found. Training from CSV...")
            labels = df["label"].tolist()

            dataset = TextDataset(texts, labels, word_to_idx)
            model = CustomLSTM(emb_matrix=emb_matrix,
                               hidden_dim=64, num_classes=5)
            model = train_model(model, dataset, epochs=4, batch_size=4)

            torch.save(model.state_dict(), LSTM_WEIGHTS)
            print("Custom LSTM model trained successfully.")

        # Step 3: Save Model to Pickle
        with open(PICKLE_NAME, "wb") as f:
            pickle.dump({
                "model": model,
                "word_to_idx": word_to_idx,
                "emb_matrix": emb_matrix
            }, f)
        print("Custom LSTM model pickled successfully.")

    else:
        print("Invalid selection. Please choose from transformer, rnn, or lstm.")


if __name__ == "__main__":
    main()
