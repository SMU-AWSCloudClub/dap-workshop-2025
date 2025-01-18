import os
import pandas as pd
import torch
import pickle
import gensim.downloader as api

from models import (
    TextDataset,
    CustomCNN,
    CustomLSTM,
    train_model,
    load_huggingface_model,
    build_vocab_and_matrix
)

PICKLE_NAME = "model.pkl"
LSTM_WEIGHTS = "lstm_model.pt"
CNN_WEIGHTS = "cnn_model.pt"
TRAINING_DATA = "training_data.csv"


def main():
    model_type = input(
        "Enter model type (hugging_face/cnn/lstm): ").strip().lower()

    if model_type == "hugging_face":
        # Load pre-trained HF tokenizer/model
        tokenizer, model = load_huggingface_model()
        with open(PICKLE_NAME, "wb") as f:
            pickle.dump({"tokenizer": tokenizer, "model": model}, f)
        print("Hugging Face model pickled successfully.")

    elif model_type == "cnn":
        if os.path.exists(CNN_WEIGHTS):
            print("CNN weights found. Loading and pickling...")
            # Load CSV to build the same vocab (not strictly necessary if word_to_idx unchanged)
            df = pd.read_csv(TRAINING_DATA)
            texts = df["text"].tolist()
            w2v = api.load("word2vec-google-news-300")

            # Build the embedding matrix so we can init the same architecture
            word_to_idx, idx_to_word, emb_matrix = build_vocab_and_matrix(
                texts, w2v, emb_dim=300)

            # Create model and load existing weights
            model = CustomCNN(emb_matrix=emb_matrix, num_classes=5)
            model.load_state_dict(torch.load(CNN_WEIGHTS))

            # Pickle the loaded model
            with open(PICKLE_NAME, "wb") as f:
                pickle.dump({
                    "model": model,
                    "word_to_idx": word_to_idx,
                    "emb_matrix": emb_matrix
                }, f)
            print("CNN model pickled successfully.")

        else:
            print("No CNN weights found. Training from CSV...")
            df = pd.read_csv(TRAINING_DATA)
            texts = df["text"].tolist()
            labels = df["label"].tolist()

            # Load Word2Vec, build vocab and embedding matrix
            w2v = api.load("word2vec-google-news-300")
            word_to_idx, idx_to_word, emb_matrix = build_vocab_and_matrix(
                texts, w2v, emb_dim=300)

            # Create dataset and model
            dataset = TextDataset(texts, labels, word_to_idx)
            model = CustomCNN(emb_matrix=emb_matrix, num_classes=5)

            # Train
            trained_model = train_model(model, dataset, epochs=2, batch_size=2)

            # Save weights
            torch.save(trained_model.state_dict(), CNN_WEIGHTS)

            # Pickle
            with open(PICKLE_NAME, "wb") as f:
                pickle.dump({
                    "model": trained_model,
                    "word_to_idx": word_to_idx,
                    "emb_matrix": emb_matrix
                }, f)
            print("Custom CNN model trained and pickled successfully.")

    elif model_type == "lstm":
        if os.path.exists(LSTM_WEIGHTS):
            print("LSTM weights found. Loading and pickling...")
            df = pd.read_csv(TRAINING_DATA)
            texts = df["text"].tolist()
            w2v = api.load("word2vec-google-news-300")

            word_to_idx, idx_to_word, emb_matrix = build_vocab_and_matrix(
                texts, w2v, emb_dim=300)

            model = CustomLSTM(emb_matrix=emb_matrix,
                               hidden_dim=64, num_classes=5)
            model.load_state_dict(torch.load(LSTM_WEIGHTS))

            with open(PICKLE_NAME, "wb") as f:
                pickle.dump({
                    "model": model,
                    "word_to_idx": word_to_idx,
                    "emb_matrix": emb_matrix
                }, f)
            print("Custom LSTM model pickled successfully.")

        else:
            print("No LSTM weights found. Training from CSV...")
            df = pd.read_csv(TRAINING_DATA)
            texts = df["text"].tolist()
            labels = df["label"].tolist()

            w2v = api.load("word2vec-google-news-300")
            word_to_idx, idx_to_word, emb_matrix = build_vocab_and_matrix(
                texts, w2v, emb_dim=300)

            dataset = TextDataset(texts, labels, word_to_idx)
            model = CustomLSTM(emb_matrix=emb_matrix,
                               hidden_dim=64, num_classes=5)
            trained_model = train_model(model, dataset, epochs=2, batch_size=2)

            torch.save(trained_model.state_dict(), LSTM_WEIGHTS)

            with open(PICKLE_NAME, "wb") as f:
                pickle.dump({
                    "model": trained_model,
                    "word_to_idx": word_to_idx,
                    "emb_matrix": emb_matrix
                }, f)
            print("Custom LSTM model trained and pickled successfully.")

    else:
        print("Invalid selection. Please choose from hugging_face, cnn, or lstm.")


if __name__ == "__main__":
    main()
