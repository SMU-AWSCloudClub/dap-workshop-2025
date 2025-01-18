# This code uses vscode regions to better organize the different models
# To collapse / open a section, simply click the dropdown arrow next to a # region comment

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# region Dataset

UNKNOWN_TOKEN = "[UNK]"
class TextDataset(Dataset):
    """
    Accepts the texts, labels, and word_to_idx that come from build_vocab_and_matrix.
    We do simple whitespace splitting on each text, then look up each token
    in our word_to_idx; if not found, use [UNK].
    """

    def __init__(self, texts, labels, word_to_idx):
        self.texts = texts
        self.labels = labels
        self.word_to_idx = word_to_idx

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.tokenize(self.texts[idx])
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

    def tokenize(self, text):
        return [
            self.word_to_idx[token] if token in self.word_to_idx else self.word_to_idx[UNKNOWN_TOKEN]
            for token in text.lower().split()
        ]
# endregion

# region Embedding Matrix


def build_vocab_and_matrix(texts, w2v, emb_dim=300):
    """
    texts: list of strings
    w2v: gensim word2vec model (pretrained)
    emb_dim: dimension of the word2vec embedding (e.g. 300)

    Returns:
        word_to_idx (dict): token -> integer index
        idx_to_word (list): index -> token
        emb_matrix (np.array): shape (vocab_size, emb_dim),
                               containing the vectors for each index
    """
    word_to_idx = {}
    idx_to_word = []
    vectors = []

    # Add a special [UNK] token for words not in Word2Vec
    word_to_idx[UNKNOWN_TOKEN] = 0
    idx_to_word.append(UNKNOWN_TOKEN)
    vectors.append(np.zeros(emb_dim))

    # Build the vocab by iterating over every token in your dataset
    # Only add if it's in the Word2Vec model
    # Otherwise we ignore it, and it will map to [UNK]
    for text in texts:
        for token in text.lower().split():
            if token in word_to_idx or token in w2v:
                continue

            word_to_idx[token] = len(idx_to_word)
            idx_to_word.append(token)
            vectors.append(w2v[token])

    # Build the final embedding matrix
    emb_matrix = np.zeros((len(idx_to_word), emb_dim))
    for i, vec in enumerate(vectors):
        emb_matrix[i] = vec

    return word_to_idx, idx_to_word, emb_matrix
# endregion

# region Custom CNN


class CustomCNN(nn.Module):
    """ Simple CNN for demonstration (embedding -> conv -> pool -> linear). """

    def __init__(self, emb_matrix, num_classes=5):
        super().__init__()
        # Create an embedding layer from the pretrained matrix
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(emb_matrix),
            freeze=True
        )
        embedding_dim = emb_matrix.shape[1]
        self.conv = nn.Conv1d(in_channels=embedding_dim,
                              out_channels=16, kernel_size=3)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(16, num_classes)

    # x shape: (batch_size, seq_len)
    def forward(self, x):
        embedded = self.embedding(x).permute(
            0, 2, 1)  # (batch, embed_dim, seq_len)
        # (batch, out_channels, new_seq_len)
        conved = self.conv(embedded)
        pooled = self.pool(conved).squeeze(-1)      # (batch, out_channels)
        return self.fc(pooled)
# endregion

# region Custom LSTM


class CustomLSTM(nn.Module):
    """ Simple LSTM for demonstration (embedding -> LSTM -> linear). """

    def __init__(self, emb_matrix, hidden_dim=64, num_classes=5):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(emb_matrix),
            freeze=False  # Let the LSTM fine-tune embeddings
        )
        embedding_dim = emb_matrix.shape[1]
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        # (batch, seq_len, embedding_dim)
        embedded = self.embedding(x)
        # hidden: (num_layers, batch, hidden_dim)
        _, (hidden, _) = self.lstm(embedded)
        return self.fc(hidden[-1])                   # (batch, num_classes)
# endregion

# region Training helpers


def train_model(model, dataset, epochs=2, batch_size=2):
    """
    Simple training loop with Adam optimizer + weight decay.
    Shows progress with TQDM.
    """
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.1)

    for epoch in range(epochs):
        for inputs, targets in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    return model


def inference(model, text, word_to_idx):
    """
    Inference example for the custom CNN/LSTM models.
    """
    tokens = [
        word_to_idx[token] if token in word_to_idx else word_to_idx[UNKNOWN_TOKEN]
        for token in text.lower().split()
    ]
    inputs = torch.tensor(tokens, dtype=torch.long).unsqueeze(
        0)  # shape (1, seq_len)
    with torch.no_grad():
        logits = model(inputs)
        prediction = torch.argmax(logits, dim=-1).item()
    return prediction
# endregion

# region Hugging Face model


def load_huggingface_model():
    """
    Loads a pretrained Hugging Face model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        "nlptown/bert-base-multilingual-uncased-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained(
        "nlptown/bert-base-multilingual-uncased-sentiment")
    return tokenizer, model


def huggingface_inference(tokenizer, model, text):
    """
    Inference example for the Hugging Face model.
    """
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, padding=True)
    outputs = model(**inputs)
    return torch.argmax(outputs.logits, dim=-1).item()
# endregion
