# This code uses vscode regions to better organize the different models
# To collapse / open a section, simply click the dropdown arrow next to a # region comment

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from statistics import mean
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import numpy as np

# region Dataset

UNKNOWN_TOKEN = "[UNK]"

def simple_tokenize(text: str):
    """
    A simple whitespace-based tokenizer that lowercases the text
    and splits on whitespace.
    """
    return text.lower().split()

class TextDataset(Dataset):
    """
    Accepts the texts, labels, and word_to_idx that come from build_vocab_and_matrix.
    We utilise a simple whitespace-based tokeniser, then look up each token
    in our word_to_idx; if not found, use [UNK].
    """

    def __init__(self, texts, labels, word_to_idx):
        self.texts = texts
        self.labels = labels
        self.word_to_idx = word_to_idx
        self.unk_idx = self.word_to_idx[UNKNOWN_TOKEN]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = simple_tokenize(text)
        token_ids = [
            self.word_to_idx.get(t, self.unk_idx)
            for t in tokens
        ]
        label = self.labels[idx]
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)
# endregion

# region Embedding Matrix

def build_vocab_and_matrix(texts, w2v, emb_dim=300):
    """
    texts: list of strings
    w2v: a pretrained gensim Word2Vec model
    emb_dim: dimension of the word2vec embedding

    Returns:
        word_to_idx: dict mapping token -> index
        idx_to_word: list of tokens in index order
        emb_matrix: np array (vocab_size, emb_dim)
    """
    word_to_idx = {}
    idx_to_word = []
    vectors = []

    # Add special [UNK] token
    UNKNOWN_TOKEN = "[UNK]"
    word_to_idx[UNKNOWN_TOKEN] = 0
    idx_to_word.append(UNKNOWN_TOKEN)
    vectors.append(np.zeros(emb_dim))

    for text in texts:
        tokens = simple_tokenize(text)
        for token in tokens:
            if token in word_to_idx:
                continue
            if token in w2v:  # only add if in Word2Vec
                word_to_idx[token] = len(idx_to_word)
                idx_to_word.append(token)
                vectors.append(w2v[token])
            else:
                pass  # map to [UNK] at runtime

    # Build final embedding matrix
    emb_matrix = np.zeros((len(idx_to_word), emb_dim))
    for i, vec in enumerate(vectors):
        emb_matrix[i] = vec

    return word_to_idx, idx_to_word, emb_matrix

# endregion

# region Custom RNN

class CustomRNN(nn.Module):
    """
    Simple RNN for demonstration (embedding -> RNN -> dropout -> linear) with dynamic token lengths.
    """

    def __init__(self, emb_matrix, hidden_dim=64, num_classes=5, dropout_prob=0.5):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(emb_matrix),
            freeze=False
        )
        embedding_dim = emb_matrix.shape[1]
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.dropout_in = nn.Dropout(dropout_prob)
        self.dropout_out = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, padded_inputs, lengths):
        embedded = self.embedding(padded_inputs)
        packed_inputs = pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        _, hidden = self.rnn(packed_inputs)
        dropped = self.dropout_out(hidden[-1])
        return self.fc(dropped)

# endregion

# region Custom LSTM

class CustomLSTM(nn.Module):
    """ Simple LSTM for demonstration (embedding -> LSTM -> dropout -> linear). """

    def __init__(self, emb_matrix, hidden_dim=64, num_classes=5, dropout_prob=0.5):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(emb_matrix),
            freeze=False
        )
        embedding_dim = emb_matrix.shape[1]
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout_in = nn.Dropout(dropout_prob)
        self.dropout_out = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        embedded = self.dropout_in(embedded)
        packed = pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        dropped = self.dropout_out(hidden[-1])
        return self.fc(dropped)

# endregion

# region Training helpers

def train_model(model, dataset, epochs=2, batch_size=2):
    dataloader = DataLoader(
        dataset,
        batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=dynamic_collate_fn
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.05)

    for epoch in range(epochs):
        total_loss = 0.0
        for inputs, lengths, targets in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            outputs = model(inputs, lengths)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} — avg loss={avg_loss:.4f}")
    return model

def dynamic_collate_fn(batch):
    batch.sort(key=lambda x: x[0].size(0), reverse=True)
    sequences, labels = zip(*batch)
    lengths = [seq.size(0) for seq in sequences]
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return padded_sequences, torch.tensor(lengths), labels

def inference(model, text, word_to_idx):
    """
    Inference example for the custom RNN/LSTM models with the simple tokenizer.
    """
    tokens = [
        word_to_idx[token] if token in word_to_idx else word_to_idx[UNKNOWN_TOKEN]
        for token in text.lower().split()
    ]
    inputs = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        logits = model(inputs, [len(tokens)])
        prediction = torch.argmax(logits, dim=-1).item()
    return prediction

# endregion

# region Hugging Face model

def load_huggingface_model():
    tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    return tokenizer, model

def huggingface_inference(tokenizer, model, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    return torch.argmax(outputs.logits, dim=-1).item()

# endregion
