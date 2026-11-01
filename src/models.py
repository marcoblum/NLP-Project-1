import torch
import torch.nn as nn


class WordEmbeddingModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim=50, dropout=0.0):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 3, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def mean_pooling(self, token_ids):
        # Token-IDs in Embeddings umwandeln
        embeddings = self.embedding(token_ids)

        # Maske erstellen: 1 für echte Tokens, 0 für Padding
        mask = (token_ids != 0).unsqueeze(-1).float()

        # Padding-Embeddings ausschliessen
        masked_embeddings = embeddings * mask

        # Embeddings aufsummieren
        summed_embeddings = masked_embeddings.sum(dim=1)

        # Anzahl echter Tokens bestimmen
        token_counts = mask.sum(dim=1).clamp(min=1)

        # Durchschnitt berechnen
        return summed_embeddings / token_counts

    def combine_vectors(self, batch):
        # Satzvektoren für alle vier Textfelder berechnen
        obs1 = self.mean_pooling(batch["observation_1"])
        obs2 = self.mean_pooling(batch["observation_2"])
        hyp1 = self.mean_pooling(batch["hypothesis_1"])
        hyp2 = self.mean_pooling(batch["hypothesis_2"])

        # Beobachtungen mit der jeweiligen Hypothese verbinden
        combination_1 = torch.cat([obs1, obs2, hyp1], dim=1)
        combination_2 = torch.cat([obs1, obs2, hyp2], dim=1)

        return combination_1, combination_2

    def forward(self, batch):
        # Beobachtungen und Hypothesen kombinieren
        combination_1, combination_2 = self.combine_vectors(batch)

        # Beide Kombinationen bewerten
        score_1 = self.classifier(combination_1)
        score_2 = self.classifier(combination_2)

        # Scores zu einem Tensor zusammenfügen
        logits = torch.cat([score_1, score_2], dim=1)

        return logits


class RNNModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=50,
        hidden_dim=64,
        dropout=0.0
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        self.rnn = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def encode(self, tokens):
        # Anzahl tatsächlicher Wörter (ohne Padding)
        lengths = (tokens != 0).sum(dim=1).clamp(min=1)

        embeddings = self.embedding(tokens)

        packed = nn.utils.rnn.pack_padded_sequence(
            embeddings,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, hidden = self.rnn(packed)

        return hidden[-1]

    def forward(self, batch):
        o1 = self.encode(batch["observation_1"])
        o2 = self.encode(batch["observation_2"])
        h1 = self.encode(batch["hypothesis_1"])
        h2 = self.encode(batch["hypothesis_2"])

        features_1 = torch.cat([o1, o2, h1], dim=1)
        features_2 = torch.cat([o1, o2, h2], dim=1)

        score_1 = self.classifier(features_1)
        score_2 = self.classifier(features_2)

        return torch.cat([score_1, score_2], dim=1)