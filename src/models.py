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