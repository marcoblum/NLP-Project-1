import re
from collections import Counter
import torch
from torch.nn.utils.rnn import pad_sequence


def tokenize(text):
    """
    Zerlegt einen Text in Wörter und Satzzeichen.
    Alle Wörter werden in Kleinbuchstaben umgewandelt.
    """
    text = text.lower()
    tokens = re.findall(r"\w+|[^\w\s]", text)
    return tokens

def build_vocab(dataset, min_freq=1):
    """
    Erstellt ein Vokabular aus den Trainingsdaten.
    Tokens mit einer Häufigkeit unter min_freq werden ausgeschlossen.
    """

    columns = [
        "observation_1",
        "observation_2",
        "hypothesis_1",
        "hypothesis_2"
    ]

    word_counts = Counter()

    for example in dataset:
        for column in columns:
            word_counts.update(tokenize(example[column]))

    vocab = {
        "<PAD>": 0,
        "<UNK>": 1
    }

    for word, count in word_counts.items():
        if count >= min_freq:
            vocab[word] = len(vocab)

    return vocab


def encode(text, vocab):
    """
    Wandelt einen Text mithilfe des Vokabulars in Token-IDs um.
    Unbekannte Tokens erhalten die ID von <UNK>.
    """
    tokens = tokenize(text)

    token_ids = [
        vocab.get(token, vocab["<UNK>"])
        for token in tokens
    ]

    return token_ids


def encode_example(example, vocab):
    """
    Kodiert alle vier Texte eines ART-Beispiels
    und wandelt das Label von 1/2 in 0/1 um.
    """
    return {
        "observation_1": encode(example["observation_1"], vocab),
        "observation_2": encode(example["observation_2"], vocab),
        "hypothesis_1": encode(example["hypothesis_1"], vocab),
        "hypothesis_2": encode(example["hypothesis_2"], vocab),
        "label": int(example["label"]) - 1
    }


def collate_batch(batch, pad_id=0):
    """
    Fasst mehrere kodierte ART-Beispiele zu einem Batch zusammen.
    Unterschiedlich lange Sequenzen werden mit Padding ergänzt.
    """
    columns = [
        "observation_1",
        "observation_2",
        "hypothesis_1",
        "hypothesis_2"
    ]

    result = {}

    for column in columns:
        sequences = [
            torch.tensor(example[column], dtype=torch.long)
            for example in batch
        ]

        result[column] = pad_sequence(
            sequences,
            batch_first=True,
            padding_value=pad_id
        )

    result["label"] = torch.tensor(
        [example["label"] for example in batch],
        dtype=torch.long
    )

    return result