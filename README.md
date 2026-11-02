# NLP-Project-1
Abductive Commonsense Reasoning using Word Embeddings and RNNs

## Project Overview
This project addresses **Abductive Natural Language Inference (αNLI)** on the dataset [`amarfurt/art-nlp-hs26`](https://huggingface.co/datasets/amarfurt/art-nlp-hs26) (based on ART). Each example consists of two observations (`observation_1`, `observation_2`) and two hypotheses (`hypothesis_1`, `hypothesis_2`). The task is to select the hypothesis that more plausibly explains how the first observation leads to the second.

Two neural models are implemented in PyTorch and compared:

1. A **Word-Embedding model** with masked mean pooling

2. An **RNN model** based on a GRU

The evaluation metric is accuracy. Since there are two candidate hypotheses, random guessing yields about 50%. The expectations defined before training are documented in [`EXPECTATIONS.md`](EXPECTATIONS.md).

## Dataset
| Split | Examples | Usage |
|---|---:|---|
| Train | 16,801 | Training, vocabulary construction |
| Validation | 1,000 | Model selection, early stopping, hyperparameter search |
| Test | 1,532 | Final evaluation only (used once, in `04_final_evaluation.ipynb`) |
Labels are `1` or `2` (index of the correct hypothesis) and are mapped to `0` / `1` for PyTorch. All splits are approximately balanced.

## Preprocessing
All preprocessing functions are implemented in [`src/preprocessing.py`](src/preprocessing.py) and shared by all notebooks:

- **Tokenization** (`tokenize`): lowercasing and splitting into words and punctuation with the regex `\w+|[^\w\s]`.

- **Vocabulary** (`build_vocab`): built **only from the training data** with `min_freq=1` (15,818 tokens). Special tokens: `<PAD>` = 0, `<UNK>` = 1.

- **Unknown words** (`encode`): tokens not in the vocabulary are mapped to `<UNK>`. In the test set, 1.45% of all tokens are unknown.

- **Encoding** (`encode_example`): encodes all four texts of an example and maps the label to `0` / `1`.

- **Padding** (`collate_batch`): pads variable-length sequences within a batch with `<PAD>` (ID 0). Padding is ignored by the masked mean pooling and by the packed sequences of the GRU.

For the final evaluation, the vocabulary is loaded from the model checkpoints, guaranteeing exactly the same word-to-index mapping as during training.

## Models
Both models are defined in [`src/models.py`](src/models.py). They share the same overall structure: each of the four texts is encoded into a vector, the two observations are concatenated with each hypothesis, and a shared feed-forward classifier produces one score per hypothesis. The two scores form the logits `[batch_size, 2]`, trained with `CrossEntropyLoss`.
| | `WordEmbeddingModel` | `RNNModel` |
|---|---|---|
| Embedding | 50 dimensions, trainable, `padding_idx=0` | 50 dimensions, trainable, `padding_idx=0` |
| Text encoder | Masked mean pooling | GRU (hidden size 64, packed sequences), last hidden state |
| Text vector | 50 dimensions | 64 dimensions |
| Classifier input | `[obs_1; obs_2; hyp_i]` = 150 | `[obs_1; obs_2; hyp_i]` = 192 |
| Classifier | Linear → ReLU → Dropout → Linear (hidden size 64) | Linear → ReLU → Dropout → Linear (hidden size 64) |
| Parameters | 800,629 | 825,589 |
Mean pooling ignores word order, while the GRU processes tokens sequentially and can take word order and context into account.

## Experiments
Training is implemented in [`src/training.py`](src/training.py). The function `run_experiment()` trains either model (`model_type="word_embeddings"` or `"rnn"`) with:

- **Optimizer:** AdamW

- **Early stopping** on validation accuracy (patience 5; patience 3 in the first short runs), max. 60 epochs

- **Checkpointing:** the model state with the highest validation accuracy, including the vocabulary and hyperparameters, is saved to `outputs/checkpoints/`

- **Result caching:** results and training logs are stored as JSON in `outputs/results/`; experiments that were already run (with an existing checkpoint) are loaded instead of retrained

- **Experiment tracking:** each run is logged to Weights & Biases (W&B)

- **Reproducibility:** fixed seeds for Python, NumPy, PyTorch and data shuffling

**Hyperparameter search (Word Embeddings, `02_word_embeddings_model.ipynb`):**
| Hyperparameter | Values tested | Selected |
|---|---|---|
| Batch size | 16, 32, 64, 128, 256, 512 | 256 |
| Learning rate | 1e-4, 3.16e-4, 1e-3, 3.16e-3, 1e-2 | 1e-3 |
| Dropout | 0.0, 0.2, 0.5 | 0.0 |
| Weight decay | 0.0, 0.001, 0.01, 0.1 | 0.001 |
| Seed (stability) | 42, 123, 456 | 42 |
**RNN experiments (`03_rnn_model.ipynb`):** Starting from the best Word-Embedding configuration (batch size 256, learning rate 1e-3, weight decay 0.001), dropout values 0.0, 0.2 and 0.5 were compared. The best configuration (dropout 0.5) was repeated with seeds 123 and 456.
| Model (best configuration) | Validation accuracy over 3 seeds |
|---|---|
| Word Embeddings | 66.23% ± 1.11 pp |
| RNN (GRU) | 65.10% ± 0.75 pp |
## Results
Final models selected by validation accuracy (seed 42), evaluated once on the held-out test set:
| Model | Best Validation Accuracy | Test Accuracy |
|---|---|---|
| Word Embeddings | 67.80% | 53.72% |
| RNN (GRU) | 66.10% | 53.13% |
## Discussion
Both models lose about 13–14 percentage points from validation to test and perform only slightly above chance on the test set. The RNN does not outperform the simpler Word-Embedding model, so the hypothesis from [`EXPECTATIONS.md`](EXPECTATIONS.md) is not confirmed, and both models remain below the expected test accuracies.

The evaluation pipeline was checked and is technically correct: the checkpoints reproduce the validation accuracies exactly, tokenization, vocabulary and label mapping are identical for all splits, and no observation pairs from the validation or test set occur in the training data.

A plausible explanation is a **distribution shift** between training/validation and test:
| | Train | Validation | Test |
|---|---:|---:|---:|
| Word overlap between the two hypotheses (mean Jaccard) | 0.60 | 0.61 | 0.22 |
| Minimal pairs (≤ 2 differing tokens) | 34.6% | 34.4% | 3.6% |
| Shorter hypothesis is correct | 57.6% | 57.6% | 50.1% |
In the training and validation data, the wrong hypothesis is often a slightly modified copy of the correct one (e.g., two hypotheses differing in only one word), and the shorter hypothesis is more often correct. Such lexical and length cues can be exploited by embedding-based models. Because the validation set shares these properties, validation accuracy overestimates generalization. In the test set, the two hypotheses are mostly independently written and similarly plausible, so these shortcuts no longer work.

In addition, both models overfit strongly (training accuracy above 90%), and model selection on a single small validation set adds optimistic bias. The test results were not used for any further tuning.

## Repository Structure
```

NLP-Project-1/

├── notebooks/

│   ├── 01_data_exploration.ipynb      # Data exploration, tokenization, vocabulary, padding, mean pooling

│   ├── 02_word_embeddings_model.ipynb # Word-Embedding model, training, hyperparameter search

│   ├── 03_rnn_model.ipynb             # RNN (GRU) model, dropout and seed experiments

│   └── 04_final_evaluation.ipynb      # Loading the best checkpoints, final test evaluation

├── src/

│   ├── preprocessing.py               # tokenize, build_vocab, encode, encode_example, collate_batch

│   ├── models.py                      # WordEmbeddingModel, RNNModel

│   └── training.py                    # train_one_epoch, validate, run_experiment

├── outputs/

│   ├── checkpoints/                   # Model checkpoints (*.pt), two final checkpoints tracked; other checkpoints ignored

│   └── results/                       # Cached experiment results and logs (*.json)

├── EXPECTATIONS.md                    # Expectations defined before training

├── requirements.txt

└── README.md

```
## How to Run
1. Create a virtual environment and install the dependencies:

   ```bash

   python -m venv .venv

   .venv\Scripts\activate          # Windows

   source .venv/bin/activate       # macOS / Linux

   pip install -r requirements.txt

   ```
   For GPU training, install a CUDA-enabled PyTorch build matching your system.

2. Log in to Weights & Biases (required for training runs):

   ```bash

   wandb login

   ```
3. Run the notebooks in `notebooks/` in order (`01` → `04`). The dataset is downloaded automatically from the Hugging Face Hub.

Training and GPU stress-test cells are disabled by default so that **Run All** does not start new training runs, GPU stress tests, or W&B runs. They can be enabled with the following switches:
| Notebook | Switch | Experiments |
|---|---|---|
| `02_word_embeddings_model.ipynb` | `RUN_BASELINE_DEMOS` | Introductory training demos (sections 2.10–2.18) |
| `02_word_embeddings_model.ipynb` | `RUN_EXPERIMENTS` | Hyperparameter search |
| `02_word_embeddings_model.ipynb` | `RUN_GPU_TESTS` | GPU memory and batch-size stress tests |
| `03_rnn_model.ipynb` | `RUN_RNN_TRAINING` | First RNN training run |
| `03_rnn_model.ipynb` | `RUN_RNN_DROPOUT` | RNN dropout experiments |
| `03_rnn_model.ipynb` | `RUN_RNN_SEEDS` | RNN seed experiments |
The two final model checkpoints are included in the repository, allowing `04_final_evaluation.ipynb` to be executed without retraining the models. Other experimental checkpoints are excluded through `.gitignore`. To reproduce the training experiments, enable the corresponding training switches in the notebooks. Experiments with an existing checkpoint and result file are loaded from the cache instead of being retrained.

## Experiment Tracking
All training runs are logged to Weights & Biases in the project `NLP-Project-1`.

W&B Saved View: [NLP-Project-1 Workspace](https://forge.coreweave.com/wandb/marco-blum-hochschule-luzern/NLP-Project-1/workspace?nw=tlv7kslidej)
