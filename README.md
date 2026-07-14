# Arabic Hotel Review Sentiment Analysis (NLP)

A three-class sentiment classifier (Negative / Neutral / Positive) for Arabic hotel reviews, built on a frozen AraBERT backbone with a custom Bahdanau-style additive attention layer and a lightweight classifier head. Developed as an NLP coursework project.



## Overview

- **Task:** Multi-class sentiment classification on Arabic hospitality reviews
- **Dataset:** [HARD — Hotel Arabic-Reviews Dataset](https://huggingface.co/datasets/Elnagara/hard) (93,700 Booking.com reviews, MSA + regional dialects), using a QA-corrected 22,500-sample working subset
- **Backbone:** `aubmindlab/bert-base-arabertv02` (136M params, fully frozen, used as a feature extractor)
- **Custom components:** Hand-coded Bahdanau-style additive attention layer + 2-layer MLP classifier head (~394K trainable parameters total)
- **Bonus phase:** 500 synthetically generated "hard" Arabic reviews (sarcasm, mixed-sentiment, dialect-heavy) via few-shot LLM prompting, injected to improve robustness

## Pipeline

1. **Data Engineering** — load HARD from HuggingFace, map star ratings → sentiment, run an 11-step Arabic text normalization pipeline (HTML/URL/emoji stripping, diacritic removal, Alef/Ya/Ta-Marbuta normalization, AraBERT-specific preprocessing).
2. **Custom Architecture** — frozen AraBERT → custom additive attention (produces a context vector + interpretable attention weights) → Linear(768→256) → ReLU → Dropout(0.3) → Linear(256→3).
3. **Training** — AdamW (lr=2e-4, weight_decay=0.01), class-weighted CrossEntropyLoss, ReduceLROnPlateau, EarlyStopping (patience=5), gradient clipping.
4. **Evaluation** — held-out stratified test set, Macro F1 as primary metric (mandatory given class imbalance), confusion matrices, attention visualization on individual predictions.
5. **Bonus: Synthetic Data Injection** — 12-category synthetic review generation (sarcasm, buried complaints, dialect-heavy negatives, understated positives, etc.), cleaned with the same pipeline, used to continue fine-tuning from the best checkpoint.

## Results

| Metric | Original | +Synthetic |
|---|---|---|
| Macro F1 | 0.9153 | 0.9163 |
| Accuracy | 0.9157 | 0.9162 |
| F1 — Negative | 0.9505 | 0.9503 |
| F1 — Neutral | 0.8738 | 0.8793 |
| F1 — Positive | 0.9216 | 0.9194 |

Neutral was the most challenging class (fewest samples, ambiguous mixed-sentiment language); synthetic data injection gave it the largest improvement, as intended.

## Repository Structure

```
.
├── NLP_Project.ipynb          # Full pipeline: data prep → training → evaluation → bonus phase
├── NLP_Technical_Report.pdf   # Full write-up with methodology, results, and analysis
├── assets/                    # Saved plots (class distribution, training curves, confusion matrix, attention viz)
└── README.md
```

> **Note:** Trained model weights (`best_model.pt`, `best_model_aug.pt`) and the AraBERT tokenizer cache are not included in this repository due to file size (the full checkpoint includes the frozen 136M-parameter backbone, ~540MB). Re-run the notebook to reproduce them, or see [Reproducing](#reproducing) below.

## Setup

```bash
pip install -q datasets transformers arabert pyarabic scikit-learn matplotlib seaborn tqdm torch
```

## Reproducing

Open `NLP_Project.ipynb` in Google Colab or Jupyter and run all cells in order. The notebook will:
1. Download the HARD dataset from HuggingFace automatically.
2. Train the model and save `best_model.pt` locally.
3. (Optional) Load a `synthetic_data.json` file of your own to run the bonus augmentation phase.

To reload a trained model for inference afterward:
```python
import json, torch
from transformers import AutoTokenizer

with open('model_config.json') as f:
    CFG = json.load(f)

tokenizer = AutoTokenizer.from_pretrained('arabert_tokenizer')
model = ArabicSentimentClassifier(
    model_name=CFG['MODEL_NAME'],
    hidden_dim=CFG['HIDDEN_DIM'],
    attention_dim=CFG['ATT_DIM'],
    dropout=CFG['DROPOUT'],
    num_classes=CFG['NUM_CLASSES']
)
model.load_state_dict(torch.load('best_model.pt', map_location='cpu'))
model.eval()
```

## License

Academic project — for coursework purposes.
