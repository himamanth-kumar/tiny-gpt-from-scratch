# 🚀 Tiny GPT From Scratch

A decoder-only Transformer (GPT-style) built completely from scratch using **PyTorch** — without pre-trained weights or HuggingFace Trainer.

This project focuses on understanding how GPT models work internally by implementing every core component manually.

---

## 🧠 Architecture Overview

The model includes:

- Word embeddings (`vocab_size × embedding_dim`)
- Positional embeddings
- Multi-head self-attention
- Causal masking (autoregressive constraint)
- Residual connections
- Layer Normalization
- Dropout regularization
- Linear output head for next-token prediction
- Temperature-based sampling
- `<END>` token stopping mechanism
- Streamlit-based interactive chatbot UI

> Core principle: GPT is fundamentally a next-token prediction model.

---

## 📊 Dataset

- Structured conversational dataset
- ~300–500 dialogue pairs
- CSV format

Due to limited dataset size, overfitting was observed during training.

---

## 📈 Training Performance

| Metric | Value |
|--------|--------|
| Initial Training Loss | ~4.1 |
| Final Training Loss | ~2.4–2.8 |
| Validation Loss | ~3.0–3.5 |

### Observations
- Clear overfitting on small dataset
- Improved stability after adding:
  - Dropout
  - Weight decay
- Limited generalization (expected due to small data)

---

## ⚙️ Key Technical Insights

- Embedding parameters = `vocab_size × embedding_dim`
- Changing vocabulary breaks checkpoint compatibility
- `block_size` must match between training and inference
- Causal masking is essential for autoregressive modeling
- Random token shuffling destroys language structure
- Training format must match inference prompt format
- Small datasets make validation loss unreliable
- Temperature directly controls generation entropy

---

## ⚠️ Current Limitations

- Word-level tokenization (no BPE yet)
- Limited context window
- Small dataset
- Pattern matching (not reasoning)
- Overfitting on limited data

---

## 🚀 Future Improvements (GPT 2.0 – My Version)

- Subword tokenization (BPE)
- Top-k / nucleus sampling
- Larger dataset
- Perplexity tracking
- Expanded context window
- Increased embedding dimension
- Character-level fallback

---
🎯 Project Goal

This is not a production-level LLM.

The objective was to:

Deeply understand Transformer internals

Implement GPT architecture manually

Experience real training instability

Move from “LLMs feel like magic” to “LLMs feel mechanical”

🏷️ Tech Stack

Python

PyTorch

Streamlit

## 🖥️ How to Run

```bash
# Clone repository
git clone https://github.com/yourusername/tiny-gpt-from-scratch.git
cd tiny-gpt-from-scratch

# Install dependencies
pip install -r requirements.txt

# Train the model
python train.py

# Launch Streamlit UI
streamlit run app.py
