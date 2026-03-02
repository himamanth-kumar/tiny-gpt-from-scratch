🚀 What’s New in TinyGPT v2.0

TinyGPT 2.0 introduces significant improvements over v1, focusing on better generalization, stability, and generation quality.

🔄 Tokenization Upgrade

Replaced word-level tokenization with SentencePiece (BPE)

Reduced vocabulary fragmentation

Improved handling of unseen words

Significantly reduced overfitting

📐 Model Capacity Increased

Embedding dimension increased: 32 → 48

Transformer layers increased: 2 → 3

Attention heads increased: 2 → 3

Context window expanded: 16 → 24

📊 Training Improvements

Validation loss improved from ~3.0–3.5 → ~1.82

Training became more stable

Reduced train–validation gap

Better convergence behavior

🎲 Generation Improvements

Added Top-K sampling

Tuned temperature (0.6–0.8 range)

Improved response coherence

Reduced broken / fragmented outputs

🧱 Architectural Clean-Up

Fully config-driven architecture

Improved modular structure

Cleaner tokenizer-model integration
