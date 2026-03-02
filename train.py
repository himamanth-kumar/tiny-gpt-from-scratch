import torch
import pandas as pd
import os
import sentencepiece as spm

from model import TinyGPT
from config import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------
# 1️⃣ Load and Prepare Text
# -------------------------

df = pd.read_csv("mental_chat_data.csv")

corpus = []
for _, row in df.iterrows():
    convo = f"User: {row['Context']} Bot: {row['Response']} <END>"
    corpus.append(convo)

text = "\n".join(corpus)


with open("clean_text.txt", "w", encoding="utf-8") as f:
    f.write(text)

# -------------------------
# 2️⃣ Train SentencePiece
# -------------------------

if not os.path.exists("tokenizer.model"):
    spm.SentencePieceTrainer.train(
        input="clean_text.txt",
        model_prefix="tokenizer",
        vocab_size=188,
        model_type="bpe"
    )

sp = spm.SentencePieceProcessor()
sp.load("tokenizer.model")

# -------------------------
# 3️⃣ Encode Dataset
# -------------------------

ids = sp.encode(text, out_type=int)
data = torch.tensor(ids, dtype=torch.long)

vocab_size = sp.get_piece_size()
print("Vocab Size:", vocab_size)

# -------------------------
# 4️⃣ Train/Val Split
# -------------------------

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# -------------------------
# 5️⃣ Batch Function
# -------------------------

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(
        0,
        len(d) - config["block_size"],
        (config["batch_size"],)
    )

    x = torch.stack([d[i:i+config["block_size"]] for i in ix])
    y = torch.stack([d[i+1:i+config["block_size"]+1] for i in ix])

    return x.to(device), y.to(device)

# -------------------------
# 6️⃣ Model
# -------------------------

model = TinyGPT(vocab_size, config).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config["learning_rate"],
    weight_decay=config["weight_decay"]
)

# -------------------------
# 7️⃣ Training Loop
# -------------------------

for step in range(config["epochs"]):
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % 200 == 0:
        with torch.no_grad():
            xb, yb = get_batch("val")
            _, val_loss = model(xb, yb)

        print(f"Step {step} | Train {loss.item():.4f} | Val {val_loss.item():.4f}")

# -------------------------
# 8️⃣ Save Model
# -------------------------

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.model")

torch.save({
    "model": model.state_dict(),
    "tokenizer_model": TOKENIZER_PATH
}, os.path.join(BASE_DIR, "tinygpt_chat.pth"))

print("Model Saved!")