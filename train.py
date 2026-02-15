import torch
import pandas as pd
import os
from model import TinyGPT
from config import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load CSV
df = pd.read_csv("mental_chat_data.csv")

corpus = []
for _, row in df.iterrows():
    convo = f"User: {row['Context']} Bot: {row['Response']} <END>"
    corpus.append(convo)

text = " ".join(corpus)

# Vocabulary
words = sorted(list(set(text.split())))
vocab_size = len(words)

word2idx = {w: i for i, w in enumerate(words)}
idx2word = {i: w for w, i in word2idx.items()}

data = torch.tensor([word2idx[w] for w in text.split()], dtype=torch.long)

# Split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - config["block_size"], (config["batch_size"],))
    x = torch.stack([d[i:i+config["block_size"]] for i in ix])
    y = torch.stack([d[i+1:i+config["block_size"]+1] for i in ix])
    return x.to(device), y.to(device)

model = TinyGPT(vocab_size, config).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config["learning_rate"],
    weight_decay=config["weight_decay"]
)

for step in range(config["epochs"]):
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 200 == 0:
        with torch.no_grad():
            xb, yb = get_batch("val")
            _, val_loss = model(xb, yb)
        print(f"Step {step} | Train {loss.item():.4f} | Val {val_loss.item():.4f}")

torch.save({
    "model": model.state_dict(),
    "word2idx": word2idx,
    "idx2word": idx2word
}, "tinygpt_chat.pth")

print("Model Saved!")
