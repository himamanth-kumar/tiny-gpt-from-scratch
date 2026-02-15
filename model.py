import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Head(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.key = nn.Linear(config["embedding_dim"], config["embedding_dim"] // config["n_heads"], bias=False)
        self.query = nn.Linear(config["embedding_dim"], config["embedding_dim"] // config["n_heads"], bias=False)
        self.value = nn.Linear(config["embedding_dim"], config["embedding_dim"] // config["n_heads"], bias=False)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(config["block_size"], config["block_size"]))
        )

        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([Head(config) for _ in range(config["n_heads"])])
        self.proj = nn.Linear(config["embedding_dim"], config["embedding_dim"])
        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config["embedding_dim"], 4 * config["embedding_dim"]),
            nn.ReLU(),
            nn.Linear(4 * config["embedding_dim"], config["embedding_dim"]),
            nn.Dropout(config["dropout"])
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sa = MultiHeadAttention(config)
        self.ff = FeedForward(config)
        self.ln1 = nn.LayerNorm(config["embedding_dim"])
        self.ln2 = nn.LayerNorm(config["embedding_dim"])

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()

        self.block_size = config["block_size"]
        self.token_embedding = nn.Embedding(vocab_size, config["embedding_dim"])
        self.position_embedding = nn.Embedding(config["block_size"], config["embedding_dim"])

        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config["n_layers"])]
        )

        self.ln_f = nn.LayerNorm(config["embedding_dim"])
        self.head = nn.Linear(config["embedding_dim"], vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))

        return logits, loss

    def generate(self, idx, max_new_tokens, end_token_id=None, temperature=0.8):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)

            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)

            next_idx = torch.multinomial(probs, 1)
            idx = torch.cat((idx, next_idx), dim=1)

            if end_token_id is not None:
                if next_idx.item() == end_token_id:
                    break

        return idx
