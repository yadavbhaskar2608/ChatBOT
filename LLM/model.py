import torch
import torch.nn as nn

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_heads=16):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(512, embed_dim)

        # Transformer block
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx):
        B, T = idx.shape

        # positional ids
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)

        # embeddings
        x = self.token_emb(idx) + self.pos_emb(pos)

        # --------- CAUSAL MASK (important!) ---------
        mask = torch.triu(torch.ones(T, T, device=idx.device), diagonal=1).bool()

        # attention block
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm,
            attn_mask=mask
        )
        x = x + attn_out

        # feed-forward block
        x_norm = self.ln2(x)
        ff_out = self.ff(x_norm)
        x = x + ff_out

        logits = self.lm_head(x)
        return logits
