import torch
from tokenizers import ByteLevelBPETokenizer
from model import TinyGPT
import torch.nn.functional as F

# ---------------- GPU SETUP ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------- LOAD TOKENIZER ----------------
tokenizer = ByteLevelBPETokenizer(
    "tokenizer/vocab.json",
    "tokenizer/merges.txt"
)

# ---------------- LOAD MODEL ----------------
model = TinyGPT(vocab_size=5000).to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# ---------------- SAFE SAMPLING FUNCTION ----------------
def sample_next_token(logits, temperature=0.7, top_k=40, top_p=0.9):

    # prevent INF or NAN logits
    logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

    # temperature scaling
    logits = logits / max(temperature, 1e-6)

    # convert to probabilities
    probs = F.softmax(logits, dim=-1)

    # If probs contains NaN or inf, fix:
    if torch.isnan(probs).any() or torch.isinf(probs).any():
        probs = torch.ones_like(probs) / probs.numel()

    # ---------------- TOP-K ----------------
    if top_k > 0:
        top_k = min(top_k, probs.size(-1))
        values, indices = torch.topk(probs, top_k)
        mask = torch.ones_like(probs, dtype=torch.bool)
        mask[indices] = False
        probs[mask] = 0
        if probs.sum() == 0:
            probs = F.softmax(logits, dim=-1)
        else:
            probs = probs / probs.sum()

    # ---------------- TOP-P ----------------
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    # nucleus mask
    nucleus_mask = cumulative > top_p

    # keep at least one token
    if nucleus_mask.sum() == cumulative.size(0):
        nucleus_mask[-1] = False  

    sorted_probs[nucleus_mask] = 0

    if sorted_probs.sum() == 0:
        probs = F.softmax(logits, dim=-1)
    else:
        sorted_probs = sorted_probs / sorted_probs.sum()
        probs = torch.zeros_like(probs)
        probs[sorted_indices] = sorted_probs

    # avoid zero distribution
    if probs.sum() == 0:
        probs = torch.ones_like(probs) / probs.numel()

    # final sampling
    next_token = torch.multinomial(probs, 1).item()
    return next_token

# ---------------- GENERATION FUNCTION ----------------
def generate_reply(message, max_tokens=80):
    prompt = f"User: {message}\nBot:"
    ids = tokenizer.encode(prompt).ids
    x = torch.tensor(ids).unsqueeze(0).to(device)

    generated = []

    for _ in range(max_tokens):
        logits = model(x)[:, -1, :]      # last token logits
        next_id = sample_next_token(logits[0])
        generated.append(next_id)

        next_tensor = torch.tensor([[next_id]], device=device)
        x = torch.cat([x, next_tensor], dim=1)

    # decode full text
    full_text = tokenizer.decode((ids + generated))

    # clean Bot response
    if "Bot:" in full_text:
        reply = full_text.split("Bot:", 1)[1].strip()
        if "User:" in reply:
            reply = reply.split("User:", 1)[0].strip()
    else:
        reply = full_text

    return reply

# ---------------- CHAT LOOP ----------------
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Bot: Goodbye!")
        break

    print("Bot:", generate_reply(user_input))
