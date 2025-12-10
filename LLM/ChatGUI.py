import tkinter as tk
from tkinter import scrolledtext
import torch
from tokenizers import ByteLevelBPETokenizer
from model import TinyGPT
import torch.nn.functional as F
import re

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# ---------------- LOAD TOKENIZER ----------------
tokenizer = ByteLevelBPETokenizer(
    "tokenizer/vocab.json",
    "tokenizer/merges.txt"
)

# ---------------- LOAD MODEL ----------------
model = TinyGPT(vocab_size=5000).to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# ---------------- SAMPLING ----------------
def sample_next_token(logits, temperature=0.7):
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()

# ---------------- GENERATION ----------------
def generate_reply(message, max_tokens=80):
    prompt = f"User: {message}\nBot:"
    ids = tokenizer.encode(prompt).ids

    x = torch.tensor(ids).unsqueeze(0).to(device)

    for _ in range(max_tokens):
        logits = model(x)
        next_id = sample_next_token(logits[0, -1])
        next_tensor = torch.tensor([[next_id]], device=device)
        x = torch.cat([x, next_tensor], dim=1)

    text = tokenizer.decode(x[0].cpu().tolist())

    # Clean tags if any appear
    text = re.sub(r"<.*?>", "", text)

    # Extract only bot output
    if "Bot:" in text:
        text = text.split("Bot:", 1)[1].strip()
    if "User:" in text:
        text = text.split("User:", 1)[0].strip()

    return text


# ---------------- GUI ----------------
def send_message():
    user_text = entry.get()
    if not user_text.strip():
        return

    chat_window.config(state=tk.NORMAL)
    chat_window.insert(tk.END, f"You: {user_text}\n", "user")
    
    entry.delete(0, tk.END)

    # Generate bot response
    bot_text = generate_reply(user_text)

    chat_window.insert(tk.END, f"Bot: {bot_text}\n\n", "bot")
    chat_window.config(state=tk.DISABLED)
    chat_window.yview(tk.END)


# ---------------- BUILD WINDOW ----------------
window = tk.Tk()
window.title("My LLM Chatbot")
window.geometry("600x700")

chat_window = scrolledtext.ScrolledText(window, wrap=tk.WORD, font=("Arial", 12))
chat_window.config(state=tk.DISABLED)
chat_window.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# Text tags for styling
chat_window.tag_config("user", foreground="blue")
chat_window.tag_config("bot", foreground="green")

entry = tk.Entry(window, font=("Arial", 14))
entry.pack(padx=10, pady=5, fill=tk.X)

send_btn = tk.Button(window, text="Send", font=("Arial", 12), command=send_message)
send_btn.pack(pady=5)

window.mainloop()
