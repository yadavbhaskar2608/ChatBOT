import customtkinter as ctk
import torch
from tokenizers import ByteLevelBPETokenizer
from model import TinyGPT
import torch.nn.functional as F
import re
import os

# ------------------ DEVICE ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------ TOKENIZER PATH FIX ------------------
base_path = os.path.dirname(os.path.abspath(__file__))
tokenizer_path = os.path.join(base_path, "tokenizer", "vocab.json")
merges_path = os.path.join(base_path, "tokenizer", "merges.txt")

tokenizer = ByteLevelBPETokenizer(tokenizer_path, merges_path)

# ------------------ LOAD MODEL ------------------
model = TinyGPT(vocab_size=5000).to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# ------------------ SAMPLING ------------------
def sample_next_token(logits, temperature=0.8):
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()

# ------------------ GENERATION ------------------
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

    # Clean tags
    text = re.sub(r"<.*?>", "", text)

    # Extract only bot output
    if "Bot:" in text:
        text = text.split("Bot:", 1)[1].strip()
    if "User:" in text:
        text = text.split("User:", 1)[0].strip()

    return text


# ------------------ GUI ------------------

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("My LLM ChatGPT")
app.geometry("800x700")

# ------------ Chat Frame ------------
chat_frame = ctk.CTkScrollableFrame(app, width=780, height=580)
chat_frame.pack(padx=10, pady=10)

widgets = []  # To store chat bubbles


def add_message(sender, text):
    bubble_color = "#1f6aa5" if sender == "You" else "#2e2e2e"
    anchor = "e" if sender == "You" else "w"

    msg = ctk.CTkLabel(
        chat_frame,
        text=f"{sender}: {text}",
        fg_color=bubble_color,
        corner_radius=12,
        justify="left",
        wraplength=600,
        padx=10,
        pady=8
    )
    msg.pack(anchor=anchor, pady=5, padx=10)
    widgets.append(msg)


# ------------ Input Area ------------
input_frame = ctk.CTkFrame(app)
input_frame.pack(fill="x", padx=10, pady=10)

entry = ctk.CTkEntry(input_frame, height=40, width=600, placeholder_text="Type your message...")
entry.pack(side="left", padx=10)

def send_message(event=None):
    user_text = entry.get().strip()
    if not user_text:
        return

    add_message("You", user_text)
    entry.delete(0, ctk.END)

    bot_text = generate_reply(user_text)
    add_message("Bot", bot_text)

send_btn = ctk.CTkButton(input_frame, text="Send", width=100, command=send_message)
send_btn.pack(side="right", padx=10)

entry.bind("<Return>", send_message)

app.mainloop()
