import streamlit as st
import torch
import sentencepiece as spm

from model import TinyGPT
from config import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Load Model
# -------------------------

checkpoint = torch.load("tinygpt_chat.pth", map_location=device)

# Load tokenizer
sp = spm.SentencePieceProcessor()
import os

tokenizer_path = checkpoint["tokenizer_model"]

if not os.path.exists(tokenizer_path):
    st.error(f"Tokenizer not found at {tokenizer_path}")
    st.stop()

sp.load(tokenizer_path)

vocab_size = sp.get_piece_size()

model = TinyGPT(vocab_size, config).to(device)
model.load_state_dict(checkpoint["model"])
model.eval()

# -------------------------
# Streamlit UI
# -------------------------

st.set_page_config(page_title="TinyGPT Chatbot")
st.title("MyLLM Bot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

prompt = st.chat_input("Say something...")

# -------------------------
# Inference
# -------------------------

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Format same as training
    prompt_text = f"User: {prompt} Bot:"

    # Encode using SentencePiece
    input_ids = sp.encode(prompt_text, out_type=int)
    idx = torch.tensor([input_ids], dtype=torch.long).to(device)

    end_token_id = sp.piece_to_id("<END>")

    with torch.no_grad():
        output = model.generate(
            idx,
            max_new_tokens=40,
            end_token_id=end_token_id,
            temperature=0.6
        )

    # Decode output
    decoded_text = sp.decode(output[0].tolist())

    # Remove prompt part
    response = decoded_text.replace(prompt_text, "")

    # Stop at <END>
    if "<END>" in response:
        response = response.split("<END>")[0]

    response = response.strip()

    with st.chat_message("assistant"):
        st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response})