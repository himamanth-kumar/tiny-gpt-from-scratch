import streamlit as st
import torch
from model import TinyGPT
from config import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load("tinygpt_chat.pth", map_location=device)
word2idx = checkpoint["word2idx"]
idx2word = checkpoint["idx2word"]

model = TinyGPT(len(word2idx), config).to(device)
model.load_state_dict(checkpoint["model"])
model.eval()

st.set_page_config(page_title="TinyGPT Chatbot")
st.title("MyLLM Bot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

prompt = st.chat_input("Say something...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    prompt_text = f"User: {prompt} Bot:"
    words = [w for w in prompt_text.split() if w in word2idx]

    if len(words) == 0:
        response = "I don't understand."
    else:
        idx = torch.tensor([[word2idx[w] for w in words]], dtype=torch.long).to(device)
        end_id = word2idx["<END>"]

        with torch.no_grad():
            output = model.generate(
                idx,
                max_new_tokens=30,
                end_token_id=end_id,
                temperature=0.8
            )

        decoded = [idx2word[int(i)] for i in output[0]]
        generated = decoded[len(words):]

        if "<END>" in generated:
            generated = generated[:generated.index("<END>")]

        response = " ".join(generated)

    with st.chat_message("assistant"):
        st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
