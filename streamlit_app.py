import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Title
st.title("🤖 Lightweight Chatbot")
st.write("Chatbot powered by Hugging Face (DialoGPT-small) + Streamlit")

# Load model + tokenizer (cached to avoid reloading)
@st.cache_resource
def load_model():
    model_name = "microsoft/DialoGPT-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")
    return tokenizer, model

tokenizer, model = load_model()

# Session state for chat history
if "history" not in st.session_state:
    st.session_state["history"] = []

# User input
user_input = st.text_input("You: ", "")

if st.button("Send") and user_input:
    # Encode user input with history
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    bot_input_ids = torch.cat([torch.tensor(st.session_state["history"], dtype=torch.long), new_input_ids], dim=-1) if st.session_state["history"] else new_input_ids
    
    # Generate response
    output = model.generate(bot_input_ids, max_length=200, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    # Update history
    st.session_state["history"] = output.tolist()

    # Display conversation
    st.markdown(f"**You:** {user_input}")
    st.markdown(f"**Bot:** {response}")
