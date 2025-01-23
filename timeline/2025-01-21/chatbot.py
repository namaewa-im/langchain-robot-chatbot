import streamlit as st

title = st.title("Chatbot")
prompt = st.chat_input("메시지 ChatGPT")
if prompt:
    st.write(f"User has sent the following prompt: {prompt}")