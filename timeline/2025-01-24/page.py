import streamlit as st
from langchain_openai.chat_models import ChatOpenAI

st.title("Chatbot")

# OPENAI_API_KEY 입력 사이드 바 추가
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# LLM 출력 받아오는 함수
def generate_response(input_text):
    model = ChatOpenAI(temperature=0.7, api_key=openai_api_key)
    st.info(model.invoke(input_text))

# 입력 필드 기본값을 플레이스홀더로 설정
placeholder_text = "메시지 입력..."

with st.form("my_form"):
    text = st.text_area("Enter text:", value="", placeholder=placeholder_text)
    submitted = st.form_submit_button("Submit")

    if not openai_api_key.startswith("sk-"):
        st.warning("Please enter your OpenAI API key!", icon="⚠")

    if submitted and openai_api_key.startswith("sk-"):
        generate_response(text)
