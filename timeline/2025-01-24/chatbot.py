import streamlit as st
from langchain_openai import ChatOpenAI

# --- Streamlit UI 설정 ---
st.set_page_config(page_title="Chatbot", layout="wide")  # 화면을 넓게 설정
st.title("Chatbot")

# --- 사이드바: OpenAI API Key 입력 ---
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# API Key가 없으면 경고 메시지 표시
if not openai_api_key.startswith("sk-"):
    st.sidebar.warning("Please enter your OpenAI API key!", icon="⚠")

# --- 채팅 기록을 세션 상태에 저장 ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "안녕하세요! 무엇을 도와드릴까요?"}
    ]

# --- 채팅 기록 출력 (기존 대화 유지) ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 사용자 입력 받기 ---
prompt = st.chat_input("메시지 입력...")

# --- 입력이 있을 경우 OpenAI 호출 ---
if prompt and openai_api_key.startswith("sk-"):
    # 사용자 메시지 저장
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # OpenAI API 호출
    model = ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini")
    response = model.invoke(st.session_state.messages)

    # LLM 응답 저장
    assistant_message = response.content
    st.session_state.messages.append({"role": "assistant", "content": assistant_message})

    # 챗봇 응답 출력
    with st.chat_message("assistant"):
        st.markdown(assistant_message)
