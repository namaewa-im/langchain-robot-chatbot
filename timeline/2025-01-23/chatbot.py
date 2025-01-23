import streamlit as st
import openai
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# OpenAI API 설정
api_key = os.getenv("OPENAI_API_KEY")

# OpenAI 클라이언트 생성
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Streamlit UI 설정
st.title("Chatbot")

# 채팅 기록을 세션 상태에 저장
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "안녕하세요! 무엇을 도와드릴까요?"}
    ]

# 채팅 기록 출력 (기존 대화 유지)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 받기
prompt = st.chat_input("메시지 입력...")

# 입력이 있을 경우 LLM 호출
if prompt:
    # 사용자 메시지 저장
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # OpenAI API 호출
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=st.session_state.messages  # 전체 대화 내역 전달
    )

    # LLM 응답 저장
    assistant_message = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": assistant_message})

    # 봇의 응답 출력
    with st.chat_message("assistant"):
        st.markdown(assistant_message)
