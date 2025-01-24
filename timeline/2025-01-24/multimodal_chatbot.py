import streamlit as st
import openai
import re
import requests
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# --- Streamlit UI 설정 ---
st.set_page_config(page_title="AI 이미지 분석 챗봇", layout="wide")
st.title("🖼️ AI 이미지 분석 챗봇")

# --- 사이드바: OpenAI API Key 입력 ---
openai_api_key = st.sidebar.text_input("🔑 OpenAI API Key", type="password")

# API Key가 없으면 경고 메시지 표시
if not openai_api_key.startswith("sk-"):
    st.sidebar.warning("Please enter your OpenAI API key!", icon="⚠")

# --- 채팅 기록을 세션 상태에 저장 ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "안녕하세요! 질문을 입력하거나 이미지 URL과 함께 요청해주세요."}
    ]

# --- 채팅 기록 출력 (기존 대화 유지) ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 사용자 입력 받기 ---
user_input = st.chat_input("메시지를 입력하세요...")

# --- ✅ 이미지 URL과 텍스트를 자동으로 분류하는 함수 ---
def extract_url_and_text(input_text):
    """사용자 입력에서 이미지 URL과 텍스트를 분리"""
    url_pattern = r"(https?://[^\s]+)"
    match = re.search(url_pattern, input_text)

    if match:
        image_url = match.group(0)  # 첫 번째 URL 추출
        text_prompt = input_text.replace(image_url, "").strip()  # URL을 제외한 텍스트
    else:
        image_url = None
        text_prompt = input_text.strip()

    return image_url, text_prompt

# --- ✅ URL이 실제 이미지인지 확인하는 함수 ---
def is_valid_image_url(url):
    """URL이 실제 이미지인지 확인 (HTTP 응답 헤더 사용)"""
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        content_type = response.headers.get("Content-Type", "")
        return content_type.startswith("image/")
    except requests.RequestException:
        return False

# --- ✅ OpenAI GPT-4o-turbo 호출 ---
if user_input and openai_api_key.startswith("sk-"):
    # URL과 텍스트 분류
    image_url, text_prompt = extract_url_and_text(user_input)

    # 사용자 메시지 저장
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # ✅ 이미지 URL이 있는 경우 확인
    if image_url and not is_valid_image_url(image_url):
        st.warning("🚨 제공된 URL이 유효한 이미지가 아닐 가능성이 있습니다. 다른 URL을 시도하세요.")
        image_url = None  # 유효하지 않으면 URL을 제거

    # OpenAI GPT-4V 요청 메시지 생성
    messages = [{"role": "system", "content": "You are an AI that analyzes images and answers questions based on the image and text input."}]

    # ✅ 이미지 URL이 없는 경우 → 일반 텍스트 질문 처리
    if not image_url:
        messages.append({"role": "user", "content": text_prompt})
    else:
        # ✅ 이미지 URL이 있는 경우 → 텍스트 + 이미지 URL을 함께 전달
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": text_prompt if text_prompt else "이 이미지에서 무엇이 보이나요?"},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        })

    # OpenAI API 호출 (GPT-4-turbo 사용)
    client = openai.OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages,
        max_tokens=500
    )

    # LLM 응답 저장
    assistant_message = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": assistant_message})

    # 챗봇 응답 출력
    with st.chat_message("assistant"):
        st.markdown(assistant_message)
