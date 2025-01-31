import streamlit as st
import chess
import chess.svg
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
import openai

# 🔥 Streamlit 페이지 설정
st.set_page_config(layout="wide")

# 🔥 API 키 입력 필드 추가
st.sidebar.markdown("### 🔑 OpenAI API Key 입력")
api_key = st.sidebar.text_input("OpenAI API Key를 입력하세요:", type="password")

# 🔥 API 키를 `st.session_state`에 저장
if api_key:
    st.session_state["api_key"] = api_key

# 🔥 체스 보드 관리
if "board" not in st.session_state:
    st.session_state.board = chess.Board()

# 🔥 LangGraph를 사용하여 체스 챗봇 구현
class ChatState(dict):
    user_input: str
    ai_response: str
    fen: str

# 🔥 LangGraph 초기화
workflow = StateGraph(state_schema=ChatState)

def chess_ai_node(state: ChatState):
    """체스 FEN 상태를 기반으로 AI 응답 생성"""

    # 🔥 API 키가 없으면 실행 불가
    if "api_key" not in st.session_state or not st.session_state["api_key"]:
        state["ai_response"] = "🚨 API 키가 필요합니다. 좌측 사이드바에서 입력해주세요!"
        return state

    # ✅ OpenAI 모델 인스턴스 생성 (실행 시 동적으로 API 키 사용)
    gpt4o_mini = ChatOpenAI(
        model_name="gpt-4o",
        max_tokens=200,
        temperature=0.7,
        api_key=st.session_state["api_key"]
    )

    board = chess.Board(state["fen"])
    valid_moves = [board.san(move) for move in board.legal_moves]

    # ✅ 자유로운 대화를 할 수 있도록 프롬프트 개선
    prompt = f"""
    현재 체스 보드 상태는 다음과 같습니다 (FEN):
    {state["fen"]}

    가능한 다음 수:
    {', '.join(valid_moves)}

    사용자의 질문: {state["user_input"]}

    당신은 체스 전문가이자 코치입니다. 
    - 체스 전략, 전술, 다음 수 추천 등을 도와주세요.
    - 사용자가 요청하면 추천 수를 제공하세요.
    - 전략적 조언도 가능합니다.
    - 포지션을 분석하고 사용자가 원하는 정보를 제공하세요.
    """

    # 🔥 GPT 모델 호출
    response = gpt4o_mini.invoke(prompt)

    return {"ai_response": response}

workflow.add_node("chess_ai", chess_ai_node)
workflow.set_entry_point("chess_ai")
app = workflow.compile()

# 🔥 UI 레이아웃 설정
col1, col2 = st.columns([1.5, 1])  # 왼쪽 체스판, 오른쪽 챗봇

# 🔹 왼쪽: 체스 보드 UI
with col1:
    st.markdown("### 체스 보드")
    board_placeholder = st.empty()  # 체스 보드가 표시될 자리

    # 🔥 체스 보드 업데이트 함수
    def update_board():
        svg = chess.svg.board(board=st.session_state.board, size=350)
        board_placeholder.markdown(f'<div style="width: 350px; height: 350px;">{svg}</div>', unsafe_allow_html=True)

    # 초기 보드 표시
    update_board()

    # 사용자 입력 (SAN 또는 UCI 형식 지원)
    move_input = st.text_input("체스 수 입력 (예: e4, Nf3, e2e4):")

    if st.button("이동"):
        try:
            board = st.session_state.board

            # 🔥 입력이 SAN인지 확인 후 변환 (SAN → UCI 변환 시도)
            try:
                move = board.parse_san(move_input)  # 사용자가 SAN을 입력한 경우 변환
            except ValueError:
                move = chess.Move.from_uci(move_input)  # 사용자가 UCI 형식을 입력한 경우

            # 🔥 변환된 move를 체스 보드에 적용
            if move in board.legal_moves:
                board.push(move)
                update_board()
            else:
                st.error("🚨 잘못된 수입니다. 다시 입력하세요.")
        
        except Exception as e:
            st.error(f"🚨 오류 발생: {e}")

# 🔹 오른쪽: 체스 챗봇 UI
with col2:
    st.markdown("### 체스 챗봇")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 사용자 질문 입력
    user_input = st.text_input("질문을 입력하세요 (예: 현재 보드에서 좋은 수 추천, 체스 전략 질문 등):")

    if st.button("질문 전송"):
        # LangGraph를 사용하여 AI 응답 생성
        state = {
            "user_input": user_input,
            "fen": st.session_state.board.fen()  # 현재 체스 보드의 FEN 상태
        }
        result = app.invoke(state)

        ai_response = result["ai_response"]

        # 채팅 내역 저장 및 표시
        st.session_state.chat_history.append(f"👤 사용자: {user_input}")
        st.session_state.chat_history.append(f"🤖 AI: {ai_response.content}")

    # 채팅 내역 출력
    st.write("💬 채팅 기록:")
    for msg in st.session_state.chat_history:
        st.write(msg)
