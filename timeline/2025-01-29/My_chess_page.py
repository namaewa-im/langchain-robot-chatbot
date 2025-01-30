import streamlit as st
import chess
import chess.svg
import cairosvg
import random
from io import BytesIO
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
from typing import TypedDict
from langchain.schema import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

# ✅ 환경 변수 로드
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# ✅ OpenAI LLM 설정
llm = ChatOpenAI(
    model="gpt-4o-mini",
    max_tokens=100,
    temperature=0.5,
    openai_api_key=openai_api_key
)

# ✅ 상태 초기화
if "board" not in st.session_state:
    st.session_state.board = chess.Board()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_turn" not in st.session_state:
    st.session_state.current_turn = "user"  
if "current_node" not in st.session_state:
    st.session_state.current_node = "user_move"  # 사용자(White)가 먼저 시작

# ✅ Streamlit 설정
st.set_page_config(page_title="♟️ LLM 체스 챗봇", layout="centered")
st.title("♟️ LLM 체스 챗봇")

# ✅ 체스 보드 렌더링 함수
def render_chessboard(board):
    svg_data = chess.svg.board(board=board, size=400)
    png_data = BytesIO()
    cairosvg.svg2png(bytestring=svg_data.encode('utf-8'), write_to=png_data)
    return png_data

# ✅ UI 업데이트
board_placeholder = st.empty()
board_placeholder.image(render_chessboard(st.session_state.board), caption="현재 체스 보드 상태")

# ✅ "Restart" 버튼 추가 (게임 초기화)
if st.button("🔄 Restart"):
    st.session_state.board = chess.Board()
    st.session_state.chat_history = []
    st.session_state.current_turn = "user"  
    board_placeholder.image(render_chessboard(st.session_state.board), caption="새 게임 시작")
    st.rerun()

# ✅ State 클래스 정의
class State(TypedDict): 
    fen: str
    current_turn: str
    is_check: bool
    is_checkmate: bool
    is_stalemate: bool
    current_node: str

# ✅ 체스 상태(State) 반환 함수
def get_state(board: chess.Board) -> State:
    return {
        "fen": board.fen(),
        "current_turn": "user" if board.turn == chess.WHITE else "AI",
        "is_check": board.is_check(),
        "is_checkmate": board.is_checkmate(),
        "is_stalemate": board.is_stalemate(),
        "current_node": st.session_state.current_node  # Ensure current_node is included
    }

# ✅ AI(Black) 턴 함수
def ai_move(state: dict) -> dict:
    """AI(Black)의 움직임을 수행"""
    st.write("🔹 AI Move - 현재 state:", state)  

    game_board = chess.Board(state["fen"])  # ✅ `board` 대신 `game_board` 사용하여 충돌 방지
    legal_moves = list(game_board.legal_moves)

    if legal_moves:
        best_move = legal_moves[0].uci()  
        game_board.push_uci(best_move)
    else:
        best_move = None

    st.session_state.board = game_board  # ✅ Streamlit 상태 업데이트
    st.session_state.chat_history.append(("🤖 AI (Black)", best_move))

    board_placeholder.image(render_chessboard(game_board), caption="현재 체스 보드 상태")

    new_state = {
        "fen": game_board.fen(),  # ✅ `board` 대신 `fen`만 저장
        "current_turn": "user",
        "is_check": game_board.is_check(),
        "is_checkmate": game_board.is_checkmate(),
        "is_stalemate": game_board.is_stalemate(),
        "current_node": "check_status"
    }
    
    st.write("✅ AI가 움직인 후 state:", new_state)  
    return new_state


# ✅ 사용자(White) 턴 함수 (사용자 입력을 버튼으로 받을 수 있도록 개선)
def user_move(state: dict) -> dict:
    """사용자(White)의 움직임을 수행"""
    st.write("🔹 User Move - 현재 state:", state)  

    user_msg = st.text_input("당신의 움직임을 입력하세요 (예: e2e4)")

    if user_msg:
        game_board = chess.Board(state["fen"])  # ✅ `board` 대신 `game_board` 사용
        
        if user_msg in [m.uci() for m in game_board.legal_moves]:  
            game_board.push_uci(user_msg)
            st.session_state.board = game_board  
            st.session_state.chat_history.append(("🧑‍💻 사용자 (White)", user_msg))

            board_placeholder.image(render_chessboard(game_board), caption="현재 체스 보드 상태")

            new_state = {
                "fen": game_board.fen(),
                "current_turn": "AI",
                "is_check": game_board.is_check(),
                "is_checkmate": game_board.is_checkmate(),
                "is_stalemate": game_board.is_stalemate(),
                "current_node": "check_status"
            }

            st.write("✅ 사용자가 움직인 후 state:", new_state)  
            return new_state

        else:
            st.warning("잘못된 움직임입니다. 다시 입력하세요.")

    return {**state, "current_node": "user_move"}  

# ✅ 게임 상태 체크 함수 (더 직관적으로 개선)
def check_status(state: dict) -> dict:
    st.write("🔹 check state - 현재 state:", state)
    """게임 상태를 확인하고, 항상 `current_node`를 설정"""
    if state["is_checkmate"]:
        st.success("체크메이트! 게임 종료")
        return {**state, "current_node": "game_over"}

    if state["is_stalemate"]:
        st.warning("스테일메이트! 게임 종료")
        return {**state, "current_node": "game_over"}

    if state["current_turn"] == "AI":
        return {**state, "current_node": "ai_move"}
    st.write("🔹 check state - 이후 state:", state)
    return {**state, "current_node": "user_move"}  # ✅ 항상 `current_node` 포함

# ✅ 게임 종료 함수
def game_over(state: dict) -> dict:
    st.info("게임이 종료되었습니다.")
    return {**state, "current_node": "game_over"}

# ✅ LangGraph 그래프 정의
graph = StateGraph(State)
graph.add_node("ai_move", ai_move)
graph.add_node("user_move", user_move)
graph.add_node("check_status", check_status)
graph.add_node("game_over", game_over)

graph.add_edge(START, "user_move")
graph.add_edge("ai_move", "check_status")
graph.add_edge("user_move", "check_status")

# ✅ 조건부 엣지 추가 
# 📌 add_conditional_edges(현재 노드, { "다음 노드": 조건 함수 })
graph.add_conditional_edges(
    "check_status",  # ✅ 현재 노드
    {
        "game_over": lambda state: state["current_node"] == "game_over",
        "ai_move": lambda state: state["current_node"] == "ai_move",
        "user_move": lambda state: state["current_node"] == "user_move",
    }
)
graph.add_edge("game_over", END)

# ✅ LangGraph 실행
app = graph.compile()

# ✅ 초기 상태 가져오기
def initialize_state():
    return {
        "fen": st.session_state.board.fen(),
        "current_turn": "user",
        "is_checkmate": st.session_state.board.is_checkmate(),
        "is_stalemate": st.session_state.board.is_stalemate(),
        "current_node": "user_move"
    }

# ✅ LangGraph 실행을 `invoke()` 대신 `stream()`으로 변경
state = initialize_state()  # ✅ 초기 상태 설정

# ✅ `stream()`을 사용하여 LangGraph가 상태 변화에 따라 자동 실행
for updated_state in app.stream(state):
    st.write("🔄 LangGraph 진행 중 - updated_state:", updated_state)  # ✅ 현재 상태 출력

    if "current_node" not in updated_state:
        raise KeyError("❌ current_node가 state에서 누락되었습니다!")  # ✅ 디버깅용 에러 체크
    
    if updated_state["current_node"] == "game_over":
        st.write("🎉 게임 종료! 최종 state:", updated_state)  # ✅ 최종 상태 출력
        break  # 게임 종료 시 루프 중단
