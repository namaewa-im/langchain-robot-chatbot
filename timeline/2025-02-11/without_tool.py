import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

from langgraph.graph import StateGraph
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from pydantic import BaseModel

# 1️⃣ 상태 정의
class TaskState(BaseModel):
    user_input: str = ""
    parsed_task: str = ""
    task_result: str = ""
    error: str = ""
    end: bool = False  # 종료 여부 확인

# 2️⃣ LLM 설정
llm = ChatOpenAI(model="gpt-4", openai_api_key=openai_api_key)

# 3️⃣ 그래프 생성
graph = StateGraph(TaskState)

# 4️⃣ 디버깅용 출력 함수
def print_state(node_name, state: TaskState):
    print(f"\n🟢 현재 실행 중인 노드: {node_name}")
    print(f"🔹 State: {state.model_dump()}\n")

# 5️⃣ 사용자 입력 노드
def get_user_input(state: TaskState):
    print_state("get_user_input", state)
    user_input = input("사용자 입력: ").strip().lower()
    
    if user_input in ["그만", "종료", "quit"]:
        return {"end": True}
    
    return {"user_input": user_input}

# 6️⃣ LLM을 통해 입력을 파싱하는 노드
def parse_task(state: TaskState):
    print_state("parse_task", state)
    if state.end:  # 사용자가 종료 의도를 보였을 경우 종료
        return {"parsed_task": "종료"}

    prompt = f"다음 입력에서 수행할 태스크를 하나의 단어로 지정하세요 (번역, 요약, 분석 중 하나): {state.user_input}"
    response = llm([HumanMessage(content=prompt)]).content.strip().lower()
    
    if response not in ["번역", "요약", "분석"]:
        return {"error": "올바른 태스크를 찾을 수 없습니다. 다시 입력하세요.", "parsed_task": ""}
    
    return {"parsed_task": response, "error": ""}


import openai
from googletrans import Translator
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# nltk.download("vader_lexicon")  # 감정 분석을 위한 데이터셋 다운로드
sia = SentimentIntensityAnalyzer()

# 7️⃣ Task별 Tool 정의
# 번역 툴 (Google Translate API 사용)
from deep_translator import GoogleTranslator

def translate_tool(text: str, target_lang: str = "en") -> str:
    try:
        translated_text = GoogleTranslator(source='auto', target=target_lang).translate(text)
        return f"🔠 번역 결과: {translated_text}"
    except Exception as e:
        return f"❌ 번역 오류: {str(e)}"

# 요약 툴 (OpenAI GPT API 사용)
def summarize_tool(text: str) -> str:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are a helpful assistant that summarizes text."},
                      {"role": "user", "content": f"다음 내용을 짧게 요약해줘: {text}"}]
        )
        summary = response["choices"][0]["message"]["content"]
        return f"📄 요약 결과: {summary}"
    except Exception as e:
        return f"❌ 요약 오류: {str(e)}"

# 감정 분석 툴 (VADER Sentiment Analyzer 사용)
def analyze_tool(text: str) -> str:
    """한국어 텍스트를 영어로 번역한 후 감정 분석을 수행하는 함수"""
    try:
        # 1️⃣ 한국어 → 영어 번역
        translated_text = GoogleTranslator(source='auto', target='en').translate(text)

        # 2️⃣ NLTK VADER 감정 분석 수행
        sentiment_scores = sia.polarity_scores(translated_text)
        compound_score = sentiment_scores["compound"]

        # 3️⃣ 감정 분석 결과 결정
        if compound_score > 0.05:
            sentiment_label = "긍정적 😀"
        elif compound_score < -0.05:
            sentiment_label = "부정적 😞"
        else:
            sentiment_label = "중립적 😐"

        return f"📊 감정 분석 결과: {sentiment_label} (점수: {compound_score})"
    
    except Exception as e:
        return f"❌ 감정 분석 오류: {str(e)}"

# 8️⃣ Task 수행 노드 정의
def translate_task(state: TaskState):
    print_state("translate_task", state)
    return {"task_result": translate_tool(state.user_input)}

def summarize_task(state: TaskState):
    print_state("summarize_task", state)
    return {"task_result": summarize_tool(state.user_input)}

def analyze_task(state: TaskState):
    print_state("analyze_task", state)
    return {"task_result": analyze_tool(state.user_input)}

# 9️⃣ 오류 처리 노드
def handle_error(state: TaskState):
    print_state("handle_error", state)
    return {"task_result": state.error}

# 🔟 종료 노드
def end_node(state: TaskState):
    print_state("end_node", state)
    return {"task_result": "🛑 대화가 종료되었습니다."}

# 1️⃣1️⃣ 조건부 Edge 설정
def task_selector(state: TaskState):
    if state.end:
        return "end_node"
    if state.error:
        return "error_handler"
    return {
        "번역": "translate",
        "요약": "summarize",
        "분석": "analyze"
    }.get(state.parsed_task, "error_handler")

# 1️⃣2️⃣ 그래프 구조 설정
graph.add_node("get_user_input", get_user_input)
graph.add_node("parse_task", parse_task)
graph.add_node("translate", translate_task)
graph.add_node("summarize", summarize_task)
graph.add_node("analyze", analyze_task)
graph.add_node("error_handler", handle_error)
graph.add_node("end_node", end_node)  # 종료 노드 추가

graph.add_edge("get_user_input", "parse_task")
graph.add_conditional_edges("parse_task", task_selector)
graph.add_edge("translate", "get_user_input")
graph.add_edge("summarize", "get_user_input")
graph.add_edge("analyze", "get_user_input")
graph.add_edge("error_handler", "get_user_input")

# 1️⃣3️⃣ 시작 및 종료 지점 설정
graph.set_entry_point("get_user_input")
graph.set_finish_point("end_node")  # 종료 노드 설정

# 1️⃣4️⃣ 그래프 실행 (무한 루프 가능)
app = graph.compile()
state = TaskState()
while True:
    state = app.invoke(state)  # `state`는 이제 `dict` 형태임
    print(f"✅ 실행 결과: {state.get('task_result', '')}\n")  # `.get()`을 사용하여 오류 방지
    if state.get("end", False):  # 종료 조건 확인
        break