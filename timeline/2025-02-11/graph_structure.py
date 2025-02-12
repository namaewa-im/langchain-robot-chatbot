import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

from langgraph.graph import StateGraph
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from pydantic import BaseModel

# 1️. 상태 정의
class TaskState(BaseModel):
    user_input: str = ""
    parsed_task: str = ""
    task_result: str = ""
    error: str = ""

# 2️. LLM 설정
llm = ChatOpenAI(model="gpt-4", openai_api_key=openai_api_key)

# 3️. 그래프 생성
graph = StateGraph(TaskState)

# 4️. 디버깅용 출력 함수
def print_state(node_name, state: TaskState):
    print(f"🟢 현재 실행 중인 노드: {node_name}")
    print(f"🔹 State: {state.model_dump()}\n")

# 5️. 사용자 입력 노드
def get_user_input(state: TaskState):
    print_state("get_user_input", state)
    user_input = input("사용자 입력: ")  # 실제 응용에서는 UI에서 받을 수 있음
    return {"user_input": user_input}

# 6️. LLM을 통해 입력을 파싱하는 노드
def parse_task(state: TaskState):
    print_state("parse_task", state)
    prompt = f"다음 입력에서 수행할 태스크를 하나의 단어로 지정하세요 (번역, 요약, 분석 중 하나): {state.user_input}"
    response = llm([HumanMessage(content=prompt)]).content.strip().lower()
    
    if response not in ["번역", "요약", "분석"]:
        return {"error": "올바른 태스크를 찾을 수 없습니다. 다시 입력하세요.", "parsed_task": ""}
    
    return {"parsed_task": response, "error": ""}

# 7️. Task 수행 노드 정의
def translate_task(state: TaskState):
    print_state("translate_task", state)
    return {"task_result": f"번역 완료: {state.user_input}"}

def summarize_task(state: TaskState):
    print_state("summarize_task", state)
    return {"task_result": f"요약 완료: {state.user_input[:10]}..."}

def analyze_task(state: TaskState):
    print_state("analyze_task", state)
    return {"task_result": f"분석 완료: 입력 길이 {len(state.user_input)}"}

# 8️. 오류 처리 노드
def handle_error(state: TaskState):
    print_state("handle_error", state)
    return {"task_result": state.error}

# 9. 조건부 Edge 설정
def task_selector(state: TaskState):
    if state.error:
        return "error_handler"
    return {
        "번역": "translate",
        "요약": "summarize",
        "분석": "analyze"
    }.get(state.parsed_task, "error_handler")

# 10. 루프 구조: 작업 수행 후 다시 사용자 입력 받기
graph.add_node("get_user_input", get_user_input)
graph.add_node("parse_task", parse_task)
graph.add_node("translate", translate_task)
graph.add_node("summarize", summarize_task)
graph.add_node("analyze", analyze_task)
graph.add_node("error_handler", handle_error)

graph.add_edge("get_user_input", "parse_task")
graph.add_conditional_edges("parse_task", task_selector)
graph.add_edge("translate", "get_user_input")
graph.add_edge("summarize", "get_user_input")
graph.add_edge("analyze", "get_user_input")
graph.add_edge("error_handler", "get_user_input")

# 1️1. 시작 및 종료 지점 설정
graph.set_entry_point("get_user_input")

# 1️2️. 그래프 실행 (무한 루프 가능)
app = graph.compile()
state = TaskState()
while True:
    state = app.invoke(state)
    print(f"실행 결과: {state.task_result}\n")
