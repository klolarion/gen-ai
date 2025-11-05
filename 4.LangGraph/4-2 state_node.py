import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage

# load env
load_dotenv()

# API KEY
api_key = os.getenv("OPENAI_API_KEY")

# model
llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0)


# ============================================================================
# State와 노드에 대한 심화 학습
# ============================================================================
# 이 파일에서는 State의 구조와 여러 노드가 어떻게 상호작용하는지 학습합니다.
# ============================================================================


# ============================================================================
# 복잡한 State 정의
# ============================================================================
# State에는 messages 외에도 다양한 정보를 저장할 수 있습니다.
# 예: 대화 횟수, 사용자 정보, 중간 결과 등
# ============================================================================

class ComplexState(TypedDict):
    """
    복잡한 상태 구조 예제
    
    messages: 대화 메시지 리스트 (자동 병합)
    turn_count: 대화 턴 수 (덮어쓰기)
    user_name: 사용자 이름 (덮어쓰기)
    context: 추가 컨텍스트 정보 (덮어쓰기)
    """
    messages: Annotated[list, add_messages]
    turn_count: int  # 단순 타입은 기본적으로 덮어쓰기 (마지막 값으로 대체)
    user_name: str
    context: dict


# ============================================================================
# 여러 노드 함수 정의
# ============================================================================
# 각 노드는 특정 작업을 담당하며, State를 읽고 수정할 수 있습니다.
# ============================================================================

def initialize_state(state: ComplexState) -> ComplexState:
    """
    상태를 초기화하는 노드
    - turn_count를 0으로 설정
    - context를 빈 딕셔너리로 초기화
    
    Args:
        state: 현재 상태
    
    Returns:
        초기화된 상태
    """
    return {
        "turn_count": 0,
        "context": {},
        "messages": []  # 기존 메시지 유지 (빈 리스트 추가 시 기존 메시지와 병합)
    }


def greet_user(state: ComplexState) -> ComplexState:
    """
    사용자에게 인사하는 노드
    - 사용자 이름을 기반으로 인사 메시지 생성
    
    Args:
        state: 현재 상태
    
    Returns:
        인사 메시지가 추가된 상태
    """
    user_name = state.get("user_name", "손님")
    greeting = f"안녕하세요, {user_name}님! 무엇을 도와드릴까요?"
    
    return {
        "messages": [AIMessage(content=greeting)]
    }


def process_query(state: ComplexState) -> ComplexState:
    """
    사용자 쿼리를 처리하는 노드
    - 마지막 사용자 메시지를 가져와 LLM으로 처리
    - turn_count 증가
    
    Args:
        state: 현재 상태
    
    Returns:
        AI 응답이 추가되고 turn_count가 증가한 상태
    """
    messages = state["messages"]
    turn_count = state.get("turn_count", 0)
    
    # LLM 호출
    response = llm.invoke(messages)
    
    # turn_count 증가
    new_turn_count = turn_count + 1
    
    return {
        "messages": [response],
        "turn_count": new_turn_count
    }


def add_context(state: ComplexState) -> ComplexState:
    """
    컨텍스트 정보를 추가하는 노드
    - 대화 횟수와 마지막 메시지 정보를 context에 저장
    
    Args:
        state: 현재 상태
    
    Returns:
        context가 업데이트된 상태
    """
    messages = state["messages"]
    turn_count = state.get("turn_count", 0)
    
    # 마지막 메시지 가져오기
    last_message = messages[-1] if messages else None
    
    # context 업데이트
    context = {
        "last_message_type": type(last_message).__name__ if last_message else None,
        "total_turns": turn_count,
        "has_messages": len(messages) > 0
    }
    
    return {
        "context": context
    }


def print_summary(state: ComplexState) -> ComplexState:
    """
    상태 요약을 출력하는 노드
    - 현재 상태의 주요 정보를 출력
    
    Args:
        state: 현재 상태
    
    Returns:
        변경 없이 상태 반환 (그대로 전달)
    """
    print("\n" + "=" * 80)
    print("상태 요약:")
    print("=" * 80)
    print(f"사용자 이름: {state.get('user_name', 'N/A')}")
    print(f"대화 턴 수: {state.get('turn_count', 0)}")
    print(f"메시지 개수: {len(state.get('messages', []))}")
    print(f"컨텍스트: {state.get('context', {})}")
    print("=" * 80 + "\n")
    
    # 상태를 변경하지 않고 그대로 반환
    return state


# ============================================================================
# 그래프 구성: 여러 노드를 순차적으로 연결
# ============================================================================

workflow = StateGraph(ComplexState)

# 노드 추가
workflow.add_node("initialize", initialize_state)
workflow.add_node("greet", greet_user)
workflow.add_node("process", process_query)
workflow.add_node("add_context", add_context)
workflow.add_node("summary", print_summary)

# 엣지 설정: 순차적 실행
workflow.set_entry_point("initialize")  # 시작: 초기화
workflow.add_edge("initialize", "greet")  # 초기화 → 인사
workflow.add_edge("greet", "process")     # 인사 → 쿼리 처리
workflow.add_edge("process", "add_context")  # 쿼리 처리 → 컨텍스트 추가
workflow.add_edge("add_context", "summary")  # 컨텍스트 추가 → 요약 출력
workflow.add_edge("summary", END)  # 요약 출력 → 종료

# 그래프 컴파일
app = workflow.compile()


# ============================================================================
# 그래프 실행 예제
# ============================================================================

# 초기 상태 설정
initial_state = {
    "messages": [HumanMessage(content="지구의 자전 주기는 얼마인가요?")],
    "user_name": "홍길동",
    "turn_count": 0,
    "context": {}
}

print("\n그래프 실행 시작...\n")
result = app.invoke(initial_state)

# 최종 결과 출력
print("\n최종 대화 내용:")
print("-" * 80)
for message in result["messages"]:
    if isinstance(message, HumanMessage):
        print(f"[사용자]: {message.content}")
    elif isinstance(message, AIMessage):
        print(f"[AI]: {message.content}")
print("-" * 80)


# ============================================================================
# 참고: State 병합 규칙
# ============================================================================
# 1. Annotated[list, add_messages]: 리스트 병합 (메시지 추가)
# 2. 일반 타입 (int, str, dict 등): 덮어쓰기 (마지막 값으로 대체)
# 3. Annotated[dict, operator.add]: 딕셔너리 병합 (키별로 합침)
#
# 예시:
#   state1 = {"messages": [msg1], "count": 1}
#   state2 = {"messages": [msg2], "count": 2}
#   병합 결과 = {"messages": [msg1, msg2], "count": 2}
# ============================================================================

