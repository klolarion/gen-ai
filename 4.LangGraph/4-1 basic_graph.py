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
# LangGraph 기본 개념
# ============================================================================
#
# LangGraph란?
# - LangChain의 확장 라이브러리로, 복잡한 AI 워크플로우를 그래프 구조로 표현
# - 상태(State) 기반으로 여러 노드(Node)를 연결하여 복잡한 로직 구현 가능
# - 조건부 라우팅, 루프, 병렬 처리 등 고급 기능 지원
#
# 주요 구성 요소:
# 1. State: 그래프 전체에서 공유되는 상태 정보 (딕셔너리 형태)
# 2. Node: 실제 작업을 수행하는 함수 (각 노드는 입력 State를 받아 출력 State를 반환)
# 3. Edge: 노드 간 연결 관계 (단순 연결 또는 조건부 라우팅)
# 4. Graph: StateGraph 객체로 전체 워크플로우 정의
#
# 기본 구조:
# START → Node1 → Node2 → END
#   또는
# START → Node1 → [조건에 따라] → Node2 or Node3 → END
# ============================================================================


# ============================================================================
# Step 1: State 정의
# ============================================================================
# State는 그래프 전체에서 공유되는 데이터 구조입니다.
# TypedDict를 사용하여 State의 구조를 명시적으로 정의합니다.
#
# Annotated 사용 이유:
# - LangGraph는 State의 각 필드를 어떻게 병합(merge)할지 정의해야 함
# - add_messages는 메시지 리스트를 자동으로 병합 (대화 기록 유지)
# - messages 필드에 새로운 메시지를 추가하면 기존 메시지와 자동 병합됨
# ============================================================================

class GraphState(TypedDict):
    """
    그래프의 상태를 정의하는 클래스
    
    messages: Annotated[list, add_messages]
        - 대화 메시지들을 저장하는 리스트
        - HumanMessage, AIMessage 등이 저장됨
        - add_messages 함수로 인해 새 메시지가 추가될 때 기존 메시지와 자동 병합
    """
    messages: Annotated[list, add_messages]


# ============================================================================
# Step 2: 노드(Node) 함수 정의
# ============================================================================
# 노드는 그래프에서 실제 작업을 수행하는 단위입니다.
# 각 노드 함수는:
# 1. State를 입력으로 받음
# 2. State를 수정하거나 새로운 State를 반환
# 3. 반환된 State는 다음 노드로 전달됨
# ============================================================================

def call_model(state: GraphState) -> GraphState:
    """
    LLM을 호출하여 응답을 생성하는 노드 함수
    
    Args:
        state: 현재 그래프의 상태 (messages 포함)
    
    Returns:
        수정된 상태 (AI 응답 메시지가 추가됨)
    """
    # State에서 메시지 목록 가져오기
    messages = state["messages"]
    
    # LLM에 메시지 전달하여 응답 생성
    # invoke()는 메시지 리스트를 받아 AIMessage 객체를 반환
    response = llm.invoke(messages)
    
    # 응답을 State의 messages에 추가
    # add_messages로 인해 기존 messages와 자동 병합됨
    return {"messages": [response]}


# ============================================================================
# Step 3: 그래프 구성 및 실행
# ============================================================================
# StateGraph를 사용하여 노드들을 연결하고 워크플로우를 정의합니다.
# ============================================================================

# 그래프 생성
# StateGraph는 State 타입을 인자로 받아 그래프를 초기화
workflow = StateGraph(GraphState)

# 노드 추가
# add_node("노드이름", 노드함수)
# - 첫 번째 인자: 노드의 이름 (나중에 엣지에서 참조할 때 사용)
# - 두 번째 인자: 노드 함수 (State를 받아 State를 반환하는 함수)
workflow.add_node("agent", call_model)

# 엣지(Edge) 설정
# 엣지는 노드 간의 연결을 정의합니다.
# - set_entry_point("시작노드"): 그래프의 시작점 설정
# - add_edge("이전노드", "다음노드"): 단순 연결 (무조건 다음 노드로 이동)
# - add_edge("이전노드", END): 그래프 종료

workflow.set_entry_point("agent")  # "agent" 노드가 그래프의 시작점
workflow.add_edge("agent", END)    # "agent" 노드 실행 후 그래프 종료

# 그래프 컴파일
# compile()을 호출해야 그래프가 실행 가능한 객체가 됨
app = workflow.compile()


# ============================================================================
# Step 4: 그래프 실행
# ============================================================================
# 컴파일된 그래프에 초기 상태를 전달하여 실행합니다.
# ============================================================================

# 초기 상태 설정
# HumanMessage를 messages 리스트에 포함하여 초기 상태 생성
initial_state = {
    "messages": [HumanMessage(content="안녕하세요! 지구의 자전 주기에 대해 설명해주세요.")]
}

# 그래프 실행
# invoke(state)는 그래프를 실행하고 최종 상태를 반환
result = app.invoke(initial_state)

# 결과 출력
print("=" * 80)
print("그래프 실행 결과:")
print("=" * 80)
for message in result["messages"]:
    if isinstance(message, HumanMessage):
        print(f"\n[사용자]: {message.content}")
    elif isinstance(message, AIMessage):
        print(f"\n[AI]: {message.content}")
print("=" * 80)


# ============================================================================
# 참고: 그래프 구조 시각화
# ============================================================================
# 그래프 구조를 시각화하려면 다음 코드를 사용할 수 있습니다:
# print(app.get_graph().draw_mermaid())
# 또는
# print(app.get_graph().print_ascii())
# ============================================================================

