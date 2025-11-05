import os
from typing import TypedDict, Annotated, Literal
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
# 조건부 엣지(Conditional Edges) 학습
# ============================================================================
# 조건부 엣지는 노드의 출력이나 상태에 따라 다음 노드를 동적으로 선택할 수 있게 합니다.
# 이를 통해 분기, 루프, 복잡한 워크플로우를 구현할 수 있습니다.
# ============================================================================


class RouterState(TypedDict):
    """
    라우팅을 위한 상태 구조
    """
    messages: Annotated[list, add_messages]
    next_step: str  # 다음 단계를 결정하는 플래그


# ============================================================================
# 노드 함수들
# ============================================================================

def classify_intent(state: RouterState) -> RouterState:
    """
    사용자 의도를 분류하는 노드
    - 마지막 사용자 메시지를 분석하여 의도 분류
    - 간단한 키워드 기반 분류 (실제로는 LLM이나 분류 모델 사용 가능)
    
    Args:
        state: 현재 상태
    
    Returns:
        next_step 필드에 의도가 설정된 상태
    """
    messages = state["messages"]
    
    # 마지막 사용자 메시지 찾기
    last_user_message = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_user_message = msg.content.lower()
            break
    
    if not last_user_message:
        return {"next_step": "unknown"}
    
    # 키워드 기반 의도 분류
    if "안녕" in last_user_message or "인사" in last_user_message:
        next_step = "greeting"
    elif "질문" in last_user_message or "?" in last_user_message:
        next_step = "question"
    elif "감사" in last_user_message or "고마" in last_user_message:
        next_step = "thanks"
    else:
        next_step = "general"
    
    return {"next_step": next_step}


def handle_greeting(state: RouterState) -> RouterState:
    """
    인사 처리 노드
    
    Args:
        state: 현재 상태
    
    Returns:
        인사 응답이 추가된 상태
    """
    response = AIMessage(content="안녕하세요! 무엇을 도와드릴까요?")
    return {"messages": [response]}


def handle_question(state: RouterState) -> RouterState:
    """
    질문 처리 노드
    - LLM을 사용하여 질문에 답변
    
    Args:
        state: 현재 상태
    
    Returns:
        AI 응답이 추가된 상태
    """
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


def handle_thanks(state: RouterState) -> RouterState:
    """
    감사 인사 처리 노드
    
    Args:
        state: 현재 상태
    
    Returns:
        감사 응답이 추가된 상태
    """
    response = AIMessage(content="천만에요! 다른 도움이 필요하시면 언제든 말씀해주세요.")
    return {"messages": [response]}


def handle_general(state: RouterState) -> RouterState:
    """
    일반 대화 처리 노드
    - LLM을 사용하여 일반적인 대화 처리
    
    Args:
        state: 현재 상태
    
    Returns:
        AI 응답이 추가된 상태
    """
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


# ============================================================================
# 조건부 라우팅 함수
# ============================================================================
# 조건부 엣지를 사용하려면 라우팅 함수를 정의해야 합니다.
# 이 함수는 State를 받아서 다음 노드 이름을 반환합니다.
# ============================================================================

def route_to_next_node(state: RouterState) -> Literal["greeting", "question", "thanks", "general", END]:
    """
    상태에 따라 다음 노드를 결정하는 라우팅 함수
    
    Args:
        state: 현재 상태
    
    Returns:
        다음에 실행할 노드 이름 또는 END
    """
    next_step = state.get("next_step", "general")
    
    # next_step 값에 따라 다음 노드 결정
    if next_step == "greeting":
        return "greeting"
    elif next_step == "question":
        return "question"
    elif next_step == "thanks":
        return "thanks"
    elif next_step == "general":
        return "general"
    else:
        return END  # 알 수 없는 경우 종료


# ============================================================================
# 그래프 구성: 조건부 라우팅 포함
# ============================================================================

workflow = StateGraph(RouterState)

# 노드 추가
workflow.add_node("classify", classify_intent)      # 의도 분류 노드
workflow.add_node("greeting", handle_greeting)      # 인사 처리 노드
workflow.add_node("question", handle_question)      # 질문 처리 노드
workflow.add_node("thanks", handle_thanks)         # 감사 처리 노드
workflow.add_node("general", handle_general)       # 일반 대화 처리 노드

# 엣지 설정
workflow.set_entry_point("classify")  # 시작: 의도 분류

# 조건부 엣지 추가
# add_conditional_edges(이전노드, 라우팅함수, {조건값: 다음노드})
# - 첫 번째 인자: 출발 노드
# - 두 번째 인자: 라우팅 함수 (State를 받아 다음 노드 이름 반환)
# - 세 번째 인자: 딕셔너리 매핑 (라우팅 함수의 반환값 → 실제 노드 이름)
workflow.add_conditional_edges(
    "classify",  # 출발 노드
    route_to_next_node,  # 라우팅 함수
    {
        "greeting": "greeting",  # route_to_next_node가 "greeting" 반환 → "greeting" 노드로
        "question": "question",
        "thanks": "thanks",
        "general": "general",
        END: END  # 직접 END 반환도 가능
    }
)

# 모든 처리 노드에서 종료
workflow.add_edge("greeting", END)
workflow.add_edge("question", END)
workflow.add_edge("thanks", END)
workflow.add_edge("general", END)

# 그래프 컴파일
app = workflow.compile()


# ============================================================================
# 실행 예제
# ============================================================================

test_cases = [
    "안녕하세요!",
    "지구의 자전 주기는 얼마인가요?",
    "감사합니다!",
    "오늘 날씨가 좋네요."
]

for test_message in test_cases:
    print("\n" + "=" * 80)
    print(f"테스트 메시지: {test_message}")
    print("=" * 80)
    
    initial_state = {
        "messages": [HumanMessage(content=test_message)],
        "next_step": ""
    }
    
    result = app.invoke(initial_state)
    
    print("\n대화 내용:")
    for message in result["messages"]:
        if isinstance(message, HumanMessage):
            print(f"[사용자]: {message.content}")
        elif isinstance(message, AIMessage):
            print(f"[AI]: {message.content}")
    
    print(f"\n분류된 의도: {result.get('next_step', 'N/A')}")
    print("-" * 80)


# ============================================================================
# 참고: 조건부 엣지의 다양한 활용
# ============================================================================
# 1. 단순 분기: 조건에 따라 다른 노드로 라우팅
# 2. 루프 구현: 같은 노드로 다시 돌아가도록 설정
# 3. 다중 조건: 여러 조건을 체크하여 복잡한 라우팅
# 4. 동적 라우팅: State의 값에 따라 동적으로 노드 선택
#
# 예시: 루프 구현
#   def should_continue(state):
#       if state["count"] < 5:
#           return "loop_node"  # 같은 노드로 다시
#       return END
#
#   workflow.add_conditional_edges("loop_node", should_continue, {
#       "loop_node": "loop_node",
#       END: END
#   })
# ============================================================================

