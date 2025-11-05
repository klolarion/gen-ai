import os
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# load env
load_dotenv()

# API KEY
api_key = os.getenv("OPENAI_API_KEY")

# model
llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0.7)


# ============================================================================
# 실전 예제: 대화형 챗봇 구현
# ============================================================================
# 이 예제는 LangGraph를 사용하여 완전한 대화형 챗봇을 구현합니다.
# 특징:
# 1. 대화 기록 유지
# 2. 시스템 프롬프트로 챗봇 역할 설정
# 3. 조건부 라우팅으로 대화 종료 감지
# 4. 상태 관리로 대화 턴 추적
# ============================================================================


class ChatbotState(TypedDict):
    """
    챗봇 상태 구조
    
    messages: 대화 메시지 리스트 (자동 병합)
    turn_count: 현재 대화 턴 수
    should_end: 대화 종료 여부
    """
    messages: Annotated[list, add_messages]
    turn_count: int
    should_end: bool


# ============================================================================
# 노드 함수들
# ============================================================================

def initialize_chat(state: ChatbotState) -> ChatbotState:
    """
    챗봇 초기화 노드
    - 시스템 메시지를 추가하여 챗봇의 역할 설정
    - turn_count 초기화
    
    Args:
        state: 현재 상태
    
    Returns:
        초기화된 상태
    """
    system_message = SystemMessage(
        content="당신은 친절하고 도움이 되는 AI 어시스턴트입니다. "
                "사용자의 질문에 정확하고 자세하게 답변해주세요. "
                "한국어로 대화합니다."
    )
    
    return {
        "messages": [system_message],
        "turn_count": 0,
        "should_end": False
    }


def check_should_end(state: ChatbotState) -> ChatbotState:
    """
    대화 종료 여부를 확인하는 노드
    - 마지막 사용자 메시지에서 종료 키워드 확인
    
    Args:
        state: 현재 상태
    
    Returns:
        should_end 플래그가 설정된 상태
    """
    messages = state["messages"]
    
    # 마지막 사용자 메시지 찾기
    last_user_message = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_user_message = msg.content.lower()
            break
    
    # 종료 키워드 확인
    end_keywords = ["종료", "끝", "그만", "bye", "안녕히", "나가기", "exit", "quit"]
    should_end = any(keyword in last_user_message for keyword in end_keywords) if last_user_message else False
    
    return {"should_end": should_end}


def generate_response(state: ChatbotState) -> ChatbotState:
    """
    AI 응답 생성 노드
    - 전체 대화 기록을 포함하여 LLM 호출
    - turn_count 증가
    
    Args:
        state: 현재 상태
    
    Returns:
        AI 응답이 추가되고 turn_count가 증가한 상태
    """
    messages = state["messages"]
    turn_count = state.get("turn_count", 0)
    
    # LLM 호출 (전체 대화 기록 포함)
    response = llm.invoke(messages)
    
    return {
        "messages": [response],
        "turn_count": turn_count + 1
    }


def handle_goodbye(state: ChatbotState) -> ChatbotState:
    """
    작별 인사 노드
    - 대화 종료 시 작별 메시지 추가
    
    Args:
        state: 현재 상태
    
    Returns:
        작별 메시지가 추가된 상태
    """
    goodbye_message = AIMessage(
        content="대화해주셔서 감사합니다! 또 만나요! 👋"
    )
    
    return {"messages": [goodbye_message]}


# ============================================================================
# 조건부 라우팅 함수
# ============================================================================

def route_after_check(state: ChatbotState) -> Literal["generate", "goodbye", END]:
    """
    종료 확인 후 라우팅 함수
    
    Args:
        state: 현재 상태
    
    Returns:
        다음 노드 이름
    """
    if state.get("should_end", False):
        return "goodbye"
    else:
        return "generate"


def route_after_response(state: ChatbotState) -> Literal["check_end", END]:
    """
    응답 생성 후 라우팅 함수
    
    Args:
        state: 현재 상태
    
    Returns:
        다음 노드 이름
    """
    # 대화가 너무 길면 종료 (예: 10턴 이상)
    if state.get("turn_count", 0) >= 10:
        return END
    
    return "check_end"


# ============================================================================
# 그래프 구성
# ============================================================================

workflow = StateGraph(ChatbotState)

# 노드 추가
workflow.add_node("init", initialize_chat)
workflow.add_node("check_end", check_should_end)
workflow.add_node("generate", generate_response)
workflow.add_node("goodbye", handle_goodbye)

# 엣지 설정
workflow.set_entry_point("init")
workflow.add_edge("init", "check_end")

# 조건부 엣지: 종료 확인 후 분기
workflow.add_conditional_edges(
    "check_end",
    route_after_check,
    {
        "generate": "generate",
        "goodbye": "goodbye",
        END: END
    }
)

# 조건부 엣지: 응답 생성 후 분기
workflow.add_conditional_edges(
    "generate",
    route_after_response,
    {
        "check_end": "check_end",
        END: END
    }
)

# 작별 인사 후 종료
workflow.add_edge("goodbye", END)

# 그래프 컴파일
app = workflow.compile()


# ============================================================================
# 대화형 실행 함수
# ============================================================================

def run_chatbot():
    """
    대화형 챗봇 실행 함수
    - 사용자 입력을 받아 그래프 실행
    - 대화 기록 유지
    """
    print("=" * 80)
    print("LangGraph 챗봇에 오신 것을 환영합니다!")
    print("종료하려면 '종료', '끝', '그만' 등을 입력하세요.")
    print("=" * 80)
    print()
    
    # 초기 상태
    current_state = {
        "messages": [],
        "turn_count": 0,
        "should_end": False
    }
    
    # 초기화 실행
    current_state = app.invoke(current_state)
    
    # 초기 인사 메시지 출력
    for message in current_state["messages"]:
        if isinstance(message, AIMessage):
            print(f"[챗봇]: {message.content}")
            print()
    
    # 대화 루프
    while not current_state.get("should_end", False) and current_state.get("turn_count", 0) < 10:
        # 사용자 입력 받기
        user_input = input("[사용자]: ").strip()
        
        if not user_input:
            continue
        
        # 사용자 메시지 추가
        current_state["messages"].append(HumanMessage(content=user_input))
        
        # 그래프 실행 방식 1: 노드 함수 직접 호출 (간단한 방식)
        # 주의: 실제 프로덕션에서는 그래프를 다시 실행하거나 스트림을 사용하는 것이 좋습니다
        # 여기서는 학습 목적으로 노드 함수를 직접 호출하여 상태를 업데이트합니다
        
        # check_end 노드 실행
        check_result = check_should_end(current_state)
        current_state.update(check_result)
        
        if current_state.get("should_end", False):
            # goodbye 노드 실행
            goodbye_result = handle_goodbye(current_state)
            current_state.update(goodbye_result)
            print(f"[챗봇]: {current_state['messages'][-1].content}")
            break
        
        # generate 노드 실행
        generate_result = generate_response(current_state)
        current_state.update(generate_result)
        
        # AI 응답 출력
        print(f"[챗봇]: {current_state['messages'][-1].content}")
        print()
    
    print("\n대화가 종료되었습니다. 감사합니다!")


# ============================================================================
# 단일 실행 예제 (대화형 아님)
# ============================================================================

def run_single_example():
    """
    단일 메시지로 그래프 실행 예제
    """
    print("\n" + "=" * 80)
    print("단일 실행 예제")
    print("=" * 80)
    
    initial_state = {
        "messages": [HumanMessage(content="안녕하세요! 파이썬에 대해 설명해주세요.")],
        "turn_count": 0,
        "should_end": False
    }
    
    result = app.invoke(initial_state)
    
    print("\n대화 내용:")
    print("-" * 80)
    for message in result["messages"]:
        if isinstance(message, SystemMessage):
            print(f"[시스템]: {message.content[:50]}...")
        elif isinstance(message, HumanMessage):
            print(f"[사용자]: {message.content}")
        elif isinstance(message, AIMessage):
            print(f"[챗봇]: {message.content}")
    print("-" * 80)
    print(f"\n총 대화 턴: {result.get('turn_count', 0)}")


# ============================================================================
# 실행
# ============================================================================

if __name__ == "__main__":
    # 단일 실행 예제
    run_single_example()
    
    # 대화형 실행 (주석 해제하여 사용)
    # run_chatbot()


# ============================================================================
# 참고: 개선 사항
# ============================================================================
# 1. 스트리밍 응답: LLM 응답을 실시간으로 스트리밍
# 2. 메모리 관리: 대화가 길어질 때 오래된 메시지 제거
# 3. 에러 처리: LLM 호출 실패 시 재시도 로직
# 4. 로깅: 대화 기록을 파일로 저장
# 5. 멀티 에이전트: 여러 전문 에이전트를 조합
# 6. 도구 사용: 외부 API, 계산기 등 도구 호출
# ============================================================================

