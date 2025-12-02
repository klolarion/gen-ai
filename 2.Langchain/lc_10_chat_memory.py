import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser

# ============================================================================
# Chat Memory: 대화의 맥락(Context)을 기억하는 체인 만들기
# ============================================================================
# 기본 LLM 체인은 Stateless(무상태)입니다. 즉, 이전 대화를 기억하지 못함.
# RunnableWithMessageHistory를 사용하여 대화 내역을 저장하고 관리하는 방법을 배움.
# ============================================================================

# load env
load_dotenv()

# LLM & Parser
llm = ChatOpenAI(model="gpt-4o-mini")
output_parser = StrOutputParser()

# 1. 프롬프트 템플릿 생성
# MessagesPlaceholder: 대화 내역이 들어갈 위치를 지정합니다.
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 친절한 AI 챗봇입니다."),
    MessagesPlaceholder(variable_name="chat_history"),  # 대화 내역이 여기에 주입됨
    ("human", "{input}")
])

# 2. 기본 체인 생성
chain = prompt | llm | output_parser

# 3. 대화 내역 저장소 (메모리) 설정
# 세션 ID별로 대화 내역을 저장할 딕셔너리
store = {}

def get_session_history(session_id: str):
    """세션 ID에 해당하는 대화 내역을 반환"""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 4. 메모리 기능이 추가된 체인 생성
# RunnableWithMessageHistory로 기존 체인을 래핑
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# ============================================================================
# 실행 테스트
# ============================================================================

print("="*60)
print("메모리 기능 테스트 (세션 ID: user1)")
print("="*60)

# 첫 번째 질문
query1 = "내 이름은 BMAPS야."
print(f"\n👤 사용자: {query1}")
response1 = chain_with_memory.invoke(
    {"input": query1},
    config={"configurable": {"session_id": "user1"}}
)
print(f"🤖 AI: {response1}")

# 두 번째 질문 (이전 대화 기억 확인)
query2 = "내 이름이 뭐였지?"
print(f"\n👤 사용자: {query2}")
response2 = chain_with_memory.invoke(
    {"input": query2},
    config={"configurable": {"session_id": "user1"}}
)
print(f"🤖 AI: {response2}")

# ============================================================================
# 다른 세션 테스트 (격리 확인)
# ============================================================================

print("\n" + "="*60)
print("다른 세션 테스트 (세션 ID: user2)")
print("="*60)

# 다른 세션 ID로 질문 (user1의 정보를 몰라야함)
print(f"\n👤 사용자 (user2): {query2}")
response3 = chain_with_memory.invoke(
    {"input": query2},
    config={"configurable": {"session_id": "user2"}}
)
print(f"🤖 AI: {response3}")

