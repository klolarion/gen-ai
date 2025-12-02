import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnablePick
from langchain_core.output_parsers import StrOutputParser

# ============================================================================
# LCEL Data Manipulation: 체인 내부의 데이터 흐름 제어하기
# ============================================================================
# 왜 필요한가?
# 1. Chain은 기본적으로 앞 단계의 '출력'을 뒷 단계의 '입력'으로 덮어쓴다.
# 2. 하지만 뒷 단계에서 원래 질문과 검색된 문서가 동시에 필요하다면?
# 3. 데이터를 잃어버리지 않고 딕셔너리 형태로 누적(Assign)하거나,
#    필요한 것만 선택(Pick)해서 전달하는 기술이 필요하다.
#
# 활용 사례:
# - RAG(검색 증강 생성): 질문(question)을 유지하면서 검색 결과(context)를 추가할 때.
# - 대화 기록(History): 사용자 입력에 대화 기록(chat_history)을 덧붙여 프롬프트로 보낼 때.
# - API 응답 처리: 복잡한 JSON 결과에서 특정 필드(예: answer)만 뽑아서 클라이언트에 줄 때.
# ============================================================================

# load env
load_dotenv()

# LLM & Parser
llm = ChatOpenAI(model="gpt-4o-mini")
output_parser = StrOutputParser()

# 🔍 디버깅 헬퍼: 체인 중간 데이터 확인용
def debug_step(step_name):
    def _print_data(x):
        print(f"\n👀 [Debug] {step_name}")
        print(f"   데이터: {x}")
        return x
    return RunnableLambda(_print_data)

# ------------------------------------------------------------------
# 1. 기본: RunnablePassthrough.assign() - "데이터 누적하기"
# ------------------------------------------------------------------
# 설명: 입력받은 딕셔너리에 새로운 키-값을 추가합니다. (기존 데이터 삭제 X)
# 흐름: {key1: val1}  --->  assign(key2=func)  --->  {key1: val1, key2: val2}

print("="*60)
print("1. RunnablePassthrough.assign() - 데이터 누적 예제")
print("="*60)

def get_user_info(input_dict):
    # DB에서 사용자 정보를 가져오는 것을 가정
    return "VIP_Member" if input_dict.get("user_id") == "user_123" else "Standard_Member"

# 체인 구성
# 1. 입력: {"user_id": "...", "query": "..."}
# 2. assign: 입력값은 그대로 두고, 'user_grade'라는 필드만 계산해서 추가함
chain_with_assign = (
    debug_step("1. 초기 입력 데이터")
    | RunnablePassthrough.assign(user_grade=get_user_info)
    | debug_step("2. assign('user_grade') 실행 후 데이터")
)

# 실행
print(">>> 실행 결과:")
result = chain_with_assign.invoke({"user_id": "user_123", "query": "환불 규정이 어떻게 되나요?"})
# 결과 딕셔너리에 user_id, query는 그대로 있고 user_grade가 추가됨


# ------------------------------------------------------------------
# 2. 실전 응용: RAG 패턴 (질문 + 문서 동시에 전달하기)
# ------------------------------------------------------------------
# 가장 많이 쓰는 패턴입니다.
# 프롬프트에는 {question}과 {context} 두 가지 변수가 필요하다.
# 리트리버(검색기)는 {context}만 찾아준다. 이때 {question}을 잃어버리면 안됨  → 문서 검색 결과에 질문이 포함되어야 함.

print("\n" + "="*60)
print("2. 실전 응용: RAG 패턴 (질문 보존 + 문서 추가)")
print("="*60)

# 가상의 문서 검색 함수 (Retriever)
def fake_retriever(query):
    print(f"   (시스템: '{query}'에 대한 문서를 검색합니다...)")
    return "랭체인(LangChain)은 LLM 애플리케이션 개발을 위한 프레임워크입니다."

# 프롬프트: 변수 2개 필요
rag_prompt = ChatPromptTemplate.from_template(
    "다음 문서를 참고하여 질문에 답하세요.\n\n[문서]: {context}\n\n[질문]: {question}"
)

# RAG 체인 구성
# input: {"question": "LangChain이 뭐야?"}
# step 1: assign(context=...) -> question은 유지하고, context 키에 검색 결과 추가
#         결과: {"question": "...", "context": "검색된 내용"}
# step 2: prompt -> 완성된 딕셔너리가 프롬프트의 {question}, {context}에 매핑됨
rag_chain = (
    debug_step("1. 초기 질문 데이터")
    | RunnablePassthrough.assign(context=lambda x: fake_retriever(x["question"]))
    | debug_step("2. assign('context') 실행 후 (프롬프트 입력값)")
    | rag_prompt 
    | llm 
    | output_parser
)

# 실행
query = "LangChain이 뭐야?"
print(f"질문: {query}")
rag_result = rag_chain.invoke({"question": query})
print(f"답변: {rag_result}")


# ------------------------------------------------------------------
# 3. RunnablePick: 필요한 데이터만 뽑기
# ------------------------------------------------------------------
# 체인 중간이나 끝에서 너무 많은 정보가 딕셔너리에 쌓여있을 때,
# 원하는 키 값만 추출해서 다음 단계로 넘기거나 최종 출력.

print("\n" + "="*60)
print("3. RunnablePick - 데이터 추출 예제")
print("="*60)

# 가상의 복잡한 API 응답 (이전 단계의 결과라고 가정)
complex_output = {
    "status": 200,
    "metadata": {"latency": 0.5, "tokens": 150},
    "content": {
        "answer": "서울입니다.",
        "sources": ["wiki", "news"]
    }
}

# 체인: 전체 데이터 -> 'content' 추출 -> 그 안에서 'answer' 추출
# 딕셔너리 접근법: complex_output["content"]["answer"] 와 같음
pick_chain = (
    debug_step("1. 원본 데이터")
    | RunnablePick("content") 
    | debug_step("2. Pick('content') 실행 후")
    | RunnablePick("answer")
    | debug_step("3. Pick('answer') 실행 후 (최종 결과)")
)

# 실행
# invoke에 들어가는 값이 위에서 정의한 딕셔너리라고 가정
result_pick = pick_chain.invoke(complex_output)
print(f"원본 데이터 키: {complex_output.keys()}")
print(f"Pick 결과: {result_pick}")
