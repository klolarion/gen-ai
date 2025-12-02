import os
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# ============================================================================
# RunnableParallel: 여러 작업을 동시에 병렬로 실행하기
# ============================================================================
# 하나의 입력에 대해 여러 가지 처리(예: 요약, 번역, 분석)를 동시에 수행할 때 사용한다.
# 처리 시간을 단축하고 다양한 관점의 결과를 한 번에 얻을 수 있다.
# ============================================================================

# load env
load_dotenv()

# LLM
llm = ChatOpenAI(model="gpt-4o-mini")
output_parser = StrOutputParser()

# 시간 측정 및 로깅을 위한 래퍼 함수
def log_execution_time(chain_name):
    def wrapper(input_data):
        start_time = time.time()
        print(f"🚀 [{chain_name}] 시작: {time.strftime('%H:%M:%S')}")
        
        # 실제 체인 실행 (여기서는 단순 호출을 흉내내기 위해 input_data를 그대로 넘김)
        # 실제로는 이 래퍼가 체인 내부에서 사용된다.
        return input_data
    return wrapper

# 1. 개별 체인 정의
# 각 체인의 실행 시간을 확인하기 위해 RunnableLambda를 사용하여 로깅 추가

# 체인 A: 주제에 대한 '장점' 분석
pros_prompt = ChatPromptTemplate.from_template("{topic}의 장점을 3가지 요약해줘.")
# 로깅을 위해 커스텀 함수 추가
pros_chain = (
    RunnableLambda(lambda x: (print(f"🚀 [장점 분석] 시작: {time.strftime('%H:%M:%S')}"), x)[1])
    | pros_prompt 
    | llm 
    | output_parser
    | RunnableLambda(lambda x: (print(f"✅ [장점 분석] 완료: {time.strftime('%H:%M:%S')}"), x)[1])
)

# 체인 B: 주제에 대한 '단점' 분석
cons_prompt = ChatPromptTemplate.from_template("{topic}의 단점을 3가지 요약해줘.")
cons_chain = (
    RunnableLambda(lambda x: (print(f"🚀 [단점 분석] 시작: {time.strftime('%H:%M:%S')}"), x)[1])
    | cons_prompt 
    | llm 
    | output_parser
    | RunnableLambda(lambda x: (print(f"✅ [단점 분석] 완료: {time.strftime('%H:%M:%S')}"), x)[1])
)

# 체인 C: 주제로 '시' 작성
poem_prompt = ChatPromptTemplate.from_template("{topic}를 주제로 짧은 시를 써줘.")
poem_chain = (
    RunnableLambda(lambda x: (print(f"🚀 [시  작성] 시작: {time.strftime('%H:%M:%S')}"), x)[1])
    | poem_prompt 
    | llm 
    | output_parser
    | RunnableLambda(lambda x: (print(f"✅ [시  작성] 완료: {time.strftime('%H:%M:%S')}"), x)[1])
)

# 2. 병렬 체인 구성 (RunnableParallel)
# 딕셔너리 형태로 각 작업의 키와 실행할 체인을 지정
parallel_chain = RunnableParallel(
    pros=pros_chain,
    cons=cons_chain,
    poem=poem_chain,
    original_topic=RunnablePassthrough()  # 원본 입력도 그대로 통과시켜 결과에 포함
)

# ============================================================================
# 실행 테스트
# ============================================================================

topic = "재택근무"

print("="*60)
print(f"병렬 처리 시작 (주제: {topic})")
print("장점 분석, 단점 분석, 시 작성을 동시에 수행합니다...")
print(f"전체 시작 시간: {time.strftime('%H:%M:%S')}")
print("="*60)

start_total = time.time()

# invoke 한 번으로 3가지 작업이 병렬 실행됨
result = parallel_chain.invoke({"topic": topic})

end_total = time.time()

print("\n" + "="*60)
print(f"전체 완료 시간: {time.strftime('%H:%M:%S')}")
print(f"총 소요 시간: {end_total - start_total:.2f}초")
print("="*60)

# 결과 출력
print("\n[1. 장점 분석]")
print(result['pros'])

print("\n[2. 단점 분석]")
print(result['cons'])

print("\n[3. 시 작성]")
print(result['poem'])

print("\n[4. 원본 데이터]")
print(result['original_topic'])

