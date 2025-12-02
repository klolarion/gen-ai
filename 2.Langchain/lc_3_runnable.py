import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

import nest_asyncio
import asyncio

# ============================================================================
# Runnable: LangChain의 모든 컴포넌트가 공통으로 구현하는 인터페이스 (invoke, batch, stream, ainvoke)
# Runnable이어야 체인에 연결 가능하다.
# ============================================================================

# load env
load_dotenv()

# API KEY
api_key = os.getenv("OPENAI_API_KEY")

# model
llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)

# parser
output_parser = StrOutputParser()

# prompt
prompt = ChatPromptTemplate.from_template("지구과학에서 {topic}에 대해 간단히 설명해주세요.")

# chain
chain = prompt | llm | output_parser

# --------------------------------------------------------------------
# invoke: 단일 입력 처리 (동기)
# --------------------------------------------------------------------
# resultI = chain.invoke({"topic": "지구 자전"})
# print(resultI)

# --------------------------------------------------------------------
# batch: 여러 입력을 한 번에 처리 (동기)
# --------------------------------------------------------------------
# topics = ["지구 공전", "화산 활동", "대륙 이동"]
# resultB = chain.batch([{"topic": t} for t in topics])
# for topic, resultB in zip(topics, resultB):
#     print(f"{topic} 설명: {resultB[:50]}...")  # 결과의 처음 50자만 출력

# --------------------------------------------------------------------
# stream: 실시간 스트리밍으로 토큰 단위 응답 받기
# --------------------------------------------------------------------
stream = chain.stream({"topic": "지진"}) # LLM의 응답을 한 글자/토큰 단위로 실시간 스트리밍 시작
print("stream 결과:")
for chunk in stream: # 응답을 순차적으로 받아 출력
    print(chunk, end="", flush=True) # 줄바꿈 없이 계속 출력되도록 함
#    chunk: LLM이 스트리밍으로 보내온 문자열 (보통 한 단어/문장 또는 토큰)
#    end="" : 기본값 "\n" → 줄바꿈 발생 → ""으로 지정하면 줄바꿈 없이 이어서 출력
#    flush=True : 	버퍼에 남아 있는 내용을 즉시 콘솔에 출력 → 지연 없이 타자 치듯 출력됨
print()

# --------------------------------------------------------------------
# async method: 비동기 실행 (ainvoke)
# --------------------------------------------------------------------
# nest_asyncio.apply()
#
# async def run_async():
#     result = await chain.ainvoke({"topic": "해류"})
#     print("ainvoke 결과:", result[:50], "...")
#
# asyncio.run(run_async())