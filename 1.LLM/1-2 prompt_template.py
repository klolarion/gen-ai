import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# 프롬프트 작성 규칙

# 1. 명확성과 구체성
# 질문은 명확하고 구체적이어야 한다.
# 예시: "주식 시장에 대해 알려줘." -> "다음 주 주식 시장에 영향을 줄 수 있는 예정된 이벤트들은 뭐가 있을까?"

# 2. 배경 정보를 포함
# 모델이 문맥을 이해할 수 있도록 필요한 배경 정보를 제공하는 것이 좋다.
# 이는 환각 현상이 발생할 위험을 낮추고, 관련성 높은 응답을 생성하는 데 도움을 준다.
# 예시: "2020년 미국 대선의 결과를 바탕으로 현재 정치 상황에 대한 분석을 해줘."

# 3. 간결함
# 핵심 정보에 초점을 맞추고, 불필요한 정보는 배제한다.
# 프롬프트가 길어지면 모델이 덜 중요한 부분에 집중하는 문제가 발생할 수 있다.
# 예시: "2021년에 발표된 삼성전자의 ESG 보고서를 요약해줘."

# 4. 열린 질문 사용
# 열린 질문을 통해 모델이 자세하고 풍부한 답변을 제공하도록 유도한다.
# 단순한 '예' 또는 '아니오'로 대답할 수 있는 질문보다는 더 많은 정보를 제공하는 질문이 좋다.
# 예시: "신재생에너지에 대한 최신 연구 동향은 뭐가 있을까?"

# 5. 명확한 목표 설정
# 얻고자 하는 정보나 결과의 유형을 정확하게 정의한다. 이는 모델이 명확한 지침에 따라 응답을 생성하도록 돕는다.
# 예시: "AI 윤리에 대한 문제점과 해결 방안을 요약하여 설명해줘."

# 6. 언어와 문체
# 대화의 맥락에 적합한 언어와 문체를 선택한다. 이는 모델이 상황에 맞는 표현을 선택하는데 도움이 된다.
# 예시: 공식적인 보고서를 요청하는 경우, "XX 보고서에 대한 전문적인 요약을 부탁드립니다."와 같이 정중한 문체를 사용한다.


# 지시	언어 모델에게 어떤 작업을 수행하도록 요청하는 구체적인 지시.
# 예시	요청된 작업을 수행하는 방법에 대한 하나 이상의 예시.
# 맥락	특정 작업을 수행하기 위한 추가적인 맥락
# 질문	어떤 답변을 요구하는 구체적인 질문.

# 예시: 제품 리뷰 요약
# 지시: "아래 제공된 제품 리뷰를 요약해주세요."
# 예시: "예를 들어, '이 제품은 매우 사용하기 편리하며 배터리 수명이 길다'라는 리뷰는 '사용 편리성과 긴 배터리 수명이 특징'으로 요약할 수 있습니다."
# 맥락: "리뷰는 스마트워치에 대한 것이며, 사용자 경험에 초점을 맞추고 있습니다."
# 질문: "이 리뷰를 바탕으로 스마트워치의 주요 장점을 두세 문장으로 요약해주세요."

# load env
load_dotenv()

# API KEY
api_key = os.getenv("OPENAI_API_KEY")

# model
llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)

# parser
output_parser = StrOutputParser()

# PromptTemplate

# 문자열 템플릿
# 'name'과 'age'라는 두 개의 변수를 사용하는 프롬프트 템플릿을 정의
template_text = "안녕하세요, 제 이름은 {name}이고, 나이는 {age}살입니다."

# PromptTemplate 인스턴스를 생성
prompt_template = PromptTemplate.from_template(template_text)

# 템플릿에 값을 채워서 프롬프트를 완성
filled_prompt = prompt_template.format(name="홍길동", age=30)

print(filled_prompt)

# 프롬프트 결합
# 문자열 템플릿 결합 (PromptTemplate + PromptTemplate + 문자열)
combined_prompt = (
              prompt_template
              + PromptTemplate.from_template("\n\n아버지를 아버지라 부를 수 없습니다.")
              + "\n\n{language}로 번역해주세요."
)

PromptTemplate(
    input_variables=['age', 'language', 'name'],
    template='안녕하세요, 제 이름은 {name}이고, 나이는 {age}살입니다.\n\n아버지를 아버지라 부를 수 없습니다.\n\n{language}로 번역해주세요.'
)
combined_prompt.format(name="홍길동", age=30, language="영어")

chain = combined_prompt | llm | output_parser

result = chain.invoke({"age":30, "language":"영어", "name":"홍길동"})

print(result)


