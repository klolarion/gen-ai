import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# load env
load_dotenv()

# API KEY
api_key = os.getenv("OPENAI_API_KEY")

# parser
output_parser = StrOutputParser()

# Temperature: 생성된 텍스트의 다양성을 조정한다. 0~1 사이의 값을 가지고 높은수록 답변의 다양성이 증가한다.
# Max Tokens (최대 토큰 수): 생성할 최대 토큰 수를 지정한다.
# Top P (Top Probability): 생성 과정에서 특정 확률 분포 내에서 상위 P% 토큰만을 고려하는 방식입니다. 이는 출력의 다양성을 조정하는 데 도움이 됩니다.
# Frequency Penalty (빈도 패널티): 값이 클수록 이미 등장한 단어나 구절이 다시 등장할 확률을 감소시킨다. 이를 통해 반복을 줄이고 텍스트의 다양성을 증가시킬 수 있다. (0~1)
# Presence Penalty (존재 패널티): 텍스트 내에서 단어의 존재 유무에 따라 그 단어의 선택 확률을 조정한다. 값이 클수록 아직 텍스트에 등장하지 않은 새로운 단어의 사용이 장려된다. (0~1)
# Stop Sequences (정지 시퀀스): 특정 단어나 구절이 등장할 경우 생성을 멈추도록 설정한다.

# 모델 파라미터 설정
params = {
    "temperature": 0.7,         # 생성된 텍스트의 다양성 조정 0~1. 낮을수록 정적.
    "max_tokens": 100,          # 생성할 최대 토큰 수
}

kwargs = {
    "frequency_penalty": 0.5,   # 이미 등장한 단어의 재등장 확률
    "presence_penalty": 0.5,    # 새로운 단어의 도입을 장려
    "stop": ["\n"]              # 정지 시퀀스 설정
}

# 모델 인스턴스를 생성할 때 설정
llm = ChatOpenAI(model="gpt-4o-mini", **params, **kwargs)

# 모델 호출
question = "태양계에서 가장 큰 행성은 무엇인가요?"
result = llm.invoke(input=question)

# 전체 응답 출력
print(result)

# # --------------------------------------------------------------------

# 모델 파라미터 설정
params = {
    "temperature": 0.7,         # 생성된 텍스트의 다양성 조정
    "max_tokens": 10,          # 생성할 최대 토큰 수
}

# 모델 인스턴스를 호출할 때 전달
result = llm.invoke(input=question, **params, **kwargs)

# 문자열 출력
print(result.content)