import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser, JsonOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

# ============================================================================
# Output Parser: LLM의 원시 텍스트 출력을 구조화된 형식(리스트, JSON, 객체 등)으로 변환
# ============================================================================
# 1. StrOutputParser: 기본 문자열 반환 - 텍스트 처리 등 기본적인 태스크용
# 2. CommaSeparatedListOutputParser: 콤마로 구분된 리스트 반환 - 키워드 추출 등 간단한 태스크용
# 3. JsonOutputParser: JSON 포맷의 딕셔너리 반환 - 동적 타입 검증에 활용
# 4. PydanticOutputParser: Python 객체(Class Instance) 반환 - 타입 검증과 유효성 확인에 활용
# ============================================================================

# load env
load_dotenv()

# LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

print("="*60)
print("1. Comma Separated List Parser (리스트 변환)")
print("="*60)

# 1. 파서 생성
list_parser = CommaSeparatedListOutputParser()
list_format_instructions = list_parser.get_format_instructions()

# 2. 프롬프트 생성
list_prompt = PromptTemplate(
    template="List five {subject}.\n{format_instructions}",
    input_variables=["subject"],
    partial_variables={"format_instructions": list_format_instructions},
)

# 3. 체인 실행
list_chain = list_prompt | llm | list_parser
list_result = list_chain.invoke({"subject": "popular Korean food"})

print(f"질문: 'popular Korean food'")
print(f"결과 타입: {type(list_result)}")
print(f"결과 값: {list_result}")


print("\n" + "="*60)
print("2. JSON Output Parser (딕셔너리 변환)")
print("="*60)

# 자료구조 정의 (스키마용)
class CuisineRecipe(BaseModel):
    name: str = Field(description="name of the cuisine")
    recipe: str = Field(description="step-by-step recipe to cook the cuisine")
    ingredients: List[str] = Field(description="list of ingredients")

# 1. 파서 생성
json_parser = JsonOutputParser(pydantic_object=CuisineRecipe)
json_format_instructions = json_parser.get_format_instructions()

# 2. 프롬프트 생성
json_prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\nQuery: {query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": json_format_instructions},
)

# 3. 체인 실행
json_chain = json_prompt | llm | json_parser
json_result = json_chain.invoke({"query": "How to cook Bibimbap?"})

print(f"질문: 'How to cook Bibimbap?'")
print(f"결과 타입: {type(json_result)}")  # <class 'dict'>
print(f"요리명: {json_result['name']}")
print(f"재료: {json_result['ingredients']}")


print("\n" + "="*60)
print("3. Pydantic Output Parser (객체 변환)")
print("="*60)
# JsonOutputParser는 dict를 반환하지만, PydanticOutputParser는 객체 자체를 반환합니다.
# 더 엄격한 타입 검사와 객체 메서드 활용이 가능합니다.

# 자료구조 정의 (실제 객체 생성용)
class MovieReview(BaseModel):
    title: str = Field(description="영화 제목")
    genre: str = Field(description="영화 장르")
    rating: float = Field(description="영화 평점 (1-10점)")
    summary: str = Field(description="영화 한 줄 요약")

# 1. 파서 생성
pydantic_parser = PydanticOutputParser(pydantic_object=MovieReview)
pydantic_format_instructions = pydantic_parser.get_format_instructions()

# 2. 프롬프트 생성
pydantic_prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\nQuery: {query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": pydantic_format_instructions},
)

# 3. 체인 실행
pydantic_chain = pydantic_prompt | llm | pydantic_parser
pydantic_result = pydantic_chain.invoke({"query": "영화 '인셉션'에 대해 알려줘."})

print(f"질문: '영화 '인셉션'에 대해 알려줘.'")
print(f"결과 타입: {type(pydantic_result)}")  # <class '__main__.MovieReview'>
print(f"제목: {pydantic_result.title}")  # 객체 속성으로 접근
print(f"평점: {pydantic_result.rating}")
print(f"요약: {pydantic_result.summary}")

# 객체를 다시 딕셔너리로 변환 가능
print("\n[객체 -> 딕셔너리 변환]")
print(pydantic_result.model_dump())
