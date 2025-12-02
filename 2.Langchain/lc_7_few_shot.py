import os

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate, ChatPromptTemplate, \
    FewShotChatMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings


# load env
load_dotenv()

# API KEY
api_key = os.getenv("OPENAI_API_KEY")

# model
llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0.0)

# parser
output_parser = StrOutputParser()


# Few-shot: 예시를 제공하여 LLM이 원하는 형식으로 응답하도록 유도 (고정 예시 또는 의미 유사도 기반 동적 선택)
# - LLM에게 몇 가지 예시(Q&A)를 제공해, 원하는 방식으로 응답하도록 유도하는 기법이다.
# - 프롬프트 기반 지식이나 패턴 유도에 주로 사용된다.

# 사용 예:
# 1) 질문 표현은 다양하지만, 정답은 하나로 고정되어야 할 때
# 2) 외부 검색이나 데이터베이스 없이, 프롬프트만으로 정답을 유도하고자 할 때
# 3) 모델이 특정 질문에서 오답을 잘 낼 경우, 예시로 올바른 답변 형식을 교정할 때

# → 특정 도메인, 질문 형식, 어조 등에서 모델의 일관된 성능 향상에 효과적이다.


# # Few-shot formatter
example_prompt1 = PromptTemplate.from_template("Q: {question}\nA: {answer}\n")
#
# 예제 세트
examples1 = [
    {
        "question": "지구의 대기 중 가장 많은 기체는?",
        "answer": "지구 대기의 약 78%를 차지하는 질소입니다."
    },
    {
        "question": "지구 대기 구성 물질 중 가장 많은 것은?",
        "answer": "지구 대기의 약 78%를 차지하는 질소입니다."
    },
    {
        "question": "공기 중에 가장 많은 기체는?",
        "answer": "주성분은 질소이며, 이는 지구 대기의 대부분을 차지합니다"
    },
    {
        "question": "광합성에 필요한 주요 요소들은 무엇인가요?",
        "answer": "광합성에 필요한 주요 요소는 빛, 이산화탄소, 물입니다."
    },
    {
        "question": "피타고라스 정리를 설명해주세요.",
        "answer": "피타고라스 정리는 직각삼각형에서 빗변의 제곱이 다른 두 변의 제곱의 합과 같다는 것입니다."
    },
    {
        "question": "지구의 자전 주기는 얼마인가요?",
        "answer": "지구의 자전 주기는 약 24시간(정확히는 23시간 56분 4초)입니다."
    },
    {
        "question": "DNA의 기본 구조를 간단히 설명해주세요.",
        "answer": "DNA는 두 개의 폴리뉴클레오티드 사슬이 이중 나선 구조를 이루고 있습니다."
    },
    {
        "question": "원주율(π)의 정의는 무엇인가요?",
        "answer": "원주율(π)은 원의 지름에 대한 원의 둘레의 비율입니다."
    }
]
#
# prompt = FewShotPromptTemplate(
#     examples=examples1,                                # 예제 데이터 목록
#     example_prompt=example_prompt1,                    # 각 예제를 어떤 형식으로 보여줄지 정의
#     prefix="다음은 과학과 수학 질문에 대한 예시입니다:",     # 예제 앞에 추가될 접미사 -> LLM에게 예제를 소개하는 문구
#     suffix="Q: {input}\nA:",                          # 예제 뒤에 추가될 접미사 -> 실제 사용자 입력 질문이 들어갈 위치
#     input_variables=["input"],                        # 최종 질문에서 들어올 변수명
# )
#
# chain = prompt | llm | output_parser
# result = chain.invoke({"input": "지구 공기의 주 성분?"})
# print(result)

# # --------------------------------------------------------------------

# SemanticSimilarityExampleSelector
# # 의미적 유사성을 기반으로 가장 관련성 높은 응답을 선택하여 응답 품질을 향상시킨다.
# example_selector1 = SemanticSimilarityExampleSelector.from_examples(
#     examples1,  # 사용할 예제들
#     OpenAIEmbeddings(),  # 임베딩 모델
#     Chroma,  # 벡터 저장소
#     k=1,  # 선택할 예제 수
# )



# 새로운 질문에 대해 가장 유사한 예제를 선택한다.
question = "화성의 대기 구조는 어떻게 되나요?"
# selected_examples1 = example_selector1.select_examples({"question": question})
# print(f"\n입력 질문: {question}")
# print("유사한 예제:")
# for ex in selected_examples1:
#     print(f"- {ex['question']} → {ex['answer']}")


# prompt = FewShotPromptTemplate(
#     example_selector=example_selector1,                #
#     example_prompt=example_prompt1,                    # 각 예제를 어떤 형식으로 보여줄지 정의
#     prefix="다음은 과학과 수학 질문에 대한 예시입니다:",     # 예제 앞에 추가될 접미사 -> LLM에게 예제를 소개하는 문구
#     suffix="Q: {input}\nA:",                          # 예제 뒤에 추가될 접미사 -> 실제 사용자 입력 질문이 들어갈 위치
#     input_variables=["input"],                        # 최종 질문에서 들어올 변수명
# )

# prompt.format(input="화성의 대기 구조는 어떻게 되나요?")

## 실제 실행
# chain1 = prompt | llm | output_parser
# result = chain1.invoke({"input": question})
# print("\n🟢 최종 응답:")
# print(result)

# # --------------------------------------------------------------------

# Fixed Few-shot in ChatPrompt

# 예제 정의
examples2 = [
    {"input": "지구의 대기 중 가장 많은 비율을 차지하는 기체는 무엇인가요?", "output": "질소입니다."},
    {"input": "광합성에 필요한 주요 요소들은 무엇인가요?", "output": "빛, 이산화탄소, 물입니다."},
]

# 예제 프롬프트 템플릿 정의
example_prompt2 = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

# Few-shot 프롬프트 템플릿 생성
few_shot_prompt2 = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt2,
    examples=examples2,
)

# 최종 프롬프트 템플릿 생성
final_prompt2 = ChatPromptTemplate.from_messages(
    [
        ("system", "당신은 과학과 수학에 대해 잘 아는 교육자입니다."),
        few_shot_prompt2,
        ("human", "{input}"),
    ]
)

# chain2 = final_prompt2 | llm
# result = chain2.invoke({"input": "지구의 자전 주기는 얼마인가요?"})
# print(result.content)




# Dynamic Few-shot in ChatPrompt

# SemanticSimilarityExampleSelector는 입력과 가장 유사한 예시 k개를 벡터스토어에서 선택한다.
# 선택된 예시는 ChatPromptTemplate 형식으로 포맷되어, 실제 사용자 질문 전에 LLM에게 힌트로 제공된다.

# 더 많은 예제 추가
examples3 = [
    {"input": "지구의 대기 중 가장 많은 비율을 차지하는 기체는 무엇인가요?", "output": "질소입니다."},
    {"input": "광합성에 필요한 주요 요소들은 무엇인가요?", "output": "빛, 이산화탄소, 물입니다."},
    {"input": "피타고라스 정리를 설명해주세요.", "output": "직각삼각형에서 빗변의 제곱은 다른 두 변의 제곱의 합과 같습니다."},
    {"input": "DNA의 기본 구조를 간단히 설명해주세요.", "output": "DNA는 이중 나선 구조를 가진 핵산입니다."},
    {"input": "원주율(π)의 정의는 무엇인가요?", "output": "원의 둘레와 지름의 비율입니다."},
]

# 벡터 저장소 생성
to_vectorize = [" ".join(example.values()) for example in examples3]
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=examples3)

# 예제 선택기 생성
example_selector3 = SemanticSimilarityExampleSelector(
    vectorstore=vectorstore,
    k=2,
)

# Few-shot 프롬프트 템플릿 생성
few_shot_prompt3 = FewShotChatMessagePromptTemplate(
    example_selector=example_selector3,
    example_prompt=ChatPromptTemplate.from_messages(
        [("human", "{input}"), ("ai", "{output}")]
    ),
)

# 최종 프롬프트 템플릿 생성
final_prompt3 = ChatPromptTemplate.from_messages(
    [
        ("system", "당신은 과학과 수학에 대해 잘 아는 교육자입니다."),
        few_shot_prompt3,
        ("human", "{input}"),
    ]
)

# 모델과 체인 생성
chain3 = final_prompt3 | llm

# 모델에 질문하기
result = chain3.invoke({"input": "태양계에서 가장 큰 행성은 무엇인가요?"})
print(result.content)