import os
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

# ============================================================================
# Token Counting: API 비용 예측 및 컨텍스트 제한 관리
# ============================================================================
# LLM은 단어가 아닌 '토큰(Token)' 단위로 텍스트를 처리한다.
# API 비용은 토큰 수에 비례하며, 모델마다 처리 가능한 최대 토큰 수가 정해져 있음.
# tiktoken 라이브러리를 사용하여 정확한 토큰 수를 계산하는 방법을 배움.
# ============================================================================

# load env
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def num_tokens_from_string(string: str, model_name: str) -> int:
    """주어진 문자열의 토큰 수를 반환한다."""
    try:
        # 모델에 맞는 인코딩 방식 로드
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        # 모델 정보를 찾을 수 없으면 기본 인코딩(cl100k_base) 사용 (GPT-4, GPT-3.5용)
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    
    num_tokens = len(encoding.encode(string))
    return num_tokens

# 테스트할 텍스트
texts = [
    "Hello, world!",  # 영어 (짧음)
    "안녕하세요, 반갑습니다.",  # 한국어 (토큰 수가 더 많이 나옴)
    "Python is an interpreted, high-level, general-purpose programming language."  # 영어 (긺)
]

# 모델별 토큰 계산 테스트
models = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]

print("="*60)
print("토큰 수 계산 테스트")
print("="*60)

for text in texts:
    print(f"\n📝 텍스트: '{text}'")
    for model in models:
        token_count = num_tokens_from_string(text, model)
        print(f"   - [{model}]: {token_count} tokens")

# ============================================================================
# 실무 활용 예시: 예산에 맞춰 텍스트 자르기
# ============================================================================
print("\n" + "="*60)
print("활용 예시: 최대 토큰 제한 (Truncation)")
print("="*60)

long_text = "데이터 분석과 인공지능 기술이 발전함에 따라... " * 100  # 매우 긴 텍스트
max_limit = 50
model_name = "gpt-4o-mini"

encoding = tiktoken.encoding_for_model(model_name)
tokens = encoding.encode(long_text)

print(f"원본 텍스트 토큰 수: {len(tokens)}")

if len(tokens) > max_limit:
    # 토큰 단위로 자르기
    truncated_tokens = tokens[:max_limit]
    truncated_text = encoding.decode(truncated_tokens)
    print(f"\n{max_limit} 토큰으로 자른 텍스트:\n{truncated_text}...")
    print(f"(실제 비용 청구 기준은 이 잘린 텍스트가 된다.)")
else:
    print("텍스트가 제한보다 짧음.")

