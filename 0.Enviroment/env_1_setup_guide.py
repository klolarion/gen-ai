# ============================================================================
# Python 환경 설정 가이드 (venv vs Conda vs Poetry vs Pyenv)
# ============================================================================
# 프로젝트 목적에 맞는 가상환경 도구를 선택하여 사용하세요.
# 
# 1. venv (Python 기본 내장)
#    - 장점: 별도 설치 불필요, 가볍고 빠름, 표준 라이브러리
#    - 단점: Python 버전 자체를 관리해주진 않음
#    - 추천: 일반적인 소규모 프로젝트, 학습용
#    [명령어]
#    $ python -m venv .venv          # 생성
#    $ source .venv/bin/activate     # 활성화 (Mac/Linux)
#    $ .venv\Scripts\activate        # 활성화 (Windows)
#    $ pip install -r requirements.txt
#
# ----------------------------------------------------------------------------
# 2. Conda (Anaconda / Miniconda)
#    - 장점: Python 버전 관리 가능, 비-Python 라이브러리(C++ 등) 설치 용이
#    - 단점: 무거움, 상용 라이선스 이슈(Anaconda)
#    - 추천: 데이터 사이언스, AI/ML 프로젝트 (GPU 설정 등)
#    [명령어]
#    $ conda create -n myenv python=3.11  # 생성 (버전 지정 가능)
#    $ conda activate myenv               # 활성화
#    $ conda install numpy pandas         # 패키지 설치
#
# ----------------------------------------------------------------------------
# 3. Poetry (Modern Dependency Manager)
#    - 장점: 의존성 충돌 해결(Lock 파일), 패키지 관리 + 가상환경 통합
#    - 단점: 학습 곡선 있음, 일부 비표준 패키지 설치 까다로움
#    - 추천: 실무 프로젝트, 패키지 배포, 협업 시 버전 고정 필요할 때
#    [명령어]
#    $ poetry init                        # 설정 파일(pyproject.toml) 생성
#    $ poetry add langchain openai        # 패키지 추가
#    $ poetry shell                       # 가상환경 활성화
#
# ----------------------------------------------------------------------------
# 4. Pyenv (Python Version Manager)
#    - 역할: 여러 버전의 Python을 로컬에 설치하고 스위칭 (가상환경 도구는 아님)
#    - 추천: 프로젝트마다 다른 Python 버전(3.8, 3.11 등)을 써야 할 때
#    [명령어]
#    $ pyenv install 3.11.7               # 특정 버전 설치
#    $ pyenv local 3.11.7                 # 현재 폴더에서만 이 버전 사용
# ============================================================================

import sys
import os
import pkg_resources
from dotenv import load_dotenv

def check_current_env():
    """현재 실행 환경을 점검합니다."""
    print("\n" + "="*60)
    print("🛠️  현재 Python 실행 환경 점검")
    print("="*60)

    # 1. Python 버전
    print(f"✅ Python Version: {sys.version.split()[0]}")

    # 2. 가상환경 여부
    # sys.prefix와 sys.base_prefix가 다르면 가상환경 내부임
    is_venv = sys.prefix != sys.base_prefix
    # Conda 환경 확인
    is_conda = 'CONDA_DEFAULT_ENV' in os.environ

    if is_conda:
        print(f"✅ Environment Type: Conda ({os.environ['CONDA_DEFAULT_ENV']})")
    elif is_venv:
        print(f"✅ Environment Type: Virtual Environment (venv/virtualenv)")
    else:
        print(f"⚠️  Environment Type: Global System Python (권장하지 않음)")
    
    print(f"📂 Path: {sys.prefix}")

    # 3. .env 파일 점검
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print(f"✅ API Key Status: Found (.env 로드 성공)")
    else:
        print(f"⚠️  API Key Status: Not Found (.env 파일 확인 필요)")

    # 4. 필수 패키지 점검
    print("-" * 60)
    required = ['langchain', 'openai', 'tiktoken']
    installed = {pkg.key for pkg in pkg_resources.working_set}
    
    missing = [pkg for pkg in required if pkg not in installed]
    
    if not missing:
        print("🎉 필수 패키지(LangChain, OpenAI)가 설치되어 있습니다.")
    else:
        print(f"❌ 설치되지 않은 패키지: {', '.join(missing)}")
        print("   -> pip install -r requirements.txt (또는 poetry install)")

    print("="*60 + "\n")

if __name__ == "__main__":
    check_current_env()

