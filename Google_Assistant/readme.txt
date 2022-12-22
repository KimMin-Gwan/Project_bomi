구글 어시스턴트 API 인증 by JH KIM
1. 구글 어시스턴트 라이브러리 설치 및 샘플 설치
	python -m pip install --upgrade google-assistant-library
	python -m pip install --upgrade google-assistant-sdk[samples]
2. 구글 인증 툴 설치
	python -m pip install --upgrade google-auth-oauthlib[tool]
3. credentials.json 인증파일을 통한 인증
	google-oauthlib-tool --scope https://www.googleapis.com/auth/assistant-sdk-prototype --save --headless --client-secrets JSON인증파일 경로
