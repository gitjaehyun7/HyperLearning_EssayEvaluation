# HyperLearning_EssayEvaluation

## 1. 프로젝트 제목 및 간단한 설명
이 프로젝트는 AI Hub에서 제공된 고등학교 2학년 에세이 데이터를 분석하고, 글의 특성과 점수를 예측하는 딥러닝 모델을 개발하는 것을 목표로 합니다.  
전처리 단계를 거쳐 학습 가능한 데이터로 변환한 후, 딥러닝 모델로 학습하고 성능을 평가하였습니다.

---

## 2. 설치 및 실행 방법

### **1. 환경 설정**
1. Python 3.8 이상 설치  

2. 저장소 다운로드:
   ```bash
   git clone https://github.com/gitjaehyun7/HyperLearning_EssayEvaluation.git
   cd HyperLearning_EssayEvaluation
   ```

3. 필요한 라이브러리 설치:
   ```bash
   pip install -r requirements.txt
   ```

### **2. 실행 방법**

#### **원본 데이터 준비**
1. AI Hub에서 다운로드한 원본(검증) 데이터(`글짓기`, `대안제시`, `설명글`, `주장`, `찬성반대` 폴더)를 준비합니다.
2. 생성된 프로젝트 디렉토리 안에 `data/essays` 폴더를 생성한 후, 해당 폴더 안에 원본 데이터를 복사합니다:
    ```
    HyperLearning_EssayEvaluation/
    ├── data/
    │   └── essays/
    │       ├── 글짓기/
    │       ├── 대안제시/
    │       ├── 설명글/
    │       ├── 주장/
    │       └── 찬성반대/
    ```

#### **(1) 데이터 전처리**
`/data/process_data.py` 스크립트를 실행하여 JSON 데이터를 전처리하고 CSV 파일로 변환합니다:
```bash
python -m data.process_data
```

#### **(2) 데이터 모델링 및 평가**
`/src/train_model.py` 스크립트를 실행하여 전처리된 CSV 파일을 사용해 딥러닝 모델을 학습시키고 성능을 평가합니다:
```bash
python -m src.train_model
```
- 모델 학습: 학습 데이터를 바탕으로 모델을 학습시키고 교차 검증을 수행합니다.
- 평가 지표: MAE(Mean Absolute Error) 값을 출력합니다.
- 결과: 평가 결과는 터미널에 출력됩니다.

---

## 3. 파일 구조 설명

```
HyperLearning_EssayEvaluation/
├── /src/                # 소스 코드 파일
│   ├── train_model.py   # 모델 학습 및 평가 코드
├── /data/               # 데이터 처리 코드
│   ├── process_data.py  # JSON 데이터를 CSV로 변환 및 전처리
├── /docs/               # 프로젝트 문서화 파일
│   ├── 결과 보고서 (PDF 형식)
├── README.md            # 프로젝트 개요 및 실행 방법
├── requirements.txt     # 사용된 라이브러리 목록
```

---

## 4. 사용한 주요 기술 및 라이브러리
- **Python 라이브러리**:
  - `numpy`, `pandas`: 데이터 분석 및 전처리
  - `scikit-learn`:
     - 데이터 분리 (train_test_split, kFold) 
     - 데이터 스케일링 (StandardScaler)
     - 텍스트 벡터화 (TfidVectorizer, CountVertorizer)
  - `tensorflow`,`keras` : 딥러닝 모델 구축 및 학습
  - `os`, `sys` : 파일 및 시스템 경로 관리
  - `data.process_data` : JSON 데이터를 CSV로 변환하는 사용자 정의 모듈
- **딥러닝 모델 구조**:
  ***Word Count 기반 모델***
  - 은닉 레이어: [128, 64] 노드, ReLU 활성화 함수
  - Dropout: 30% 비율
  - 출력 레이어: 3개 노드(sigmoid 활성화)
  ***Word Count 기반 모델***
  - 은닉 레이어: [128, 64] 노드, ReLU 활성화 함수
  - Dropout: 20% 비율
  - 출력 레이어: 3개 노드(linear 활성화)
- **평가 지표**:
  - MAE(Mean Absolute Error)를 기반으로 성능 평가
  - K-Fold 교차 검증 : 성능의 일반화 평가

---
