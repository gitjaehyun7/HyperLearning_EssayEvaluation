import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os
from data.process_data import process_json_to_csv  # JSON 처리 함수 가져오기
import sys
import os

# 프로젝트 루트 경로를 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from data.process_data import process_json_to_csv  # JSON 처리 함수 가져오기


# JSON 데이터 처리 후 CSV 파일 경로 확인
csv_path = "./data/processed_essay_data.csv"
if not os.path.exists(csv_path):
    print("CSV 파일이 없으므로 JSON 데이터를 처리합니다.")
    process_json_to_csv()  # JSON 데이터를 CSV로 변환
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"JSON 데이터 처리가 완료되었지만 {csv_path} 파일이 생성되지 않았습니다.")
    print(f"CSV 파일 생성 완료: {csv_path}")

# 데이터 로드 및 전처리 함수
def load_and_preprocess_data(csv_path):
    """
    데이터 로드 및 전처리 수행
    """
    df = pd.read_csv(csv_path)

    # 사용 중인 특성 및 타겟 설정
    features = ['word_count', 'unique_word_count', 'sentence_count']
    targets = ['essay_scoreT_org', 'essay_scoreT_cont', 'essay_scoreT_exp']

    X = df[features]
    y = df[targets]

    # 데이터 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

# 데이터 정규화 함수
def normalize_data(X_train, X_val):
    """
    훈련 및 검증 데이터를 정규화하는 함수
    """
    X_train_scaled = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
    X_val_scaled = (X_val - X_train.mean(axis=0)) / X_train.std(axis=0)
    return X_train_scaled, X_val_scaled

# 딥러닝 모델 생성 함수
def create_model(input_dim, dropout_rate=0.3, neurons=[128, 64], patience=10):
    """
    딥러닝 모델 생성 함수
    """
    model = Sequential([
        Dense(neurons[0], activation='relu', input_dim=input_dim),
        Dropout(dropout_rate),
        Dense(neurons[1], activation='relu'),
        Dropout(dropout_rate),
        Dense(3, activation='sigmoid')  # 점수 예측을 위한 3개 출력 노드
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# K-Fold 교차 검증 함수
def perform_kfold_cv(X, y, n_splits=5, epochs=100, batch_size=16):
    """
    K-Fold 교차 검증 수행
    """
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_mae_scores = []

    for i, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f"Processing Fold {i+1}/{n_splits}...")

        # 데이터 분할
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        # 데이터 정규화
        X_train_scaled, X_val_scaled = normalize_data(X_train_fold, X_val_fold)
        y_train_normalized = (y_train_fold - 1) / 2
        y_val_normalized = (y_val_fold - 1) / 2

        # 모델 생성
        model = create_model(input_dim=X_train_scaled.shape[1], dropout_rate=0.3, neurons=[128, 64])

        # EarlyStopping 설정
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # 모델 학습
        model.fit(
            X_train_scaled, y_train_normalized,
            validation_data=(X_val_scaled, y_val_normalized),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=0
        )

        # 모델 평가
        _, mae = model.evaluate(X_val_scaled, y_val_normalized, verbose=0)
        cv_mae_scores.append(mae)
        print(f"Fold {i+1} MAE: {mae:.4f}")

    # 교차 검증 결과 출력
    print(f"\nCross-validation MAE: {np.mean(cv_mae_scores):.4f} ± {np.std(cv_mae_scores):.4f}")

# 메인 실행 코드
if __name__ == "__main__":
    # 데이터 로드
    X, y = load_and_preprocess_data(csv_path)

    # 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 모델 생성 및 학습
    print("Training model with train/test split...")
    model = create_model(input_dim=X_train.shape[1], dropout_rate=0.3, neurons=[128, 64])

    # 출력값을 [1, 3] 범위로 변환
    y_train_normalized = (y_train - 1) / 2
    y_test_normalized = (y_test - 1) / 2

    # EarlyStopping 설정
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # 모델 학습
    history = model.fit(
        X_train, y_train_normalized,
        validation_data=(X_test, y_test_normalized),
        epochs=100,
        batch_size=16,
        callbacks=[early_stopping],
        verbose=1
    )

    # 모델 평가
    loss, mae = model.evaluate(X_test, y_test_normalized)
    print(f"\nTest MAE: {mae:.4f}")

    # K-Fold 교차 검증
    print("\nPerforming K-Fold Cross-Validation...")
    perform_kfold_cv(X, y, n_splits=5, epochs=100, batch_size=16)
