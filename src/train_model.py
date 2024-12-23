import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os
from data.process_data import process_json_to_csv  # JSON 처리 함수 가져오기
import sys
from sklearn.feature_extraction.text import CountVectorizer

# 프로젝트 루트 경로를 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# JSON 데이터 처리 후 CSV 파일 경로 확인
csv_path = "./data/processed_essay_data.csv"
if not os.path.exists(csv_path):
    print("CSV 파일이 없으므로 JSON 데이터를 처리합니다.")
    process_json_to_csv()  # JSON 데이터를 CSV로 변환
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"JSON 데이터 처리가 완료되었지만 {csv_path} 파일이 생성되지 않았습니다.")
    print(f"CSV 파일 생성 완료: {csv_path}")

# 데이터 로드 함수

def load_and_preprocess_data_word_count(csv_path):
    """Word Count 기반 데이터 로드 및 전처리"""
    df = pd.read_csv(csv_path)
    features = ['word_count', 'unique_word_count', 'sentence_count']
    targets = ['essay_scoreT_org', 'essay_scoreT_cont', 'essay_scoreT_exp']
    X = df[features]
    y = df[targets]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def load_and_preprocess_data_vectorized(csv_path):
    """텍스트 벡터화 기반 데이터 로드 및 전처리"""
    df = pd.read_csv(csv_path)
    vectorizer = CountVectorizer()
    X_counts = vectorizer.fit_transform(df['cleaned_paragraph'])  # 텍스트를 벡터화
    unique_word_count = len(vectorizer.get_feature_names_out())  # 고유 단어 수
    vectorizer = TfidfVectorizer(max_features=unique_word_count)
    X = vectorizer.fit_transform(df['cleaned_paragraph']).toarray()
    y = df[['essay_scoreT_org', 'essay_scoreT_cont', 'essay_scoreT_exp']]
    return X, y

# 딥러닝 모델 생성 함수

def create_model_word_count(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(3, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def create_model_vectorized(input_dim):
    model = Sequential([
        Dense(256, activation='relu', input_dim=input_dim),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(3, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# K-Fold 교차 검증 함수

def perform_kfold_cv(X, y, create_model_func, n_splits=5, epochs=100, batch_size=16):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_mae_scores = []
    for i, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f"Processing Fold {i+1}/{n_splits}...")
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        y_train_normalized = (y_train_fold - 1) / 2
        y_val_normalized = (y_val_fold - 1) / 2
        model = create_model_func(input_dim=X_train_fold.shape[1])
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(
            X_train_fold, y_train_normalized,
            validation_data=(X_val_fold, y_val_normalized),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=0
        )
        _, mae = model.evaluate(X_val_fold, y_val_normalized, verbose=0)
        cv_mae_scores.append(mae)
        print(f"Fold {i+1} MAE: {mae:.4f}")
    print(f"\nCross-validation MAE: {np.mean(cv_mae_scores):.4f} ± {np.std(cv_mae_scores):.4f}")

# 메인 실행 코드

if __name__ == "__main__":
    print("Word Count 기반 데이터 처리 및 모델 학습")
    X_word_count, y_word_count = load_and_preprocess_data_word_count(csv_path)
    perform_kfold_cv(X_word_count, y_word_count, create_model_word_count, epochs=100, batch_size=16)

    print("\n텍스트 벡터화 기반 데이터 처리 및 모델 학습")
    X_vectorized, y_vectorized = load_and_preprocess_data_vectorized(csv_path)
    perform_kfold_cv(X_vectorized, y_vectorized, create_model_vectorized, epochs=50, batch_size=32)
