import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os
from data.process_data import process_json_to_csv
import sys

# 프로젝트 루트 경로를 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# JSON 데이터 처리 후 CSV 파일 경로 확인
csv_path = "./data/processed_essay_data.csv"
if not os.path.exists(csv_path):
    print("CSV 파일이 없으므로 JSON 데이터를 처리합니다.")
    process_json_to_csv()
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"JSON 데이터 처리가 완료되었지만 {csv_path} 파일이 생성되지 않았습니다.")
    print(f"CSV 파일 생성 완료: {csv_path}")

# 데이터 로드 및 전처리 함수
def load_and_preprocess_data(csv_path):
    df = pd.read_csv(csv_path)
    return df

# Word Count 기반 데이터 로드 함수
def prepare_word_count_data(df):
    features = ['word_count', 'unique_word_count', 'sentence_count']
    targets = ['essay_scoreT_org', 'essay_scoreT_cont', 'essay_scoreT_exp']

    X = df[features]
    y = df[targets]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

# 텍스트 벡터화 기반 데이터 로드 함수
def prepare_text_vectorized_data(df, max_features=5000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(df['cleaned_paragraph']).toarray()
    y = df[['essay_scoreT_org', 'essay_scoreT_cont', 'essay_scoreT_exp']]
    return X, y

# 딥러닝 모델 생성 함수
def create_model(input_dim, dropout_rate=0.3, neurons=[128, 64]):
    model = Sequential([
        Dense(neurons[0], activation='relu', input_dim=input_dim),
        Dropout(dropout_rate),
        Dense(neurons[1], activation='relu'),
        Dropout(dropout_rate),
        Dense(3, activation='linear')  # 선형 활성화 함수
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# K-Fold 교차 검증 함수
def perform_kfold_cv(X, y, input_dim, n_splits=5, epochs=50, batch_size=32):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_mae_scores = []

    for i, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f"Processing Fold {i+1}/{n_splits}...")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = create_model(input_dim=input_dim, dropout_rate=0.3, neurons=[128, 64])
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=0
        )

        _, mae = model.evaluate(X_val, y_val, verbose=0)
        cv_mae_scores.append(mae)
        print(f"Fold {i+1} MAE: {mae:.4f}")

    print(f"\nCross-validation MAE: {np.mean(cv_mae_scores):.4f} ± {np.std(cv_mae_scores):.4f}")

# 메인 실행 코드
if __name__ == "__main__":
    df = load_and_preprocess_data(csv_path)

    # Word Count 기반 데이터 처리 및 모델 학습
    print("Training model with Word Count features...")
    X_wc, y_wc = prepare_word_count_data(df)
    X_train_wc, X_test_wc, y_train_wc, y_test_wc = train_test_split(X_wc, y_wc, test_size=0.2, random_state=42)
    model_wc = create_model(input_dim=X_train_wc.shape[1])
    early_stopping_wc = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_wc.fit(X_train_wc, y_train_wc, validation_data=(X_test_wc, y_test_wc), epochs=50, batch_size=32, callbacks=[early_stopping_wc], verbose=1)
    loss, mae = model_wc.evaluate(X_test_wc, y_test_wc)
    print(f"Word Count Model Test MAE: {mae:.4f}")

    # 텍스트 벡터화 기반 데이터 처리 및 모델 학습
    print("\nTraining model with Text Vectorized features...")
    X_tv, y_tv = prepare_text_vectorized_data(df, max_features=5000)
    X_train_tv, X_test_tv, y_train_tv, y_test_tv = train_test_split(X_tv, y_tv, test_size=0.2, random_state=42)
    model_tv = create_model(input_dim=X_train_tv.shape[1])
    early_stopping_tv = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_tv.fit(X_train_tv, y_train_tv, validation_data=(X_test_tv, y_test_tv), epochs=50, batch_size=32, callbacks=[early_stopping_tv], verbose=1)
    loss, mae = model_tv.evaluate(X_test_tv, y_test_tv)
    print(f"Text Vectorized Model Test MAE: {mae:.4f}")

    # Word Count 기반 K-Fold 교차 검증
    print("\nPerforming K-Fold Cross-Validation for Word Count features...")
    perform_kfold_cv(X_wc, y_wc, input_dim=X_wc.shape[1])

    # 텍스트 벡터화 기반 K-Fold 교차 검증
    print("\nPerforming K-Fold Cross-Validation for Text Vectorized features...")
    perform_kfold_cv(X_tv, y_tv, input_dim=X_tv.shape[1])
