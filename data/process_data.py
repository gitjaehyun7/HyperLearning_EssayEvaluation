import pandas as pd
import numpy as np
import os
import json
import unicodedata
import re
from transformers import AutoTokenizer
import ast
# Kobert 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)

# 1. JSON 데이터를 CSV로 변환
def process_json_to_csv(input_dir, output_file):
    folders = ["글짓기", "대안제시", "설명글", "주장", "찬성반대"]
    data_list = []

    for folder in folders:
        folder_path = os.path.join(input_dir, folder)
        for file_name in os.listdir(folder_path):
            normalized_name = unicodedata.normalize('NFC', file_name)
            if "고등_2학년" in normalized_name and normalized_name.endswith(".json"):
                file_path = os.path.join(folder_path, file_name)
                with open(file_path, encoding="utf-8") as f:
                    data = json.load(f)
                    data_list.append(data)

    if data_list:
        df = pd.json_normalize(data_list, sep='_')
        df.to_csv(output_file, index=False, encoding="utf-8")
        print(f"데이터가 CSV로 저장되었습니다: {output_file}")
    else:
        print("조건에 맞는 파일이 없습니다.")

# 2. 텍스트 정리 함수 (특수기호, 줄바꿈 제거)
def clean_simple(text):
    if isinstance(text, str):
        text = re.sub(r'#@문장구분#', ' ', text)  # 문장 구분자 제거
        text = re.sub(r'\n', ' ', text)  # 줄바꿈 제거
        text = re.sub(r'\t', ' ', text)  # 탭 제거
        return text.strip()
    return ''

# 3. 단어 수, 문장 수, 고유 단어 수 계산 함수
def calculate_word_count(text):
    if isinstance(text, str):
        return len(text.split())
    return 0

def calculate_sentence_count(text):
    if isinstance(text, str):
        return len(re.split(r'[.!?]', text))
    return 0

def calculate_unique_word_count(text):
    if isinstance(text, str):
        words = text.split()
        unique_words = set(words)
        return len(unique_words)
    return 0

# 4. 정규화 함수
def normalize_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z가-힣0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
    return text

# 5. 토큰화 함수
def tokenize_text(text):
    return tokenizer.tokenize(text)

# 6. 데이터 전처리 수행
def preprocess_data(input_file, output_file):
    df = pd.read_csv(input_file)

    # 'paragraph' 컬럼에서 텍스트 추출
    def extract_paragraph_text(paragraph_list):
        if isinstance(paragraph_list, str):
            paragraph_list = eval(paragraph_list)  # 문자열을 리스트로 변환
        if isinstance(paragraph_list, list):
            return ' '.join([item['paragraph_txt'] for item in paragraph_list if 'paragraph_txt' in item])
        return ''

    # 텍스트 정리 및 전처리
    df['clean_paragraph'] = df['paragraph'].apply(extract_paragraph_text)
    df['cleaned_paragraph'] = df['clean_paragraph'].apply(clean_simple)

    # 추가 특성 생성
    df['word_count'] = df['cleaned_paragraph'].apply(calculate_word_count)
    df['sentence_count'] = df['cleaned_paragraph'].apply(calculate_sentence_count)
    df['unique_word_count'] = df['cleaned_paragraph'].apply(calculate_unique_word_count)

    # 정규화 및 토큰화
    df['normalized_paragraph'] = df['cleaned_paragraph'].apply(normalize_text)
    df['tokenized_paragraph'] = df['normalized_paragraph'].apply(tokenize_text)

    # 추가 특성 계산
    df['avg_word_length'] = df['tokenized_paragraph'].apply(
        lambda x: np.mean([len(word) for word in x]) if len(x) > 0 else 0
    )
    df['words_per_sentence'] = df.apply(
        lambda row: row['word_count'] / row['sentence_count'] if row['sentence_count'] > 0 else 0, axis=1
    )

    # sentence_count가 0인 경우 avg_word_length와 words_per_sentence를 0으로 대체(Infinity 방지)
    df.loc[df['sentence_count'] == 0, ['avg_word_length', 'words_per_sentence']] = 0

    # 리스트를 평균으로 변환하는 함수
    def calculate_mean(x):
        if isinstance(x, str):  # 문자열이면 eval로 리스트로 변환
            x = ast.literal_eval(x)  # 문자열을 리스트로 변환
        if isinstance(x, list):  # 리스트인지 확인
            return np.mean([np.mean(sublist) for sublist in x if isinstance(sublist, list)])
        return np.nan

    # # 데이터프레임 불러오기
    # df = pd.read_csv("path_to_your_csv_file.csv")

    # 리스트 데이터 변환
    df['essay_scoreT_org'] = df['score_essay_scoreT_detail_essay_scoreT_org'].apply(calculate_mean)
    df['essay_scoreT_cont'] = df['score_essay_scoreT_detail_essay_scoreT_cont'].apply(calculate_mean)
    df['essay_scoreT_exp'] = df['score_essay_scoreT_detail_essay_scoreT_exp'].apply(calculate_mean)

    # 전처리된 데이터 저장
    df.to_csv(output_file, index=False)
    print(f"Preprocessed data saved to {output_file}")

# 실행
if __name__ == "__main__":
    # 1. JSON 데이터를 CSV로 변환
    process_json_to_csv(
        input_dir="./data/essays", 
        output_file="./data/essays_collected.csv"
    )

    # 2. 변환된 CSV 데이터를 전처리
    preprocess_data(
        input_file="./data/essays_collected.csv", 
        output_file="./data/processed_essay_data.csv"
    )


