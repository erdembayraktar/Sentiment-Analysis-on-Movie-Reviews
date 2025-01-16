from transformers import pipeline
import pandas as pd
import os
from tqdm import tqdm
import torch
import time

# GPU kontrolü
device = 0 if torch.cuda.is_available() else -1
if device == 0:
    print("GPU kullanılacak.")
else:
    print("CPU kullanılacak.")

# Temizlenmiş verilerin yolları
cleaned_files = [
    "cleaned_data/Matrix_500_Reviews_reviews_cleaned.csv",
    "cleaned_data/Nosferatu_2024_500_reviews_cleaned.csv",
    "cleaned_data/Requiem_for_a_Dream_500_Reviews_reviews_cleaned.csv",
    "cleaned_data/Harry_Potter_and_the_Sorcerer's_Stone_500_Reviews_reviews_cleaned.csv",
    "cleaned_data/The_Lord_of_The_Rings_Two_Towers_500_Reviews_reviews_cleaned.csv",
    "cleaned_data/Mad_Max_Fury_Road_500_Reviews_reviews_cleaned.csv"
]

# Zero-shot sınıflandırıcıyı yükleme
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)

# Zero-shot duygu analizi fonksiyonu
def perform_zero_shot_sentiment_analysis(files):
    output_dir = "zero_shot_results"
    os.makedirs(output_dir, exist_ok=True)

    candidate_labels = ["positive", "negative", "neutral"]

    for path in files:
        film_name = os.path.basename(path).replace("_cleaned.csv", "")
        df = pd.read_csv(path)

        sentiments = []
        start_time = time.time()

        for i, review in enumerate(tqdm(df['Review'], desc=f"Processing {film_name}"), 1):
            result = classifier(review, candidate_labels)
            sentiments.append(result['labels'][0])

        df['ZeroShotSentiment'] = sentiments

        output_path = os.path.join(output_dir, f"{film_name}_zero_shot_sentiment.csv")
        df.to_csv(output_path, index=False)
        print(f"\n{film_name} için zero-shot duygu analizi tamamlandı: {output_path}")

# Fonksiyonları çalıştırma
if not os.path.exists("cleaned_data"):
    print("Temizlenmiş veriler bulunamadı. Lütfen önce veri temizleme adımlarını tamamlayın.")
else:
    perform_zero_shot_sentiment_analysis(cleaned_files)
    print("Zero-shot duygu analizi tamamlandı.")
