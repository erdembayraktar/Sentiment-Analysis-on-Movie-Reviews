import pandas as pd
from textblob import TextBlob
import os

# Temizlenmiş verilerin yolları
cleaned_files = [
    "cleaned_data/Matrix_500_Reviews_reviews_cleaned.csv",
    "cleaned_data/Nosferatu_2024_500_reviews_cleaned.csv",
    "cleaned_data/Requiem_for_a_Dream_500_Reviews_reviews_cleaned.csv",
    "cleaned_data/Harry_Potter_and_the_Sorcerer's_Stone_500_Reviews_reviews_cleaned.csv",
    "cleaned_data/The_Lord_of_The_Rings_Two_Towers_500_Reviews_reviews_cleaned.csv",
    "cleaned_data/Mad_Max_Fury_Road_500_Reviews_reviews_cleaned.csv"
]

# Duygu analiz fonksiyonu
def analyze_sentiment(text):
    """TextBlob kullanarak duygu analizi yapan fonksiyon."""
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return "positive"
    elif polarity < 0:
        return "negative"
    else:
        return "neutral"

# Duygu analizi uygulama ve sonuçları kaydetme
def perform_sentiment_analysis(files):
    output_dir = "sentiment_data"
    os.makedirs(output_dir, exist_ok=True)

    for path in files:
        film_name = os.path.basename(path).replace("_cleaned.csv", "")
        df = pd.read_csv(path)

        # Duygu analizi uygulama
        df['Sentiment'] = df['Review'].apply(analyze_sentiment)

        # Sonuçları kaydetme
        output_path = os.path.join(output_dir, f"{film_name}_sentiment.csv")
        df.to_csv(output_path, index=False)
        print(f"{film_name} için duygu analizi tamamlandı ve kaydedildi: {output_path}")

# Fonksiyonları çalıştırma
if not os.path.exists("cleaned_data"):
    print("Temizlenmiş veriler klasörü bulunamadı. Lütfen önce data_cleaning.py dosyasını çalıştırın.")
else:
    perform_sentiment_analysis(cleaned_files)
    print("Tüm filmler için duygu analizi tamamlandı ve sonuçlar kaydedildi.")
