import pandas as pd
import matplotlib.pyplot as plt
import os

# Duygu analizi sonuçlarının yolları
sentiment_files = [
    "sentiment_data/Matrix_500_Reviews_reviews_sentiment.csv",
    "sentiment_data/Nosferatu_2024_500_reviews_sentiment.csv",
    "sentiment_data/Requiem_for_a_Dream_500_Reviews_reviews_sentiment.csv",
    "sentiment_data/Harry_Potter_and_the_Sorcerer's_Stone_500_Reviews_reviews_sentiment.csv",
    "sentiment_data/The_Lord_of_The_Rings_Two_Towers_500_Reviews_reviews_sentiment.csv",
    "sentiment_data/Mad_Max_Fury_Road_500_Reviews_reviews_sentiment.csv"
]

# Her film için duygu dağılımını görselleştirme
def plot_sentiment_distribution(files):
    output_dir = "visualizations"
    os.makedirs(output_dir, exist_ok=True)

    for path in files:
        film_name = os.path.basename(path).replace("_sentiment.csv", "")
        df = pd.read_csv(path)

        sentiment_counts = df['Sentiment'].value_counts()

        plt.figure()
        sentiment_counts.plot(kind='bar', color=['green', 'red', 'gray'])
        plt.title(f"{film_name} Duygu Dağılımı")
        plt.xlabel("Duygu Türü")
        plt.ylabel("Yorum Sayısı")
        plt.xticks(rotation=0)
        plt.savefig(os.path.join(output_dir, f"{film_name}_sentiment_distribution.png"))
        plt.close()

    print(f"Tüm filmler için duygu dağılımı görselleştirmeleri {output_dir} klasörüne kaydedildi.")

# Genel duygu dağılımını görselleştirme
def plot_overall_sentiment_distribution(files):
    combined_df = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)
    sentiment_counts = combined_df['Sentiment'].value_counts()

    plt.figure()
    sentiment_counts.plot(kind='pie', autopct='%1.1f%%', colors=['green', 'red', 'gray'])
    plt.title("Genel Duygu Dağılımı")
    plt.ylabel("")
    plt.savefig("visualizations/overall_sentiment_distribution.png")
    plt.close()

    print("Genel duygu dağılımı görselleştirmesi visualizations klasörüne kaydedildi.")

# Fonksiyonları çalıştırma
if not os.path.exists("sentiment_data"):
    print("Duygu analizi sonuçları bulunamadı. Lütfen önce sentiment_analysis.py dosyasını çalıştırın.")
else:
    plot_sentiment_distribution(sentiment_files)
    plot_overall_sentiment_distribution(sentiment_files)
    print("Tüm görselleştirmeler tamamlandı.")
