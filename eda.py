import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
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


# Yorum uzunluklarını analiz etme ve görselleştirme
def plot_review_lengths(files):
    for path in files:
        film_name = os.path.basename(path).replace("_cleaned.csv", "")
        df = pd.read_csv(path)
        df['review_length'] = df['Review'].apply(len)

        plt.figure()
        df['review_length'].hist(bins=30, color='blue', edgecolor='black')
        plt.title(f"{film_name} Yorum Uzunluk Dağılımı")
        plt.xlabel("Yorum Uzunluğu")
        plt.ylabel("Frekans")
        plt.savefig(f"cleaned_data/{film_name}_review_length_distribution.png")
        plt.close()


# Kelime bulutları oluşturma
def create_wordclouds(files):
    for path in files:
        film_name = os.path.basename(path).replace("_cleaned.csv", "")
        df = pd.read_csv(path)
        text = " ".join(df['Review'].tolist())

        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

        plt.figure()
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"{film_name} Kelime Bulutu")
        plt.savefig(f"cleaned_data/{film_name}_wordcloud.png")
        plt.close()


# Fonksiyonları çalıştırma
if not os.path.exists("cleaned_data"):
    print("Temizlenmiş veriler klasörü bulunamadı. Lütfen önce data_cleaning.py dosyasını çalıştırın.")
else:
    plot_review_lengths(cleaned_files)
    create_wordclouds(cleaned_files)
    print("EDA işlemleri tamamlandı ve görseller kaydedildi.")
