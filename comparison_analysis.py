import pandas as pd
import os
from sklearn.metrics import classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Supervised ve zero-shot sonuçlarının birleştirileceği dosyaların yolları
supervised_files = [
    "sentiment_data/Matrix_500_Reviews_reviews_sentiment.csv",
    "sentiment_data/Nosferatu_2024_500_reviews_sentiment.csv",
    "sentiment_data/Requiem_for_a_Dream_500_Reviews_reviews_sentiment.csv",
    "sentiment_data/Harry_Potter_and_the_Sorcerer's_Stone_500_Reviews_reviews_sentiment.csv",
    "sentiment_data/The_Lord_of_The_Rings_Two_Towers_500_Reviews_reviews_sentiment.csv",
    "sentiment_data/Mad_Max_Fury_Road_500_Reviews_reviews_sentiment.csv"
]

zero_shot_files = [
    "zero_shot_results/Matrix_500_Reviews_reviews_zero_shot_sentiment.csv",
    "zero_shot_results/Nosferatu_2024_500_reviews_zero_shot_sentiment.csv",
    "zero_shot_results/Requiem_for_a_Dream_500_Reviews_reviews_zero_shot_sentiment.csv",
    "zero_shot_results/Harry_Potter_and_the_Sorcerer's_Stone_500_Reviews_reviews_zero_shot_sentiment.csv",
    "zero_shot_results/The_Lord_of_The_Rings_Two_Towers_500_Reviews_reviews_zero_shot_sentiment.csv",
    "zero_shot_results/Mad_Max_Fury_Road_500_Reviews_reviews_zero_shot_sentiment.csv"
]

# Supervised sonuçlarını birleştirme
def combine_supervised(files):
    supervised_dfs = [pd.read_csv(file) for file in files]
    combined_supervised = pd.concat(supervised_dfs, ignore_index=True)
    combined_supervised.to_csv("combined_supervised_results.csv", index=False)
    print("Supervised sonuçları birleştirildi ve 'combined_supervised_results.csv' olarak kaydedildi.")
    return combined_supervised

# Zero-shot sonuçlarını birleştirme
def combine_zero_shot(files):
    zero_shot_dfs = [pd.read_csv(file) for file in files]
    combined_zero_shot = pd.concat(zero_shot_dfs, ignore_index=True)
    combined_zero_shot.to_csv("combined_zero_shot_results.csv", index=False)
    print("Zero-shot sonuçları birleştirildi ve 'combined_zero_shot_results.csv' olarak kaydedildi.")
    return combined_zero_shot

# Supervised ve zero-shot karşılaştırması
def compare_supervised_and_zero_shot(supervised_data, zero_shot_data):
    if not all(supervised_data['Review'] == zero_shot_data['Review']):
        raise ValueError("Supervised ve zero-shot verilerinde uyuşmayan yorumlar var.")

    combined_data = supervised_data.copy()
    combined_data['ZeroShotSentiment'] = zero_shot_data['ZeroShotSentiment']
    combined_data.to_csv("combined_comparison_results.csv", index=False)
    print("Supervised ve zero-shot sonuçları karşılaştırıldı ve 'combined_comparison_results.csv' olarak kaydedildi.")

    # TF-IDF vektörleştirme
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(combined_data['Review'])
    y = combined_data['Sentiment']

    # Eğitim ve test setine ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Supervised model eğitimi ve tahmin
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Supervised Model Performansı (Test Set):")
    print(classification_report(y_test, y_pred))

    # Zero-shot model değerlendirmesi
    print("Zero-Shot Model Performansı:")
    print(classification_report(combined_data['Sentiment'], combined_data['ZeroShotSentiment']))

    return combined_data

# Performans karşılaştırması görselleştirme
def visualize_performance(combined_data):
    metrics = ['precision', 'recall', 'f1-score']
    classes = ['negative', 'neutral', 'positive']

    supervised_metrics = precision_recall_fscore_support(
        combined_data['Sentiment'], combined_data['Sentiment'], average=None
    )

    zero_shot_metrics = precision_recall_fscore_support(
        combined_data['Sentiment'], combined_data['ZeroShotSentiment'], average=None
    )

    for i, metric in enumerate(metrics):
        plt.figure()
        plt.bar(classes, supervised_metrics[i], alpha=0.6, label='Supervised')
        plt.bar(classes, zero_shot_metrics[i], alpha=0.6, label='Zero-Shot')
        plt.title(f'{metric.capitalize()} Comparison')
        plt.xlabel('Classes')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.savefig(f'performance_{metric}_comparison.png')
        plt.close()

    print("Performans karşılaştırmaları görselleştirildi ve kaydedildi.")

# Fonksiyonları çalıştırma
if not os.path.exists("sentiment_data") or not os.path.exists("zero_shot_results"):
    print("Gerekli veri klasörleri bulunamadı. Lütfen önce gerekli analizleri tamamlayın.")
else:
    combined_supervised = combine_supervised(supervised_files)
    combined_zero_shot = combine_zero_shot(zero_shot_files)
    combined_comparison = compare_supervised_and_zero_shot(combined_supervised, combined_zero_shot)
    visualize_performance(combined_comparison)
    print("Tüm işlemler tamamlandı.")
