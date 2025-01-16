import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
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

# Verileri birleştirme
def load_and_combine_data(files):
    dataframes = [pd.read_csv(file) for file in files]
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df

# Makine öğrenimi modeli eğitme
def train_ml_model(data):
    # Giriş ve çıkış değişkenlerini ayırma
    X = data['Review']
    y = data['Sentiment']

    # Eğitim ve test setine ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # TF-IDF ile metinleri sayısallaştırma
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Random Forest modeli eğitimi
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_tfidf, y_train)

    # Modeli test etme
    y_pred = model.predict(X_test_tfidf)
    print("Model Performansı:")
    print(classification_report(y_test, y_pred))

    return model, vectorizer

# Ana işlem
if not os.path.exists("sentiment_data"):
    print("Duygu analizi sonuçları bulunamadı. Lütfen önce sentiment_analysis.py dosyasını çalıştırın.")
else:
    combined_data = load_and_combine_data(sentiment_files)
    print("Veriler başarıyla birleştirildi.")

    trained_model, trained_vectorizer = train_ml_model(combined_data)
    print("Makine öğrenimi modeli eğitildi.")

    # Model ve vektörleştiriciyi kaydetme
    import joblib

    os.makedirs("model", exist_ok=True)
    joblib.dump(trained_model, "model/sentiment_model.pkl")
    joblib.dump(trained_vectorizer, "model/tfidf_vectorizer.pkl")
    print("Model ve vektörleştirici kaydedildi.")
