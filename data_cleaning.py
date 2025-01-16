import pandas as pd
import os

# CSV dosyalarının yolları
files = [
    "data/Matrix_500_Reviews_reviews.csv",
    "data/Nosferatu_2024_500_reviews.csv",
    "data/Requiem_for_a_Dream_500_Reviews_reviews.csv",
    "data/Harry_Potter_and_the_Sorcerer's_Stone_500_Reviews_reviews.csv",
    "data/The_Lord_of_The_Rings_Two_Towers_500_Reviews_reviews.csv",
    "data/Mad_Max_Fury_Road_500_Reviews_reviews.csv"
]

# Temizlenmiş veriler için klasör oluşturma
output_dir = "cleaned_data"
os.makedirs(output_dir, exist_ok=True)

def clean_text(text):
    """Yorumları temizleyen bir fonksiyon."""
    text = text.lower()  # Küçük harfe dönüştür
    text = text.replace("\n", " ")  # Satır sonlarını kaldır
    text = text.replace("[^\w\s]", "")  # Noktalama işaretlerini kaldır
    return text

def clean_dataframe(df):
    """Veri çerçevesini temizleyen bir fonksiyon."""
    df = df.drop_duplicates()  # Tekrar eden yorumları kaldır
    df = df.dropna()  # Eksik değerleri kaldır
    df["Review"] = df["Review"].apply(clean_text)  # Yorumları temizle
    return df

# Her bir dosyayı temizleyip yeni bir CSV olarak kaydetme
for path in files:
    film_name = os.path.basename(path).replace(".csv", "")
    df = pd.read_csv(path)
    cleaned_df = clean_dataframe(df)
    output_path = os.path.join(output_dir, f"{film_name}_cleaned.csv")
    cleaned_df.to_csv(output_path, index=False)
    print(f"{film_name} verileri temizlendi ve kaydedildi: {output_path}")

print("Tüm dosyalar başarıyla temizlendi ve kaydedildi.")
