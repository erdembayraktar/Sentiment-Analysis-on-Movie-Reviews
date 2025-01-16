from flask import Flask, request, render_template
import joblib
import os

# Flask uygulaması başlatma
app = Flask(__name__)

# Model ve vektörleştirici yükleme
MODEL_PATH = "model/sentiment_model.pkl"
VECTORIZER_PATH = "model/tfidf_vectorizer.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError("Gerekli model dosyaları bulunamadı. Lütfen ml_model.py dosyasını çalıştırın.")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        if not review:
            return render_template('index.html', prediction="Lütfen bir yorum girin.")

        # Yorumu TF-IDF vektörleştirme
        review_tfidf = vectorizer.transform([review])

        # Tahmin yapma
        prediction = model.predict(review_tfidf)[0]

        return render_template('index.html', prediction=f"Tahmin edilen duygu: {prediction}")

if __name__ == '__main__':
    app.run(debug=True)
