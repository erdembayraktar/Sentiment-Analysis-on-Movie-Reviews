# Sentiment Analysis Project

## **Overview**
This project performs sentiment analysis on movie reviews using two distinct approaches:
1. **Supervised Sentiment Analysis**: A machine learning model trained on labeled data.
2. **Zero-Shot Sentiment Analysis**: A pre-trained model that classifies sentiments without specific training.

The project also includes a Flask-based web application where users can input their own reviews to get sentiment predictions.

---

## **Features**
- Preprocessing and cleaning of raw text data.
- Exploratory Data Analysis (EDA) with visualizations (histograms, word clouds).
- Supervised model using TF-IDF and Random Forest Classifier.
- Zero-shot sentiment analysis with Hugging Face's `facebook/bart-large-mnli` model.
- Flask web application for real-time predictions.

---

## **Installation**
### **Step 1: Clone the Repository**
```bash
git clone https://github.com/username/repository-name.git
cd repository-name
```

### **Step 2: Set Up Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### **Step 3: Install Requirements**
```bash
pip install -r requirements.txt
```

---

## **Usage**
### **1. Run the Flask Web Application**
```bash
python app.py
```
- Access the application in your browser at `http://127.0.0.1:5000`.
- Input a movie review and receive a sentiment prediction.

### **2. Run Scripts for Analysis**
- **Data Cleaning**:
  ```bash
  python data_cleaning.py
  ```
- **Supervised Sentiment Analysis**:
  ```bash
  python ml_model.py
  ```
- **Zero-Shot Sentiment Analysis**:
  ```bash
  python zero_shot_sentiment.py
  ```
- **Comparison and Evaluation**:
  ```bash
  python comparison_analysis.py
  ```

---

## **Project Structure**
```
project-folder/
├── app.py                  # Flask application
├── data_cleaning.py        # Data preprocessing script
├── ml_model.py             # Supervised sentiment analysis script
├── zero_shot_sentiment.py  # Zero-shot sentiment analysis script
├── comparison_analysis.py  # Performance comparison script
├── requirements.txt        # Python dependencies
├── templates/              # HTML templates for Flask app
│   └── index.html          # Main interface
├── static/                 # Static files (e.g., CSS, JS)
├── cleaned_data/           # Cleaned datasets
├── sentiment_data/         # Supervised model results
├── zero_shot_results/      # Zero-shot model results
└── combined_results/       # Combined and comparison outputs
```

---

## **Results**
### **Supervised Model**
- **Accuracy**: 80%
- Strong performance on positive and neutral sentiments.

### **Zero-Shot Model**
- **Accuracy**: 72%
- Better recall for negative sentiments but struggles with neutral.

---

## **Future Improvements**
- Experiment with advanced models like GPT-4 for better performance.
- Enhance Flask app with zero-shot predictions.
- Implement visualization of real-time predictions.

---

## **License**
This project is licensed under the MIT License. See the LICENSE file for details.

---

## **Acknowledgments**
- [Hugging Face Transformers](https://huggingface.co/docs)
- [Scikit-learn](https://scikit-learn.org/)
- [Flask Framework](https://flask.palletsprojects.com/)
