from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

# Load and preprocess data
df_train = pd.read_csv("C:\\Users\\ASUS-PC\\Desktop\\miniproject\\mini_project6th sem\\updated_train.csv")
df_train.dropna(inplace=True)
X = df_train['text']
y = df_train['sentiment']

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
# Train SVM model
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(X)
svm_model = LinearSVC()
svm_model.fit(X_tfidf, y)

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    text_tfidf = tfidf_vectorizer.transform([text])
    prediction = svm_model.predict(text_tfidf)[0]
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
