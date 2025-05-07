from flask import Flask, render_template, request, redirect, flash
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import docx
import fitz
import re
import secrets
from chardet import detect

# Простейший препроцессор текста
def clean_text(text):
    text = re.sub(r'\W+', ' ', text.lower())  # Удаляем всё кроме букв и цифр
    return text.strip()

# Автоматическое определение кодировки текста
def decode_text(file_content):
    detection_result = detect(file_content)
    detected_encoding = detection_result['encoding'] or 'utf-8'
    decoded_text = file_content.decode(detected_encoding)
    return decoded_text

# Инициализация Flask-приложения
app = Flask(__name__)
app.secret_key = 'secrets.token_hex(16)'

# Загружаем предварительно обученную модель
model = load('best_svm_model.pkl')
vectorizer = load('best_tfidf_svm_vectorizer.pkl')

label_encoder = load('label_encoder.joblib')

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_genre():
    input_text = ''
    uploaded_file = request.files.get('file_input')
    
    # Если выбран файл
    if uploaded_file:
        try:
            if uploaded_file.filename.endswith('.txt'):
                content = uploaded_file.read()
                input_text = decode_text(content)
            
            elif uploaded_file.filename.endswith('.docx'):
                doc = docx.Document(uploaded_file)
                input_text = '\n'.join([para.text for para in doc.paragraphs])
                
            elif uploaded_file.filename.endswith('.pdf'):
                pdf_document = fitz.open(stream=uploaded_file.stream.read(), filetype="pdf")
                input_text = ''.join(page.get_text("text") for page in pdf_document.pages())
            
            else:
                raise ValueError("Неподдерживаемый формат файла.")
        except Exception as e:
            flash(str(e))
            return redirect('/')
    
    # Иначе используем введённый текст
    else:
        input_text = request.form['input_text']
    
    # Подготовка текста перед отправкой в векторизатор
    cleaned_text = clean_text(input_text)
    
    # Преобразовываем текст в TF-IDF-векторы
    input_vectorized = vectorizer.transform([cleaned_text])
    
    # Предсказываем класс (жанр)
    predicted_class_num = model.predict(input_vectorized)[0]
    
    # Получаем имя класса обратно
    predicted_genre = label_encoder.inverse_transform([predicted_class_num])[0]
    
    return render_template('index.html', result=predicted_genre)

if __name__ == '__main__':
    app.run(debug=True)
