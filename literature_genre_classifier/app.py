from flask import Flask, render_template, request, jsonify
from joblib import load
import PyPDF2
import docx
import re
import chardet  # Импортируем модуль для автоматической проверки кодировки

# Загружаем ранее обученную модель и векторизатор
vectorizer = load('models/best_tfidf_svm_vectorizer.pkl')
ffbp_model = load('models/best_svm_model.pkl')

# Создание экземпляра Flask
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_type = request.form.get('input_type')  # Определяем метод передачи данных (текст или файл)
        
        if input_type == 'file':
            uploaded_file = request.files['file']
            content = extract_text(uploaded_file)
        elif input_type == 'text':
            content = request.form.get('text_input')
        else:
            return jsonify({'error': 'Неверный тип ввода'})

        # Преобразовываем текст в вектор признаков
        feature_vector = vectorizer.transform([content])
        
        # Прогоняем предсказание
        prediction = ffbp_model.predict(feature_vector)[0]
        
        genres = ['Роман', 'Повесть', 'Рассказ', 'Поэма', 'Пьеса', 'Статья', 'Очерк']
        result = f'Жанр произведения: {genres[prediction]}'
    
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)})

#Функция для извлечения текста из различных форматов
def extract_text(file):
    filename = file.filename.lower()
    raw_bytes = file.read()
    detected_encoding = chardet.detect(raw_bytes)['encoding'] or 'utf-8'  # Автоматически определяем кодировку

    if filename.endswith('.txt'):
        return raw_bytes.decode(detected_encoding)
    elif filename.endswith('.pdf'):
        pdf_reader = PyPDF2.PdfReader(file.stream)
        pages = []
        for page_num in range(len(pdf_reader.pages)):
            pages.append(pdf_reader.pages[page_num].extract_text())
        return '\n'.join(pages)
    elif filename.endswith('.docx'):
        document = docx.Document(file.stream)
        fullText = []
        for para in document.paragraphs:
            fullText.append(para.text)
        return '\n'.join(fullText)
    else:
        raise ValueError("Неподдерживаемый формат файла")

if __name__ == '__main__':
    app.run(debug=True)
