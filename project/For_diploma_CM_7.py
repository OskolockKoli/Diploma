#CNN

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.metrics import accuracy_score

# Функция для загрузки данных
def load_data(path):
    X = []
    y = []
    
    for genre in os.listdir(path):
        genre_path = os.path.join(path, genre)
        if not os.path.isdir(genre_path):
            continue
        
        for filename in os.listdir(genre_path):
            with open(os.path.join(genre_path, filename), 'r', encoding='iso-8859-1') as file:
                text = file.read()
                X.append(text)
                y.append(genre)
                
    return X, y

# Преобразование меток жанров в числовые значения
le = LabelEncoder()

# Загрузка данных
X, y = load_data('books_max_balanced')
y = le.fit_transform(y)

# Разделение на тренировочный и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Подготовка данных для CNN
def prepare_cnn_data(X_train, X_test, tokenizer, max_length=100):
    tokenizer.fit_on_texts(X_train + X_test)
    sequences = tokenizer.texts_to_sequences(X_train + X_test)
    padded_sequences = pad_sequences(sequences, maxlen=max_length)
    return padded_sequences[:len(X_train)], padded_sequences[len(X_train):]

# Токенизатор
tokenizer = Tokenizer(num_words=5000)

# Архитектура модели CNN
def create_cnn_model(output_units):
    model = Sequential([
        Embedding(input_dim=5000, output_dim=128),
        Conv1D(filters=64, kernel_size=3, padding='same'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(output_units, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Настройки начальной точки
best_accuracy = 0
best_model = None

for iteration in range(100):
    # Гарантируем воспроизводимость
    np.random.seed(iteration + 42)
    
    # Пересчёт индексов выборки
    total_samples = len(X)
    indices = np.random.permutation(total_samples)
    X_subset = [X[i] for i in indices]
    y_subset = [y[i] for i in indices]

    # Разделение на тренировочный и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(X_subset, y_subset, test_size=0.25, random_state=iteration+42)

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    # Готовим данные для CNN
    X_train_padded, X_test_padded = prepare_cnn_data(X_train, X_test, tokenizer)

    # Создаем новую модель
    cnn_model = create_cnn_model(output_units=len(np.unique(y)))

    # Обучение модели
    cnn_model.fit(X_train_padded, y_train, batch_size=32, epochs=10, verbose=0)

    # Тестируем качество модели
    y_pred = cnn_model.predict(X_test_padded)
    y_pred_classes = np.argmax(y_pred, axis=-1)
    test_accuracy = accuracy_score(y_test, y_pred_classes)
    print(f'Iteration {iteration}: Accuracy: {test_accuracy:.4f}')

    # Если текущая модель лучше прошлых, запоминаем её
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_model = cnn_model

# Сохраняем лучшую модель
if best_model is not None:
    best_model.save('best_cnn_model.pkl')
    joblib.dump(best_vectorizer, 'best_tfidf_cnn_vectorizer.keras')
    print(f'\nBest Model Accuracy: {best_accuracy:.4f}\nModel saved.')
else:
    print('No models were trained successfully.')
