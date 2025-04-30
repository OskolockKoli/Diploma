#RNN

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
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

# Подготовка данных для RNN
def prepare_rnn_data(X_train, X_test, tokenizer, max_length=100):
    tokenizer.fit_on_texts(X_train + X_test)
    sequences = tokenizer.texts_to_sequences(X_train + X_test)
    padded_sequences = pad_sequences(sequences, maxlen=max_length)
    return padded_sequences[:len(X_train)], padded_sequences[len(X_train):]

# Инициализация токенайзера
tokenizer = Tokenizer(num_words=5000)

# Описание архитектуры RNN-модели
def create_rnn_model():
    model = Sequential([
        Embedding(input_dim=5000, output_dim=128),
        LSTM(units=128, dropout=0.3, recurrent_dropout=0.3),
        Dense(units=len(np.unique(y)), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Переменные для хранения лучшей модели
best_accuracy = 0
best_model = None

for iteration in range(100):
    # Установим случайное состояние для воспроизводимости
    np.random.seed(iteration + 42)

    # Перешивем индексные выборки для текущего прохода
    total_samples = len(X)
    indices = np.random.permutation(total_samples)
    X_subset = [X[i] for i in indices]
    y_subset = [y[i] for i in indices]

    # Разделение на тренировочный и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(X_subset, y_subset, test_size=0.25, random_state=iteration+42)

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    # Токенизация и подготовка последовательности символов
    X_train_padded, X_test_padded = prepare_rnn_data(X_train, X_test, tokenizer)

    #print("Shape of training data:", X_train_padded.shape)
    #print("Shape of labels:", y_train.shape)

    # Создаем новую модель
    rnn_model = create_rnn_model()

    # Обучение модели
    history = rnn_model.fit(X_train_padded, y_train, batch_size=32, epochs=100, validation_data=(X_test_padded, y_test))
    # Проверка качества модели
    y_pred = rnn_model.predict(X_test_padded)
    y_pred_classes = np.argmax(y_pred, axis=-1)
    test_accuracy = accuracy_score(y_test, y_pred_classes)
    print(f'Iteration {iteration}: Accuracy: {test_accuracy:.4f}')

    # Если текущая модель лучше прежней, запоминаем её
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_model = rnn_model

# Сохраняем лучшую модель
if best_model is not None:
    best_model.save('best_rnn_model.kerasl')
    joblib.dump(best_vectorizer, 'best_tfidf_rnn_vectorizer.keras')
    print(f'\nBest Model Accuracy: {best_accuracy:.4f}\nModel saved.')
else:
    print('No models were trained successfully.')
