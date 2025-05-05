import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Flatten

# Функция для загрузки данных
def load_data(path):
    X = []  # список всех текстов
    y = []  # список меток жанров

    for genre in os.listdir(path):
        genre_path = os.path.join(path, genre)
        if not os.path.isdir(genre_path):
            continue
        for filename in os.listdir(genre_path):
            with open(os.path.join(genre_path, filename), 'r', encoding='iso-8859-1') as file:
                text = file.read()
                X.append(text)
                y.append(genre)

    # Преобразование меток жанров в числовые значения
    le = LabelEncoder()
    y = le.fit_transform(y)

    return X, y

# Подготовка данных и разделение на тренировочные/тестовые наборы
X, y = load_data('books_max_balanced')

# Разделение на тренировочный и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Реализация функций для тренировки и оценки моделей
def fit_and_evaluate(model, X_train, y_train, X_test, y_test, vectorizer=None):
    if vectorizer is None:
        vectorizer = TfidfVectorizer()

    # Преобразование текстов в числовые признаки
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Обучение модели
    model.fit(X_train_vec, y_train)

    # Прогнозирование на тестовом наборе
    y_pred = model.predict(X_test_vec)

    # Оценка точности
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}')
    return accuracy

# Проведение серии экспериментов для изменения размера тренировочного набора
def run_experiments(model, vectorizer=None):
    results = []
    sizes = np.arange(0.5, 1.01, 0.1)

    for size in sizes:
        X_train_subset, _, y_train_subset, _ = train_test_split(
            X_train, y_train, train_size=size, random_state=42
        )
        accuracy = fit_and_evaluate(model, X_train_subset, y_train_subset, X_test, y_test, vectorizer)
        results.append(accuracy)

    return sizes * 100, results

#NB
nb_model = MultinomialNB(alpha=0.001)
sizes_nb, accuracies_nb = run_experiments(nb_model)

#SVM
svm_model = LinearSVC(max_iter=20000)
sizes_svm, accuracies_svm = run_experiments(svm_model)

#DT
dt_model = DecisionTreeClassifier(random_state=42)
sizes_dt, accuracies_dt = run_experiments(dt_model)

#FFBP
ffbp_model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', random_state=42, max_iter=1000)
sizes_ffbp, accuracies_ffbp = run_experiments(ffbp_model)

#RNN
def prepare_rnn_data(X_train, X_test, tokenizer, max_length=100):
    tokenizer.fit_on_texts(X_train + X_test)
    sequences = tokenizer.texts_to_sequences(X_train + X_test)
    padded_sequences = pad_sequences(sequences, maxlen=max_length)
    return padded_sequences[:len(X_train)], padded_sequences[len(X_train):]

def prepare_dataset(X_train_padded, y_train):
    dataset = tf.data.Dataset.from_tensor_slices((X_train_padded, y_train))
    dataset = dataset.batch(32)
    return dataset

tokenizer = Tokenizer(num_words=5000)
X_train_padded, X_test_padded = prepare_rnn_data(X_train, X_test, tokenizer)

rnn_model = Sequential([
    Embedding(input_dim=5000, output_dim=128),
    LSTM(units=64),
    Dense(len(np.unique(y)), activation='softmax')
])
rnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

def run_rnn_experiments():
    results = []
    sizes = np.arange(0.5, 1.01, 0.1)

    for size in sizes:
        X_train_subset, _, y_train_subset, _ = train_test_split(
            X_train_padded, y_train, train_size=size, random_state=42
        )
        dataset = prepare_dataset(X_train_subset, y_train_subset)
        rnn_model.fit(dataset, epochs=10, verbose=0)
        y_pred = rnn_model.predict(X_test_padded)  # Используем predict
        y_pred = np.argmax(y_pred, axis=-1)  # Получаем индексы классов
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy:.4f}')
        results.append(accuracy)

    return sizes * 100, results

sizes_rnn, accuracies_rnn = run_rnn_experiments()

#DAN2
dan2_model = Sequential([
    Embedding(input_dim=5000, output_dim=128),
    GlobalAveragePooling1D(),
    Dense(len(np.unique(y)), activation='softmax')
])
dan2_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

def run_dan2_experiments():
    results = []
    sizes = np.arange(0.5, 1.01, 0.1)

    for size in sizes:
        X_train_subset, _, y_train_subset, _ = train_test_split(
            X_train_padded, y_train, train_size=size, random_state=42
        )
        dataset = prepare_dataset(X_train_subset, y_train_subset)
        dan2_model.fit(dataset, epochs=10, verbose=0)
        y_pred = dan2_model.predict(X_test_padded)  # Используем predict
        y_pred = np.argmax(y_pred, axis=-1)  # Получаем индексы классов
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy:.4f}')
        results.append(accuracy)

    return sizes * 100, results

sizes_dan2, accuracies_dan2 = run_dan2_experiments()

#CNN
cnn_model = Sequential([
    Embedding(input_dim=5000, output_dim=128),
    Conv1D(filters=64, kernel_size=3, padding='same'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(len(np.unique(y)), activation='softmax')
])
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

def run_cnn_experiments():
    results = []
    sizes = np.arange(0.5, 1.01, 0.1)

    for size in sizes:
        X_train_subset, _, y_train_subset, _ = train_test_split(
            X_train_padded, y_train, train_size=size, random_state=42
        )
        dataset = prepare_dataset(X_train_subset, y_train_subset)
        cnn_model.fit(dataset, epochs=10, verbose=0)
        y_pred = cnn_model.predict(X_test_padded)  # Используем predict
        y_pred = np.argmax(y_pred, axis=-1)  # Получаем индексы классов
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy:.4f}')
        results.append(accuracy)

    return sizes * 100, results

sizes_cnn, accuracies_cnn = run_cnn_experiments()

# Постройка графиков
plt.figure(figsize=(12, 6))

plt.plot(sizes_nb, accuracies_nb, label='Naive Bayes', marker='o')
plt.plot(sizes_svm, accuracies_svm, label='SVM', marker='x')
plt.plot(sizes_dt, accuracies_dt, label='Decision Tree', marker='+')
plt.plot(sizes_ffbp, accuracies_ffbp, label='Feedforward NN', marker='*')
plt.plot(sizes_rnn, accuracies_rnn, label='RNN', marker='^')
plt.plot(sizes_dan2, accuracies_dan2, label='DAN2', marker='v')
plt.plot(sizes_cnn, accuracies_cnn, label='CNN', marker='>')

plt.xlabel('Размер тренировочного набора данных (%)', fontsize=14)
plt.ylabel('Точность', fontsize=14)
plt.title('Зависимость точности классификатора от размера набора данных', fontsize=16)
plt.legend(fontsize=14)
plt.grid(True)
plt.show()
