#FFBP

# Импортируем необходимые библиотеки
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Функция для загрузки данных
def load_data(path):
    X = []  # Список всех текстов
    y = []  # Список меток жанров

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
path = 'books_max_balanced'
X, y = load_data(path)
y = le.fit_transform(y)

# Ограничиваем количество примеров до n%
n = 1.0
total_samples = len(X)
num_samples_to_use = int(total_samples * n)
indices = np.random.choice(total_samples, num_samples_to_use, replace=False)

X_subset = [X[i] for i in indices]
y_subset = [y[i] for i in indices]

# Разделение на тренировочный и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X_subset, y_subset, test_size=0.35, random_state=42)

# Реализация функции для тренировки и оценки модели
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

# Запуск эксперимента для Feedforward Neural Network
ffbp_model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', alpha=0.0001, learning_rate_init=0.001, random_state=42, max_iter=1000)
accuracy = fit_and_evaluate(ffbp_model, X_train, y_train, X_test, y_test)

# Сохраняем модель
joblib.dump(ffbp_model, 'ffbp_model.pkl')
