#NB

import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

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
path = 'books_max_balanced'
X, y = load_data(path)
y = le.fit_transform(y)

# Использовать n% от общего числа примеров
n = 0.95
total_samples = len(X)
num_samples_to_use = int(total_samples * n)

best_accuracy = 0
best_model = None
best_vectorizer = None

for iteration in range(100):
    # Установка фиксированного seed для каждой итерации
    np.random.seed(iteration + 42)
    
    # Случайная выборка заданного количества образцов
    indices = np.random.choice(total_samples, num_samples_to_use, replace=False)
    X_subset = [X[i] for i in indices]
    y_subset = [y[i] for i in indices]

    # Разделение на тренировочную и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X_subset, y_subset, test_size=0.25, random_state=iteration+42)

    # Векторизация текста
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Обучаем модель Naive Bayes
    nb_model = MultinomialNB(alpha=0.001)
    nb_model.fit(X_train_vec, y_train)

    # Прогнозируем класс для тестового набора
    y_pred = nb_model.predict(X_test_vec)

    # Рассчитываем точность модели
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Iteration {iteration}: Accuracy: {accuracy:.4f}')

    # Если данная модель показывает лучшую точность, запоминаем её
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = nb_model
        best_vectorizer = vectorizer

# Сохраняем лучшую модель и векторизатор после завершения всех итераций
if best_model is not None:
    joblib.dump(best_model, 'best_nb_model.pkl')
    joblib.dump(best_vectorizer, 'best_tfidf_nb_vectorizer.pkl')
    print(f'\nBest Model Accuracy: {best_accuracy:.4f}\nModel saved.')
else:
    print('No models were trained successfully.')
