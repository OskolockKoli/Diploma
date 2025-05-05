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

# Количество используемого процента данных
n = 1.0
total_samples = len(X)
num_samples_to_use = int(total_samples * n)

best_accuracy = 0
best_model = None
best_vectorizer = None

for iteration in range(100):
    # Устанавливаем фиксированный seed для данной итерации
    np.random.seed(iteration + 42)
    
    # Случайная выборка заданного количества образцов
    indices = np.random.choice(total_samples, num_samples_to_use, replace=False)
    X_subset = [X[i] for i in indices]
    y_subset = [y[i] for i in indices]

    # Разделяем данные на тренировочную и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X_subset, y_subset, test_size=0.35, random_state=iteration+42)

    # Создание векторизатора TF-IDF
    vectorizer = TfidfVectorizer()

    # Преобразуем тексты в числовые признаки
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Создаем и обучаем модель FFNN (Feedforward Neural Network)
    ffbp_model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        learning_rate_init=0.001,
        random_state=42,
        max_iter=1000
    )
    ffbp_model.fit(X_train_vec, y_train)

    # Предсказываем классы на тестовых данных
    y_pred = ffbp_model.predict(X_test_vec)

    # Вычисляем точность модели
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Iteration {iteration}: Accuracy: {accuracy:.4f}')

    # Если данная модель точнее предыдущих, запоминаем её
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = ffbp_model
        best_vectorizer = vectorizer

# Сохраняем лучшую модель и векторизатор после окончания всех итераций
if best_model is not None:
    joblib.dump(best_model, 'best_ffbp_model.pkl')
    joblib.dump(best_vectorizer, 'best_tfidf_ffbp_vectorizer.pkl')
    print(f'\nBest Model Accuracy: {best_accuracy:.4f}\nModel saved.')
else:
    print('No models were trained successfully.')
