"""
Структура данного файла:

🔹 Импорты
🔹 Загрузка и очистка данных через `utils`
🔹 Разделение на X (тексты) и y (метки)
🔹 Преобразование y (в 0 и 1)
🔹 TF-IDF векторизация X
🔹 Train/test split
🔹 Обучение Logistic Regression
🔹 Вывод метрик (Accuracy, F1)
🔹 Сохранение модели и векторайзера

Структура main функции:

Вход:
    1. CSV-файл с колонками label и message
Выход:
    1. model.pkl (LogReg)
    2. vectorizer.pkl (TF-IDF)
    3. напечатанные метрики (в консоль)
"""

# Для работы с данными
import pandas as pd
# Для векторизации текста
from sklearn.feature_extraction.text import TfidfVectorizer
# Для разделения
from sklearn.model_selection import train_test_split
# Модели
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
# Для метрик
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
# Для сохранения
import joblib
import os
# Чтобы импортировать подготовленную функцию - обработчик данных
from utils import load_and_clean_data

text_line = "================================================================"

def train_model(train_path: str, sk_model, model_name: str) -> None:
    """
    :param model_name: Наименование модели для предсказания
    :param sk_model: Импортируемый объект - модель
    :param train_path: это путь к данным, на которых будет обучаться модель
    :return: функция сохраняет результат обработки в model.pkl (LogReg) и vectorizer.pkl (TF-IDF)
    а также печатает метрики (в консоль)
    """

    # Первичная обработка данных
    clean_data_df = load_and_clean_data(train_path)

    # Для взаимодействия модели с нашими тестовыми данными
    clean_data_df["label"] = clean_data_df["label"].map({"ham": 0, "spam": 1})

    # Разделение на X (тексты) и y (метки)
    X = clean_data_df['message']  # Признаки — текст
    y = clean_data_df['label']  # Целевой признак — метка (0/1)

    # TF-IDF векторизация X
    vectorizer = TfidfVectorizer(
        lowercase=True,         # преобразует все X к строчному формату
        #stop_words='english',  # использует вшитый словарь предлогов и междометий на английском языке
        ngram_range=(1, 3),     # создаем матрицу как для соло слов, так и для биграмм (словосочетаний из 2х слов)
        min_df=2,               # default - (измеряется в шт.) исключает слова, которые встречаются меньше чем в 1 документе
        max_df=0.9              # default - (измеряется в %) исключает слова, которые встречаются более чем в 100% документов
    )

    # Векторизуем поданный на вход текст
    # мы не можем векторизовать все данные сразу, так как будет утечка данных, модель "увидит" результаты. Такая ситуация называется Data Leakage
    # vectorized_X = vectorizer.fit_transform(X)

    # Выводим первые 10 векторизованных слов
    #print(vectorizer.get_feature_names_out()[:10])

    # Разделяем данные на обучающую и тестовую выборку
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    X_train_vect = vectorizer.fit_transform(X_train)
    X_test_vect = vectorizer.transform(X_test)

    # Обучаем модель на наших данных
    model = sk_model.fit(X_train_vect, y_train)
    print(text_line, f"\nМодель {model_name} успешно обучена на тренировочных данных.")

    # Делаем предсказание на тестовых данных
    y_pred = model.predict(X_test_vect)
    print(f"{text_line}\nПредсказание успешно выполнено.")

    # Выводим метрики качества предсказаний
    print(f"{text_line}\nAccuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print(f"{text_line}\nClassification Report:\n", classification_report(y_test, y_pred))
    print(f"{text_line}\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Сохранение модели и векторайзера
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(models_dir, exist_ok=True)

    joblib.dump(model, os.path.join(models_dir, "model.pkl"))
    joblib.dump(vectorizer, os.path.join(models_dir, "vectorizer.pkl"))

    print(f"{text_line}\nДанные успешно сохранены.")

    # Выводим топ 10 СПАМ-признаков
    words = vectorizer.get_feature_names_out()
    weights = model.coef_[0]

    top = sorted(zip(weights, words), reverse=True)[:10]
    print(f"{text_line}\n🔝 Топ-10 признаков, указывающих на СПАМ:")
    for w, word in top:
        print(f"{word}: {w:.3f}")
    print(text_line)

    return

# Для отладки
if __name__ == "__main__":

    # Для корректного вывода DataFrame
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_rows', None)

    try:
        # Получаем путь к текущему скрипту (utils.py)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Строим путь к файлу
        current_path = os.path.join(current_dir, "..", "data", "raw", "spam.csv")
        #train_model(train_path=current_path, sk_model=LogisticRegression(), model_name="LogisticRegression")
        #train_model(train_path=current_path, sk_model=MultinomialNB(), model_name="MultinomialNB")
        # Данная модель показала себя самой продуктивной в рамках основных метрик
        train_model(train_path=current_path, sk_model=LinearSVC(), model_name="LinearSVC")
        print("✅ Данные успешно загружены и обработаны!")
    except Exception as e:
        print(f"❌ Произошла ошибка: {e}")