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
# Для модели
from sklearn.linear_model import LogisticRegression
# Для метрик
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
# Для сохранения
import joblib
import os
# Чтобы импортировать подготовленную функцию - обработчик данных
from utils import load_and_clean_data

def train_model(train_path: str) -> None:
    """
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
        stop_words='english',   # использует вшитый словарь предлогов и междометий на английском языке
        ngram_range=(1, 2),     # создаем матрицу как для соло слов, так и для биграмм (словосочетаний из 2х слов)
        min_df=1,               # default - (измеряется в шт.) исключает слова, которые встречаются меньше чем в 1 документе
        max_df=1.0              # default - (измеряется в %) исключает слова, которые встречаются более чем в 100% документов
    )

    # Векторизуем поданный на вход текст
    vactorized_X = vectorizer.fit_transform(X)

    # Выводим первые 30 векторов
    print(vectorizer.get_feature_names_out()[:30])

    #Отладочная печать в консоль
    #print(X.head())

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
        train_model(train_path=current_path)
        print("✅ Данные успешно загружены и обработаны!")
    except Exception as e:
        print(f"❌ Произошла ошибка: {e}")