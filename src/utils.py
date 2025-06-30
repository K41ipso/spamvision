import pandas as pd
import os
import chardet


def load_and_clean_data(path: str) -> pd.DataFrame:
    """
    Загружает и чистит датасет:
    - убирает пропуски
    - переименовывает колонки
    - нормализует текст
    """

    # Проверяем в каком формате находятся данные в датасете
    with open(path, "rb") as f:
        result = chardet.detect(f.read(10000))
    encoding_type = result.get("encoding")

    # Преобразуем датасет из csv к DataFrame
    dataset_df = pd.read_csv(filepath_or_buffer=path, encoding=encoding_type)

    # Переименовываем нужные нам столбцы
    renamed_df = dataset_df.rename(columns={"v1": "label", "v2": "message"})

    # Оставляем только те столбцы данных, которые нам нужны на выходе
    result_df = renamed_df[["label", "message"]]

    return result_df


# Для отладки
if __name__ == "__main__":
    # Путь к тестовому CSV-файлу
    test_path = "data/spam.csv"

    # Отключаем ограничение на число столбцов
    pd.set_option('display.max_columns', None)

    # Также может понадобиться отключить ширину столбца текста (если у тебя длинные строки)
    pd.set_option('display.width', None)

    # И если хочешь, чтобы и строки выводились все (а не только первые/последние):
    pd.set_option('display.max_rows', None)

    try:
        # Получаем путь к текущему скрипту (utils.py)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Строим путь к файлу
        test_path = os.path.join(current_dir, "..", "data", "raw", "spam.csv")

        cleaned_df = load_and_clean_data(test_path)
        print("✅ Данные успешно загружены и обработаны!")
        print(cleaned_df.head())
    except Exception as e:
        print(f"❌ Произошла ошибка: {e}")