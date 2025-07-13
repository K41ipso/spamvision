import tkinter as tk
from copy import deepcopy
from tkinter import messagebox
import joblib
import os
from langdetect import detect
from googletrans import Translator
import logging


# Добавляем логгер для вывода информации
logging.basicConfig(
    level=logging.INFO,  # уровень логирования: INFO, DEBUG, WARNING, ERROR
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Класс наследуется от tk.Tk для того, чтобы создать главное окно приложения
class SpamDetectionApp(tk.Tk):
    # Эта конструкция нужна для того, чтобы при создании нового экземпляра класса SpamDetectionApp вызывался конструктор
    def __init__(self):
        # Для правильной инициализации родительского класса (вызываем его конструктор)
        super().__init__()

        # Забираем пути модели и векторизатора
        project_root = os.path.dirname(os.path.dirname(__file__))
        model_path = os.path.join(project_root, "models", "model.pkl")
        vectorizer_path = os.path.join(project_root, "models", "vectorizer.pkl")

        # Загружаем модель для предсказания и векторизатор
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

        # Создаем экземпляр переводчика
        self.translator = Translator()

        # Задаем заголовок окна
        self.title("Spam Detector")
        # Задаем размеры окна
        self.geometry(newGeometry="960x540")
        # Добавляем многострочное поле ввода
        self.text_input = tk.Text(
            self,
            height=10,
            width=80,
            bg="#fdf6e3",
            fg="#333333",
            bd=1,
            relief="solid",
            font=("Segoe UI", 12)
        )
        # Прикручиваем поле ввода к главному окну
        self.text_input.pack(
            fill="none",
            expand=False,
            anchor="center"
        )
        # Подключаем стандартные горячие клавиши к текстовому полю
        self.text_input.bind("<Control-a>", self.select_all)
        self.text_input.bind("<Control-A>", self.select_all)  # На всякий случай для заглавной A
        self.text_input.bind("<Control-v>", self.on_paste)
        self.text_input.bind("<Control-V>", self.on_paste)
        self.text_input.bind("<Control-c>", self.on_copy)
        self.text_input.bind("<Control-C>", self.on_copy)
        self.text_input.bind("<Control-x>", self.on_cut)
        self.text_input.bind("<Control-X>", self.on_cut)
        # Добавляем кнопку проверки
        self.check_button = tk.Button(
            self,
            text="ПРОВЕРИТЬ ТЕКСТ НА СПАМ",
            command=self.on_check
        )
        # Прикрепляем кнопку к главному экрану
        self.check_button.pack(pady=10)
        # Добавляем поле метки
        self.result_label = tk.Label(
            self,
            text="",
            font=("Segoe UI", 12, "bold"),
            bg="white" # "#fefefe"
        )
        # Прикрепляем поле к главному экрану
        self.result_label.pack(pady=10)
        # Запускаем цикл обработки событий, чтобы окно не закрывалось
        self.mainloop()

    def select_all(self, event):
        self.text_input.tag_add("sel", "1.0", "end")
        return "break"

    def on_paste(self, event):
        self.text_input.event_generate("<<Paste>>")
        return "break"

    def on_copy(self, event):
        self.text_input.event_generate("<<Copy>>")
        return "break"

    def on_cut(self, event):
        self.text_input.event_generate("<<Cut>>")
        return "break"

    def get_input_text(self) -> str:
        return self.text_input.get("1.0", tk.END).strip()

    def show_empty_input_warning(self) -> None:
        # Если сообщение не введено
        self.result_label.config(text="Введите сообщение для анализа.")

    def translate_to_english(self, input_text: str) -> str:
        logger.info(f"Введенный текст: {input_text}")
        try:
            languages_text = detect(input_text)
        except Exception as e:
            logger.warning("Не удалось определить язык текста. Используется 'en' по умолчанию.")
            languages_text = "en"
        logger.info(f"Данный текст на {languages_text} языке.")
        # Проверяем язык введенного текста
        if languages_text != "en":
            # Если язык не английский, то пытаемся его перевести на английский
            try:
                result_translate = self.translator.translate(
                    text=input_text,
                    src=languages_text,
                    dest='en'
                )
                output_text = result_translate.text
            except Exception as e:
                logger.error("Произошла неожиданная ошибка, текст остается необработанным.")
                output_text = input_text
            logger.info(f"Переведенный текст: {output_text}")
        else:
            output_text = input_text
        return output_text

    def classify_text(self, translated_text: str) -> int:
        vec_text_x = self.vectorizer.transform([translated_text])
        is_spam = self.model.predict(vec_text_x)[0]
        return is_spam

    def update_result_label(self, is_spam: int) -> None:
        # логика вывода сообщения
        if is_spam:
            result = "Это спам!"
            fg = "#b00020"
        else:
            result = "Не спам"
            fg = "#00695c"
        # Изменяем окно в зависимости от результата работы модели
        self.result_label.config(text=result, fg=fg)

    def on_check(self) -> None:
        input_text = self.get_input_text()
        if not input_text:
            self.show_empty_input_warning()
            return

        translated_text = self.translate_to_english(input_text)
        is_spam = self.classify_text(translated_text)
        self.update_result_label(is_spam)

# Отладочный вызов с запуском приложения
if __name__ == "__main__":
    app = SpamDetectionApp()