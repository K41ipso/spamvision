import tkinter as tk
from tkinter import messagebox
import joblib
import os

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

    def on_check(self):
        # Забираем вставленный текст в поле для ввода
        text = self.text_input.get("1.0", tk.END).strip()
        # Отладочный вывод в консоль
        print(text)
        if text:
            # ЛОГИКА АНАЛИЗА ТЕКСТА
            vec_text_x = self.vectorizer.transform([text])
            is_spam = self.model.predict(vec_text_x)[0]
            # логика вывода сообщения
            if is_spam:
                result = "Это спам!"
                fg = "#b00020"
            else:
                result = "Не спам"
                fg = "#00695c"
            # Изменяем окно в зависимости от результата работы модели
            self.result_label.config(text=result, fg=fg)
        else:
            # Если сообщение не введено
            self.result_label.config(text="Введите сообщение для анализа.")

# Отладочный вызов с запуском приложения
if __name__ == "__main__":
    app = SpamDetectionApp()