import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
import numpy as np
import insightface
import os
import logging
from tkinter import messagebox
from config_manager import get_config, CONFIG_FILE # Импортируем наш новый менеджер

# --- Настройка логирования для GUI ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])

# --- Безопасный импорт telegram_notifier ---
# Если модуль telegram_notifier не найден, предоставляем заглушку
try:
    from telegram_notifier import send_test_message
except ImportError:
    def send_test_message():
        logging.warning("Модуль 'telegram_notifier' не найден.")
        messagebox.showerror("Ошибка", "Не удалось импортировать функцию для отправки тестового сообщения Telegram. Убедитесь, что 'telegram_notifier.py' находится в той же директории или установлен.")

# --- Константы приложения ---
NUM_SNAPSHOTS = 3 # Количество снимков, необходимых для регистрации
PROMPTS = ["Посмотрите прямо в камеру...", "Теперь немного поверните голову влево", "Супер! Теперь немного вправо"] # Подсказки для пользователя

# --- ОБНОВЛЕННАЯ ФУНКЦИЯ АНАЛИЗА КАЧЕСТВА КАДРА (ТОЛЬКО ЯРКОСТЬ) ---
def is_frame_quality_good(frame_roi, bright_min, bright_max):
    """
    Проверяет переданную область (ROI) только на яркость.
    Возвращает True, если ROI качественный, иначе False.
    """
    if frame_roi is None or frame_roi.shape[0] == 0 or frame_roi.shape[1] == 0:
        logging.warning("ROI лица пуст или некорректен для проверки качества.")
        return False

    gray_roi = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
    
    # 1. Проверка на яркость (среднее значение пикселей)
    brightness = np.mean(gray_roi)
    if not (bright_min <= brightness <= bright_max):
        logging.warning(f"Качество ROI лица: ПЛОХОЕ. Яркость ({brightness:.2f}) вне диапазона ({bright_min}-{bright_max}).")
        return False
        
    logging.info(f"Качество ROI лица: ХОРОШЕЕ. Яркость={brightness:.2f}")
    return True

class FaceGuardCenter(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Центр Управления FaceGuard")
        self.geometry("1100x800") # Возвращаем исходную высоту
        self.resizable(False, False)
        ctk.set_appearance_mode("dark") # Устанавливаем темную тему по умолчанию

        # --- Инициализация переменных состояния ---
        self.embeddings = [] # Список для хранения эмбеддингов лица во время регистрации
        self.snapshot_count = 0 # Счетчик сделанных снимков
        self._loading_config = False # Флаг для предотвращения сохранения во время загрузки

        self.config = get_config() # Используем централизованный менеджер для получения конфига

        # --- Настройка сетки главного окна ---
        self.grid_columnconfigure(0, weight=2) # Колонка для регистрации занимает больше места
        self.grid_columnconfigure(1, weight=1) # Колонка для настроек занимает меньше места
        self.grid_rowconfigure(0, weight=1) # Единственная строка занимает все доступное пространство

        # --- Создание фрейма для секции регистрации ---
        enroll_frame = ctk.CTkFrame(self)
        enroll_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.create_enrollment_widgets(enroll_frame) # Заполнение фрейма виджетами регистрации

        # --- Создание фрейма для секции настроек ---
        settings_frame = ctk.CTkFrame(self)
        settings_frame.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="nsew")
        self.create_settings_widgets(settings_frame) # Заполнение фрейма виджетами настроек

        # --- Загрузка модели распознавания лиц InsightFace ---
        print("Загрузка модели распознавания лиц...")
        # 'buffalo_l' - распространенная модель InsightFace, предлагает хороший баланс точности/скорости
        # 'CPUExecutionProvider' - принудительное использование CPU, если GPU недоступен или не настроен
        self.face_analyzer = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.face_analyzer.prepare(ctx_id=-1, det_size=(640, 640)) # Подготовка модели
        print("Модель загружена.")

        # --- Инициализация захвата видео с веб-камеры ---
        self.cap = cv2.VideoCapture(0) # 0 обычно означает веб-камеру по умолчанию
        self.update_frame() # Запуск обновления видеопотока

    def create_enrollment_widgets(self, parent_frame):
        """Создает виджеты для секции регистрации лица."""
        parent_frame.grid_rowconfigure(0, weight=1)
        parent_frame.grid_columnconfigure(0, weight=1)

        # Метка для отображения видеопотока с веб-камеры
        self.video_label = ctk.CTkLabel(parent_frame, text="")
        self.video_label.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Фрейм для размещения элементов управления (счетчик, подсказка, кнопка)
        control_frame = ctk.CTkFrame(parent_frame, fg_color="transparent")
        control_frame.grid(row=1, column=0, pady=10)

        # Метка для отображения прогресса регистрации
        self.count_label = ctk.CTkLabel(control_frame, text=f"Сделано снимков: 0 из {NUM_SNAPSHOTS}", font=("Arial", 16))
        self.count_label.pack(pady=5)

        # Метка для отображения подсказок пользователю
        self.prompt_label = ctk.CTkLabel(control_frame, text=PROMPTS[0], font=("Arial", 18))
        self.prompt_label.pack(pady=10)

        # Кнопка для запуска процесса снимка
        self.snapshot_button = ctk.CTkButton(control_frame, text="Сделать снимок", command=self.take_snapshot, font=("Arial", 20), height=50)
        self.snapshot_button.pack(pady=20)

    def update_frame(self):
        """Непрерывно обновляет кадр с веб-камеры на GUI."""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read() # Читаем кадр с камеры
            if ret: # Если кадр успешно прочитан
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # OpenCV читает в BGR, Pillow и Tkinter ожидают RGB
                img = Image.fromarray(frame_rgb) # Преобразуем массив NumPy в объект PIL Image
                imgtk = ImageTk.PhotoImage(image=img) # Преобразуем PIL Image в объект Tkinter PhotoImage
                self.video_label.imgtk = imgtk # Сохраняем ссылку на объект, чтобы избежать его сборки мусором
                self.video_label.configure(image=imgtk) # Обновляем изображение в метке
            self.after(10, self.update_frame) # Планируем вызов этой функции снова через 10 мс

    def take_snapshot(self):
        """
        Захватывает кадр, детектирует лицо, проверяет качество и сохраняет эмбеддинг.
        """
        if not (self.cap and self.cap.isOpened()): return # Проверка, что камера активна
            
        ret, frame = self.cap.read()
        if not ret:
            self.prompt_label.configure(text="Ошибка камеры! Не удалось получить кадр."); return
            
        faces = self.face_analyzer.get(frame) # Детекция лиц
        
        if not faces:
            self.prompt_label.configure(text="Лицо не найдено! Попробуйте снова."); return
        if len(faces) > 1:
            self.prompt_label.configure(text="Найдено несколько лиц! В кадре должен быть один."); return
            
        # Берем первое (и единственное) обнаруженное лицо
        detected_face = faces[0]
        bbox = detected_face.bbox.astype(int)
        # Вычисляем ROI лица, убеждаясь, что координаты не выходят за границы кадра
        x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), min(frame.shape[1], bbox[2]), min(frame.shape[0], bbox[3])
        face_roi = frame[y1:y2, x1:x2] # Вырезаем область лица

        # Получаем значения из UI и вычисляем диапазон яркости
        target_bright = self.brightness_target_slider.get()
        tolerance = self.brightness_tolerance_slider.get()
        bright_min = int(max(0, target_bright - tolerance)) # Убеждаемся, что мин >= 0
        bright_max = int(min(255, target_bright + tolerance)) # Убеждаемся, что макс <= 255

        # --- ОБНОВЛЕННАЯ ПРОВЕРКА КАЧЕСТВА (ТОЛЬКО ЯРКОСТЬ) ---
        if not is_frame_quality_good(face_roi, bright_min, bright_max):
            self.prompt_label.configure(text="Плохое качество кадра лица! Измените освещение.")
            return
        # --- КОНЕЦ ОБНОВЛЕННОЙ ПРОВЕРКИ ---
            
        self.embeddings.append(detected_face.embedding) # Сохраняем эмбеддинг лица
        self.snapshot_count += 1
        self.count_label.configure(text=f"Сделано снимков: {self.snapshot_count} из {NUM_SNAPSHOTS}")
        
        if self.snapshot_count >= NUM_SNAPSHOTS:
            self.finalize_enrollment() # Если сделано достаточно снимков, завершаем регистрацию
        else:
            self.prompt_label.configure(text=PROMPTS[self.snapshot_count]) # Обновляем подсказку для следующего снимка

    def finalize_enrollment(self):
        """Завершает процесс регистрации, сохраняя эталонные эмбеддинги."""
        self.prompt_label.configure(text="Все снимки сделаны. Сохраняю...")
        self.snapshot_button.configure(state="disabled", text="Обработка...") # Отключаем кнопку на время сохранения

        if not os.path.exists('data'): # Создаем директорию 'data', если ее нет
            os.makedirs('data')
        
        # Сохраняем собранные эмбеддинги в файл NumPy
        np.save('data/owner_embedding.npy', np.array(self.embeddings))
        
        self.prompt_label.configure(text="Готово! Ваши эталонные образы сохранены.")
        # Переключаем кнопку для возможности начать регистрацию заново
        self.snapshot_button.configure(state="normal", text="Начать заново", command=self.reset_enrollment_state)

    def reset_enrollment_state(self):
        """Сбрасывает состояние регистрации для нового процесса."""
        self.snapshot_count, self.embeddings = 0, [] # Сбрасываем счетчик и очищаем список эмбеддингов
        self.count_label.configure(text=f"Сделано снимков: 0 из {NUM_SNAPSHOTS}")
        self.prompt_label.configure(text=PROMPTS[0]) # Возвращаем первую подсказку
        self.snapshot_button.configure(command=self.take_snapshot, text="Сделать снимок") # Восстанавливаем команду кнопки

    def create_settings_widgets(self, parent_frame):
        """Создает виджеты для секции настроек внутри прокручиваемой области."""
        parent_frame.grid_rowconfigure(0, weight=1)
        parent_frame.grid_columnconfigure(0, weight=1)

        # Создаем прокручиваемый фрейм, который займет все место в parent_frame
        scrollable_frame = ctk.CTkScrollableFrame(parent_frame, label_text="Настройки")
        scrollable_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        scrollable_frame.grid_columnconfigure(0, weight=1)

        # --- Теперь все блоки настроек добавляем в scrollable_frame ---

        # --- Блок "Настройки Распознавания" ---
        rec_frame = ctk.CTkFrame(scrollable_frame)
        rec_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        rec_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(rec_frame, text="Настройки Распознавания", font=("Arial", 16, "bold")).pack(anchor="w", padx=10, pady=5)
        
        self.threshold_label = ctk.CTkLabel(rec_frame, text="Чувствительность (порог): 0.50")
        self.threshold_label.pack(anchor="w", padx=10, pady=(10, 0))
        self.threshold_slider = ctk.CTkSlider(rec_frame, from_=0.1, to=1.0, command=self.update_threshold)
        self.threshold_slider.pack(fill="x", padx=10, pady=5, expand=True)
        
        ctk.CTkButton(rec_frame, text="Удалить регистрацию", command=self.delete_enrollment, fg_color="#D2042D", hover_color="#AA0022").pack(pady=10, padx=10)
        
        # --- Блок "Настройки Качества Снимка" ---
        quality_frame = ctk.CTkFrame(scrollable_frame)
        quality_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=10)
        quality_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(quality_frame, text="Настройки Качества Снимка", font=("Arial", 16, "bold")).pack(anchor="w", padx=10, pady=5)
        
        self.brightness_label = ctk.CTkLabel(quality_frame, text="Допустимая яркость: 90 - 150")
        self.brightness_label.pack(anchor="w", padx=10, pady=(10, 0))
        
        ctk.CTkLabel(quality_frame, text="Целевая яркость (0-255):").pack(anchor="w", padx=10, pady=(5,0))
        self.brightness_target_slider = ctk.CTkSlider(quality_frame, from_=0, to=255, number_of_steps=256, command=self.update_brightness)
        self.brightness_target_slider.pack(fill="x", padx=10, pady=(0,5), expand=True)
        
        ctk.CTkLabel(quality_frame, text="Допуск (+/-):").pack(anchor="w", padx=10, pady=(5,0))
        self.brightness_tolerance_slider = ctk.CTkSlider(quality_frame, from_=0, to=100, number_of_steps=101, command=self.update_brightness)
        self.brightness_tolerance_slider.pack(fill="x", padx=10, pady=(0,5), expand=True)

        # --- Блок "Расширенные Настройки Качества" ---
        adv_quality_frame = ctk.CTkFrame(scrollable_frame)
        adv_quality_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        adv_quality_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(adv_quality_frame, text="Расширенные Настройки Качества", font=("Arial", 16, "bold")).pack(anchor="w", padx=10, pady=5)

        self.det_score_label = ctk.CTkLabel(adv_quality_frame, text="Мин. уверенность детектора: 0.90")
        self.det_score_label.pack(anchor="w", padx=10, pady=(10, 0))
        self.det_score_slider = ctk.CTkSlider(adv_quality_frame, from_=0.1, to=1.0, command=self.update_adv_quality_labels)
        self.det_score_slider.pack(fill="x", padx=10, pady=5, expand=True)

        self.face_size_label = ctk.CTkLabel(adv_quality_frame, text="Мин. размер лица (px): 60")
        self.face_size_label.pack(anchor="w", padx=10, pady=(10, 0))
        self.face_size_slider = ctk.CTkSlider(adv_quality_frame, from_=30, to=200, number_of_steps=171, command=self.update_adv_quality_labels)
        self.face_size_slider.pack(fill="x", padx=10, pady=5, expand=True)

        self.blur_label = ctk.CTkLabel(adv_quality_frame, text="Порог размытости: 100.0")
        self.blur_label.pack(anchor="w", padx=10, pady=(10, 0))
        self.blur_slider = ctk.CTkSlider(adv_quality_frame, from_=10, to=300, command=self.update_adv_quality_labels)
        self.blur_slider.pack(fill="x", padx=10, pady=5, expand=True)

        # --- Блок "Настройки Проверки Нарушителя" ---
        intruder_frame = ctk.CTkFrame(scrollable_frame)
        intruder_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=10)
        intruder_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(intruder_frame, text="Настройки Проверки Нарушителя", font=("Arial", 16, "bold")).pack(anchor="w", padx=10, pady=5)

        self.retries_label = ctk.CTkLabel(intruder_frame, text="Кол-во повторных проверок: 3")
        self.retries_label.pack(anchor="w", padx=10, pady=(10, 0))
        self.retries_slider = ctk.CTkSlider(intruder_frame, from_=0, to=10, number_of_steps=11, command=self.update_intruder_labels)
        self.retries_slider.pack(fill="x", padx=10, pady=5, expand=True)

        self.delay_label = ctk.CTkLabel(intruder_frame, text="Задержка между проверками (сек): 2")
        self.delay_label.pack(anchor="w", padx=10, pady=(10, 0))
        self.delay_slider = ctk.CTkSlider(intruder_frame, from_=0, to=10, number_of_steps=11, command=self.update_intruder_labels)
        self.delay_slider.pack(fill="x", padx=10, pady=5, expand=True)
        
        # --- Блок "Уведомления Telegram" ---
        tg_frame = ctk.CTkFrame(scrollable_frame)
        tg_frame.grid(row=4, column=0, sticky="ew", padx=10, pady=10)
        tg_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(tg_frame, text="Настройки Уведомлений Telegram", font=("Arial", 16, "bold")).pack(anchor="w", padx=10, pady=5)
        
        ctk.CTkLabel(tg_frame, text="Токен Telegram-бота:").pack(anchor="w", padx=10)
        self.token_entry = ctk.CTkEntry(tg_frame)
        self.token_entry.pack(fill="x", padx=10, expand=True)
        self.token_entry.bind("<KeyRelease>", lambda e: self.save_config())
        
        ctk.CTkLabel(tg_frame, text="ID чата с ботом:").pack(anchor="w", padx=10, pady=(10,0))
        self.chat_id_entry = ctk.CTkEntry(tg_frame)
        self.chat_id_entry.pack(fill="x", padx=10, expand=True)
        self.chat_id_entry.bind("<KeyRelease>", lambda e: self.save_config())
        
        ctk.CTkButton(tg_frame, text="Проверить Telegram", command=self.test_telegram).pack(pady=10, padx=10)
        
        self.load_config()

    def load_config(self):
        """Загружает настройки из файла config.ini в UI, используя центральный менеджер."""
        self._loading_config = True # Устанавливаем флаг перед загрузкой
        self.config = get_config() # Получаем актуальный конфиг
        
        # --- FaceRecognition ---
        threshold = self.config.getfloat('FaceRecognition', 'threshold', fallback=0.5)
        self.threshold_slider.set(threshold)
        self.update_threshold(threshold)
        
        # --- QualityCheck (базовые) ---
        target_bright = self.config.getint('QualityCheck', 'target_brightness', fallback=120)
        tolerance = self.config.getint('QualityCheck', 'brightness_tolerance', fallback=30)
        self.brightness_target_slider.set(target_bright)
        self.brightness_tolerance_slider.set(tolerance)
        self.update_brightness()
        
        # --- QualityCheck (расширенные) ---
        det_score = self.config.getfloat('QualityCheck', 'min_det_score', fallback=0.9)
        face_size = self.config.getint('QualityCheck', 'min_face_size', fallback=60)
        blur = self.config.getfloat('QualityCheck', 'blur_threshold', fallback=100.0)
        self.det_score_slider.set(det_score)
        self.face_size_slider.set(face_size)
        self.blur_slider.set(blur)
        self.update_adv_quality_labels()

        # --- IntruderCheck ---
        retries = self.config.getint('IntruderCheck', 'retries', fallback=3)
        delay = self.config.getint('IntruderCheck', 'retry_delay_seconds', fallback=2)
        self.retries_slider.set(retries)
        self.delay_slider.set(delay)
        self.update_intruder_labels()

        # --- Telegram ---
        token = self.config.get('Telegram', 'bot_token', fallback='')
        self.token_entry.delete(0, 'end') # Очищаем поле перед вставкой
        self.token_entry.insert(0, token)
        
        chat_id = self.config.get('Telegram', 'chat_id', fallback='')
        self.chat_id_entry.delete(0, 'end') # Очищаем поле перед вставкой
        self.chat_id_entry.insert(0, chat_id)
        self._loading_config = False # Сбрасываем флаг после загрузки

    def save_config(self):
        """Сохраняет текущие настройки из UI в файл config.ini."""
        if self._loading_config: # Если мы в процессе загрузки, ничего не сохраняем
            return
        try:
            # --- FaceRecognition ---
            self.config.set('FaceRecognition', 'threshold', f"{self.threshold_slider.get():.2f}")
            
            # --- QualityCheck ---
            self.config.set('QualityCheck', 'target_brightness', f"{int(self.brightness_target_slider.get())}")
            self.config.set('QualityCheck', 'brightness_tolerance', f"{int(self.brightness_tolerance_slider.get())}")
            self.config.set('QualityCheck', 'min_det_score', f"{self.det_score_slider.get():.2f}")
            self.config.set('QualityCheck', 'min_face_size', f"{int(self.face_size_slider.get())}")
            self.config.set('QualityCheck', 'blur_threshold', f"{self.blur_slider.get():.1f}")

            # --- IntruderCheck ---
            self.config.set('IntruderCheck', 'retries', f"{int(self.retries_slider.get())}")
            self.config.set('IntruderCheck', 'retry_delay_seconds', f"{int(self.delay_slider.get())}")

            # --- Telegram ---
            self.config.set('Telegram', 'bot_token', self.token_entry.get())
            self.config.set('Telegram', 'chat_id', self.chat_id_entry.get())
            
            # Запись настроек в файл
            with open(CONFIG_FILE, 'w') as configfile:
                self.config.write(configfile)
            logging.info(f"Настройки сохранены в '{CONFIG_FILE}'.")
        except Exception as e:
            logging.error(f"Ошибка при сохранении конфигурации: {e}")
            messagebox.showerror("Ошибка сохранения", f"Не удалось сохранить настройки в файл '{CONFIG_FILE}'.\n\nОшибка: {e}")

    def update_threshold(self, value):
        """Обновляет метку порога схожести и сохраняет настройки."""
        self.threshold_label.configure(text=f"Чувствительность (порог): {value:.2f}")
        self.save_config()
    
    def update_brightness(self, _=None): # _=None для совместимости со слайдером, который передает значение
        """Обновляет метку диапазона яркости и сохраняет настройки."""
        target = int(self.brightness_target_slider.get())
        tolerance = int(self.brightness_tolerance_slider.get())
        min_val = max(0, target - tolerance)   # Нижняя граница не может быть меньше 0
        max_val = min(255, target + tolerance) # Верхняя граница не может быть больше 255
        self.brightness_label.configure(text=f"Допустимая яркость: {min_val} - {max_val}")
        self.save_config()

    def update_adv_quality_labels(self, _=None):
        """Обновляет метки расширенных настроек качества и сохраняет конфиг."""
        det_score = self.det_score_slider.get()
        self.det_score_label.configure(text=f"Мин. уверенность детектора: {det_score:.2f}")
        
        face_size = int(self.face_size_slider.get())
        self.face_size_label.configure(text=f"Мин. размер лица (px): {face_size}")

        blur = self.blur_slider.get()
        self.blur_label.configure(text=f"Порог размытости: {blur:.1f}")
        
        self.save_config()

    def update_intruder_labels(self, _=None):
        """Обновляет метки настроек проверки нарушителя и сохраняет конфиг."""
        retries = int(self.retries_slider.get())
        self.retries_label.configure(text=f"Кол-во повторных проверок: {retries}")

        delay = int(self.delay_slider.get())
        self.delay_label.configure(text=f"Задержка между проверками (сек): {delay}")

        self.save_config()

    def delete_enrollment(self):
        """Удаляет файл с эталонными эмбеддингами владельца."""
        if messagebox.askyesno("Подтверждение", "Вы уверены, что хотите удалить все зарегистрированные данные владельца? Это действие необратимо и потребует новой регистрации."):
            f = 'data/owner_embedding.npy'
            if os.path.exists(f):
                os.remove(f)
                messagebox.showinfo("Успех", "Регистрация владельца успешно удалена.")
                logging.info("Файл 'owner_embedding.npy' удален.")
            else:
                messagebox.showwarning("Внимание", "Файл регистрации владельца не найден. Возможно, регистрация еще не была произведена.")

    def test_telegram(self):
        """Отправляет тестовое сообщение Telegram для проверки настроек."""
        self.save_config() # Сначала сохраняем текущие настройки (на случай, если они были только что введены)
        messagebox.showinfo("Проверка Telegram", "Отправляю тестовое сообщение... Проверьте свой Telegram-чат.")
        send_test_message() # Вызываем функцию отправки тестового сообщения

    def on_closing(self):
        """Обработчик события закрытия окна."""
        if self.cap and self.cap.isOpened():
            self.cap.release() # Освобождаем ресурсы веб-камеры
        self.destroy() # Уничтожаем окно Tkinter

# --- Точка входа в приложение ---
if __name__ == "__main__":
    app = FaceGuardCenter()
    app.protocol("WM_DELETE_WINDOW", app.on_closing) # Привязываем обработчик закрытия окна
    app.mainloop() # Запускаем основной цикл событий Tkinter
