# enrollment_ui.py
import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
import numpy as np
import insightface
import os
import logging
from tkinter import messagebox
from config_manager import get_config, CONFIG_FILE
import threading
import queue
import time
from pygrabber.dshow_graph import FilterGraph

# --- Логирование ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])

# --- Заглушка для telegram_notifier (если модуль отсутствует) ---
try:
    from telegram_notifier import send_test_message
except ImportError:
    def send_test_message():
        messagebox.showerror("Ошибка", "Не удалось импортировать telegram_notifier.")

NUM_SNAPSHOTS = 3
PROMPTS = ["Посмотрите прямо в камеру...", "Теперь немного поверните голову влево", "Супер! Теперь немного вправо"]


def is_frame_quality_good(frame_roi, bright_min, bright_max):
    if frame_roi is None or frame_roi.size == 0:
        return False
    gray_roi = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
    return bright_min <= np.mean(gray_roi) <= bright_max


class FaceGuardCenter(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Центр Управления FaceGuard")
        self.geometry("1100x720")
        self.minsize(800, 600)
        ctk.set_appearance_mode("dark")

        # --- Состояния ---
        self.embeddings = []
        self.snapshot_count = 0
        self._loading_config = False
        self.is_running = True
        self.face_analyzer = None
        self.selected_camera_id = 0

        # --- Видео пайплайн ---
        # latest_frame — ссылка на последний захваченный кадр (BGR numpy)
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        # Очередь для готовых PIL.Image, размер 1 — храним только самый свежий кадр
        self.display_queue = queue.Queue(maxsize=1)
        self._photo_image_ref = None  # чтобы Tk не собирал PhotoImage
        self.last_label_size = (0, 0)
        self.is_resizing = False
        self.resize_job = None
        self.cover_mode = False  # False = contain (вписать полностью), True = cover (заполнить + crop)

        # Конфиг
        self.config = get_config()

        # --- Сетка: можно подправить веса для желаемых пропорций ---
        # По умолчанию делаем настройки чуть шире
        self.grid_columnconfigure(0, weight=2, minsize=400)  # Видео
        self.grid_columnconfigure(1, weight=3, minsize=350)  # Настройки
        self.grid_rowconfigure(0, weight=1)

        # --- Фреймы ---
        enroll_frame = ctk.CTkFrame(self)
        enroll_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.create_enrollment_widgets(enroll_frame)

        settings_frame = ctk.CTkFrame(self)
        settings_frame.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="nsew")
        self.create_settings_widgets(settings_frame)

        # --- Асинхронная загрузка модели и запуск потоков ---
        threading.Thread(target=self.load_model_async, daemon=True).start()

        # Загружаем конфиг (и запускаем камеру внизу)
        self.load_config()

        # Запускаем цикл отображения (GUI-поток)
        self.update_video_display()

        # Debounce resize
        self.bind("<Configure>", self.on_window_resize)

    # ------------------------- Модель -------------------------
    def load_model_async(self):
        logging.info("Асинхронная загрузка модели insightface...")
        try:
            analyzer = insightface.app.FaceAnalysis(name='buffalo_l',
                                                    providers=['CPUExecutionProvider'])
            analyzer.prepare(ctx_id=-1, det_size=(640, 640))
            self.face_analyzer = analyzer
            logging.info("Модель загружена.")
        except Exception as e:
            logging.error(f"Ошибка загрузки модели: {e}")
            self.after(0, lambda: messagebox.showerror("Критическая ошибка",
                                                        f"Не удалось загрузить модель: {e}"))

    # ------------------------- Виджеты для записи/видео -------------------------
    def create_enrollment_widgets(self, parent):
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        self.video_label = ctk.CTkLabel(parent, text="Загрузка модели...")
        self.video_label.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        controls = ctk.CTkFrame(parent, fg_color="transparent")
        controls.grid(row=1, column=0, pady=10, sticky="ew")
        self.count_label = ctk.CTkLabel(controls, text=f"Сделано снимков: 0/{NUM_SNAPSHOTS}",
                                        font=("Arial", 16))
        self.count_label.pack(pady=5)
        self.prompt_label = ctk.CTkLabel(controls, text=PROMPTS[0], font=("Arial", 18))
        self.prompt_label.pack(pady=10)
        self.snapshot_button = ctk.CTkButton(controls, text="Сделать снимок",
                                             command=self.take_snapshot, font=("Arial", 20),
                                             height=50)
        self.snapshot_button.pack(pady=10)

    # ------------------------- Потоки видео: захват и обработка -------------------------
    def video_processing_thread(self):
        """Быстрый поток захвата: кладём последний frame в self.latest_frame"""
        cap = cv2.VideoCapture(self.selected_camera_id, cv2.CAP_DSHOW)
        if not cap.isOpened():
            logging.error(f"Не удалось открыть камеру ID: {self.selected_camera_id}. Попытка с ID 0.")
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                logging.error("Не удалось открыть камеру по умолчанию.")
                with self.frame_lock:
                    self.latest_frame = "ERROR"
                return
            self.selected_camera_id = 0

        # Попробуем снизить latency; команда может быть проигнорирована драйвером
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        # Настройка разрешения (можно изменить в настройках)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            # сохраняем копию кадра под защитой lock
            with self.frame_lock:
                self.latest_frame = frame.copy()
            # Не спим здесь слишком долго — захват сам регулируется драйвером
        cap.release()

    def frame_processing_thread(self):
        """Фоновая обработка: ресайз + BGR->RGB -> PIL.Image; кладём в очередь"""
        while self.is_running:
            # Берём последний кадр
            with self.frame_lock:
                frame = None if self.latest_frame is None else (self.latest_frame.copy() if not isinstance(self.latest_frame, str) else self.latest_frame)

            if frame is None or isinstance(frame, str):
                time.sleep(0.03)
                continue

            label_w, label_h = self.last_label_size
            if label_w <= 1 or label_h <= 1:
                time.sleep(0.05)
                continue

            fh, fw = frame.shape[:2]

            # Выбираем режим масштабирования: cover vs contain
            if self.cover_mode:
                scale = max(label_w / fw, label_h / fh)
            else:
                scale = min(label_w / fw, label_h / fh)

            new_w = max(1, int(fw * scale))
            new_h = max(1, int(fh * scale))
            interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
            resized = cv2.resize(frame, (new_w, new_h), interpolation=interp)

            if self.cover_mode:
                # center crop до размеров label
                x = max(0, (new_w - label_w) // 2)
                y = max(0, (new_h - label_h) // 2)
                final = resized[y:y + label_h, x:x + label_w]
            else:
                final = resized

            # BGR -> RGB -> PIL
            rgb = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            # Кладём в очередь, если полна — заменяем старый кадр
            try:
                self.display_queue.put_nowait(pil_img)
            except queue.Full:
                try:
                    _ = self.display_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.display_queue.put_nowait(pil_img)
                except Exception:
                    pass

            # Ограничиваем обработку до ~30 FPS
            time.sleep(0.033)

    # ------------------------- GUI: обновление видео -------------------------
    def update_video_display(self):
        """Запускается в GUI-потоке. Берёт PIL.Image из очереди, создаёт PhotoImage и показывает."""
        # Обновляем cached size
        lw = self.video_label.winfo_width()
        lh = self.video_label.winfo_height()
        if lw > 1 and lh > 1:
            self.last_label_size = (lw, lh)

        # Если идёт resize/move — обновляем медленнее, но не блокируем GUI
        if self.is_resizing:
            try:
                img = self.display_queue.get_nowait()
            except queue.Empty:
                img = None
            if img is not None:
                photo = ImageTk.PhotoImage(img)
                self.video_label.configure(image=photo, text="")
                self._photo_image_ref = photo
            # медленнее — чтобы интерфейс оставался отзывчивым при перетаскивании окна
            self.after(66, self.update_video_display)
            return

        # Обычный режим — частота ~30 FPS gui update
        try:
            img = self.display_queue.get_nowait()
        except queue.Empty:
            img = None

        if img is not None:
            # создаём PhotoImage В ГЛАВНОМ ПОТОКЕ (важно!)
            photo = ImageTk.PhotoImage(img)
            self.video_label.configure(image=photo, text="")
            self._photo_image_ref = photo

        # Планируем следующее обновление
        self.after(33, self.update_video_display)

    # ------------------------- Resize debounce -------------------------
    def on_window_resize(self, event=None):
        # вызывается множество раз при перетаскивании/изменении - ставим флаг и дебаунсим
        self.is_resizing = True
        if self.resize_job:
            self.after_cancel(self.resize_job)
        self.resize_job = self.after(200, self._end_resize)

    def _end_resize(self):
        self.is_resizing = False
        self.resize_job = None

    # ------------------------- Снимок -------------------------
    def take_snapshot(self):
        if self.face_analyzer is None:
            self.prompt_label.configure(text="Модель еще загружается...")
            return

        with self.frame_lock:
            frame_to_process = None if self.latest_frame is None else (self.latest_frame.copy() if not isinstance(self.latest_frame, str) else self.latest_frame)

        if frame_to_process is None or (isinstance(frame_to_process, str) and frame_to_process == "ERROR"):
            self.prompt_label.configure(text="Ошибка камеры!")
            return

        try:
            faces = self.face_analyzer.get(frame_to_process)
        except Exception as e:
            logging.error(f"Ошибка анализа лица: {e}")
            faces = []

        if not faces:
            self.prompt_label.configure(text="Лицо не найдено!")
            return

        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        bbox = face.bbox.astype(int)
        roi = frame_to_process[max(0, bbox[1]):min(frame_to_process.shape[0], bbox[3]),
                               max(0, bbox[0]):min(frame_to_process.shape[1], bbox[2])]

        target, tol = self.brightness_target_slider.get(), self.brightness_tolerance_slider.get()
        if not is_frame_quality_good(roi, int(max(0, target - tol)), int(min(255, target + tol))):
            self.prompt_label.configure(text="Плохое качество кадра!")
            return

        self.embeddings.append(face.embedding)
        self.snapshot_count += 1
        self.count_label.configure(text=f"Сделано снимков: {self.snapshot_count}/{NUM_SNAPSHOTS}")

        if self.snapshot_count >= NUM_SNAPSHOTS:
            self.finalize_enrollment()
        else:
            self.prompt_label.configure(text=PROMPTS[self.snapshot_count])

    def finalize_enrollment(self):
        self.prompt_label.configure(text="Сохранение...")
        self.snapshot_button.configure(state="disabled")
        if not os.path.exists('data'):
            os.makedirs('data')
        np.save('data/owner_embedding.npy', np.array(self.embeddings))
        self.prompt_label.configure(text="Готово! Ваши данные сохранены.")
        self.snapshot_button.configure(state="normal", text="Начать заново",
                                        command=self.reset_enrollment_state)

    def reset_enrollment_state(self):
        self.snapshot_count = 0
        self.embeddings = []
        self.count_label.configure(text=f"Сделано снимков: 0/{NUM_SNAPSHOTS}")
        self.prompt_label.configure(text=PROMPTS[0])
        self.snapshot_button.configure(command=self.take_snapshot, text="Сделать снимок")

    # ------------------------- Настройки UI -------------------------
    def create_settings_widgets(self, parent):
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)
        scroll = ctk.CTkScrollableFrame(parent, label_text="Настройки")
        scroll.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        scroll.grid_columnconfigure(0, weight=1)

        # блоки настроек
        self.create_camera_settings(scroll)
        self.create_recognition_settings(scroll)
        self.create_quality_settings(scroll)
        self.create_adv_quality_settings(scroll)
        self.create_intruder_settings(scroll)
        self.create_telegram_settings(scroll)

        # Доп. переключатель: cover / contain
        extra = ctk.CTkFrame(scroll)
        extra.grid(row=6, column=0, sticky="ew", padx=10, pady=10)
        ctk.CTkLabel(extra, text="Поведение видео:", font=("Arial", 12, "bold")).pack(anchor="w", padx=10, pady=(5, 0))
        self.cover_switch = ctk.CTkSwitch(extra, text="Заполнить (crop)  /  Вписать (contain)",
                                          command=self.toggle_cover_mode)
        self.cover_switch.pack(anchor="w", padx=10, pady=5)

    def toggle_cover_mode(self):
        self.cover_mode = bool(self.cover_switch.get())

    def create_camera_settings(self, parent):
        frame = ctk.CTkFrame(parent)
        frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(frame, text="Настройки Камеры", font=("Arial", 16, "bold")).pack(anchor="w", padx=10, pady=5)
        self.camera_combobox = ctk.CTkComboBox(frame, values=["Сканирование..."],
                                                command=self.on_camera_select)
        self.camera_combobox.pack(fill="x", padx=10, pady=5, expand=True)
        ctk.CTkButton(frame, text="Обновить список камер",
                      command=self.scan_available_cameras).pack(pady=10, padx=10)

    def create_recognition_settings(self, parent):
        frame = ctk.CTkFrame(parent)
        frame.grid(row=1, column=0, sticky="ew", padx=10, pady=10)
        frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(frame, text="Настройки Распознавания", font=("Arial", 16, "bold")).pack(anchor="w", padx=10, pady=5)
        self.threshold_label = ctk.CTkLabel(frame, text="Чувствительность (порог): 0.50")
        self.threshold_label.pack(anchor="w", padx=10, pady=(10, 0))
        self.threshold_slider = ctk.CTkSlider(frame, from_=0.1, to=1.0,
                                              command=self.update_threshold)
        self.threshold_slider.pack(fill="x", padx=10, pady=5, expand=True)
        ctk.CTkButton(frame, text="Удалить регистрацию", command=self.delete_enrollment,
                      fg_color="#D2042D", hover_color="#AA0022").pack(pady=10, padx=10)

    def create_quality_settings(self, parent):
        frame = ctk.CTkFrame(parent)
        frame.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(frame, text="Настройки Качества Снимка", font=("Arial", 16, "bold")).pack(anchor="w", padx=10, pady=5)
        self.brightness_label = ctk.CTkLabel(frame, text="Допустимая яркость: 90 - 150")
        self.brightness_label.pack(anchor="w", padx=10, pady=(10, 0))
        ctk.CTkLabel(frame, text="Целевая яркость (0-255):").pack(anchor="w", padx=10, pady=(5,0))
        self.brightness_target_slider = ctk.CTkSlider(frame, from_=0, to=255,
                                                      number_of_steps=256,
                                                      command=self.update_brightness)
        self.brightness_target_slider.pack(fill="x", padx=10, pady=(0,5), expand=True)
        ctk.CTkLabel(frame, text="Допуск (+/-):").pack(anchor="w", padx=10, pady=(5,0))
        self.brightness_tolerance_slider = ctk.CTkSlider(frame, from_=0, to=100,
                                                         number_of_steps=101,
                                                         command=self.update_brightness)
        self.brightness_tolerance_slider.pack(fill="x", padx=10, pady=(0,5), expand=True)

    def create_adv_quality_settings(self, parent):
        frame = ctk.CTkFrame(parent)
        frame.grid(row=3, column=0, sticky="ew", padx=10, pady=10)
        frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(frame, text="Расширенные Настройки Качества", font=("Arial", 16, "bold")).pack(anchor="w", padx=10, pady=5)
        self.det_score_label = ctk.CTkLabel(frame, text="Мин. уверенность детектора: 0.90")
        self.det_score_label.pack(anchor="w", padx=10, pady=(10, 0))
        self.det_score_slider = ctk.CTkSlider(frame, from_=0.1, to=1.0,
                                              command=self.update_adv_quality_labels)
        self.det_score_slider.pack(fill="x", padx=10, pady=5, expand=True)
        self.face_size_label = ctk.CTkLabel(frame, text="Мин. размер лица (px): 60")
        self.face_size_label.pack(anchor="w", padx=10, pady=(10, 0))
        self.face_size_slider = ctk.CTkSlider(frame, from_=30, to=200,
                                              number_of_steps=171,
                                              command=self.update_adv_quality_labels)
        self.face_size_slider.pack(fill="x", padx=10, pady=5, expand=True)
        self.blur_label = ctk.CTkLabel(frame, text="Порог размытости: 100.0")
        self.blur_label.pack(anchor="w", padx=10, pady=(10, 0))
        self.blur_slider = ctk.CTkSlider(frame, from_=10, to=300,
                                         command=self.update_adv_quality_labels)
        self.blur_slider.pack(fill="x", padx=10, pady=5, expand=True)

    def create_intruder_settings(self, parent):
        frame = ctk.CTkFrame(parent)
        frame.grid(row=4, column=0, sticky="ew", padx=10, pady=10)
        frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(frame, text="Настройки Проверки Нарушителя", font=("Arial", 16, "bold")).pack(anchor="w", padx=10, pady=5)
        self.retries_label = ctk.CTkLabel(frame, text="Кол-во повторных проверок: 3")
        self.retries_label.pack(anchor="w", padx=10, pady=(10, 0))
        self.retries_slider = ctk.CTkSlider(frame, from_=0, to=10,
                                            number_of_steps=11,
                                            command=self.update_intruder_labels)
        self.retries_slider.pack(fill="x", padx=10, pady=5, expand=True)
        self.delay_label = ctk.CTkLabel(frame, text="Задержка между проверками (сек): 2")
        self.delay_label.pack(anchor="w", padx=10, pady=(10, 0))
        self.delay_slider = ctk.CTkSlider(frame, from_=0, to=10,
                                          number_of_steps=11,
                                          command=self.update_intruder_labels)
        self.delay_slider.pack(fill="x", padx=10, pady=5, expand=True)

    def create_telegram_settings(self, parent):
        frame = ctk.CTkFrame(parent)
        frame.grid(row=5, column=0, sticky="ew", padx=10, pady=10)
        frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(frame, text="Настройки Уведомлений Telegram", font=("Arial", 16, "bold")).pack(anchor="w", padx=10, pady=5)
        ctk.CTkLabel(frame, text="Токен Telegram-бота:").pack(anchor="w", padx=10)
        self.token_entry = ctk.CTkEntry(frame)
        self.token_entry.pack(fill="x", padx=10, expand=True)
        self.token_entry.bind("<KeyRelease>", lambda e: self.save_config())
        ctk.CTkLabel(frame, text="ID чата с ботом:").pack(anchor="w", padx=10, pady=(10,0))
        self.chat_id_entry = ctk.CTkEntry(frame)
        self.chat_id_entry.pack(fill="x", padx=10, expand=True)
        self.chat_id_entry.bind("<KeyRelease>", lambda e: self.save_config())
        ctk.CTkButton(frame, text="Проверить Telegram",
                      command=self.test_telegram).pack(pady=10, padx=10)

    # ------------------------- Конфиг и камера -------------------------
    def load_config(self):
        self._loading_config = True
        self.config = get_config()
        self.selected_camera_id = self.config.getint('Camera', 'device_id', fallback=0)
        self.scan_available_cameras()

        try:
            self.threshold_slider.set(self.config.getfloat('FaceRecognition', 'threshold', fallback=0.5))
        except Exception:
            pass
        self.update_threshold(self.threshold_slider.get() if hasattr(self, 'threshold_slider') else 0.5)

        try:
            self.brightness_target_slider.set(self.config.getint('QualityCheck', 'target_brightness', fallback=120))
            self.brightness_tolerance_slider.set(self.config.getint('QualityCheck', 'brightness_tolerance', fallback=30))
        except Exception:
            pass
        self.update_brightness()

        try:
            self.det_score_slider.set(self.config.getfloat('QualityCheck', 'min_det_score', fallback=0.9))
            self.face_size_slider.set(self.config.getint('QualityCheck', 'min_face_size', fallback=60))
            self.blur_slider.set(self.config.getfloat('QualityCheck', 'blur_threshold', fallback=100.0))
        except Exception:
            pass
        self.update_adv_quality_labels()

        try:
            self.retries_slider.set(self.config.getint('IntruderCheck', 'retries', fallback=3))
            self.delay_slider.set(self.config.getint('IntruderCheck', 'retry_delay_seconds', fallback=2))
        except Exception:
            pass
        self.update_intruder_labels()

        try:
            self.token_entry.insert(0, self.config.get('Telegram', 'bot_token', fallback=''))
            self.chat_id_entry.insert(0, self.config.get('Telegram', 'chat_id', fallback=''))
        except Exception:
            pass

        self._loading_config = False

        # Запускаем потоки камеры и обработки
        self.initialize_camera()

    def save_config(self):
        if self._loading_config:
            return
        try:
            # --- Добавлена проверка на наличие секции ---
            if not self.config.has_section('Camera'):
                self.config.add_section('Camera')
            self.config.set('Camera', 'device_id', str(self.selected_camera_id))
            if hasattr(self, 'threshold_slider'):
                self.config.set('FaceRecognition', 'threshold', f"{self.threshold_slider.get():.2f}")
            if hasattr(self, 'brightness_target_slider'):
                self.config.set('QualityCheck', 'target_brightness', f"{int(self.brightness_target_slider.get())}")
            if hasattr(self, 'brightness_tolerance_slider'):
                self.config.set('QualityCheck', 'brightness_tolerance', f"{int(self.brightness_tolerance_slider.get())}")
            if hasattr(self, 'det_score_slider'):
                self.config.set('QualityCheck', 'min_det_score', f"{self.det_score_slider.get():.2f}")
            if hasattr(self, 'face_size_slider'):
                self.config.set('QualityCheck', 'min_face_size', f"{int(self.face_size_slider.get())}")
            if hasattr(self, 'blur_slider'):
                self.config.set('QualityCheck', 'blur_threshold', f"{self.blur_slider.get():.1f}")
            if hasattr(self, 'retries_slider'):
                self.config.set('IntruderCheck', 'retries', f"{int(self.retries_slider.get())}")
            if hasattr(self, 'delay_slider'):
                self.config.set('IntruderCheck', 'retry_delay_seconds', f"{int(self.delay_slider.get())}")
            self.config.set('Telegram', 'bot_token', self.token_entry.get())
            self.config.set('Telegram', 'chat_id', self.chat_id_entry.get())
            with open(CONFIG_FILE, 'w') as f:
                self.config.write(f)
        except Exception as e:
            logging.error(f"Ошибка сохранения конфигурации: {e}")

    def scan_available_cameras(self):
        self.camera_combobox.configure(values=["Сканирование..."])
        try:
            graph = FilterGraph()
            devices = graph.get_input_devices()
            self.camera_names = {i: name for i, name in enumerate(devices)}

            if not self.camera_names:
                self.camera_combobox.configure(values=["Камеры не найдены"])
                return

            self.camera_combobox.configure(values=list(self.camera_names.values()))
            current_name = self.camera_names.get(self.selected_camera_id)
            if current_name:
                self.camera_combobox.set(current_name)
            else:
                first_id = list(self.camera_names.keys())[0]
                self.camera_combobox.set(self.camera_names[first_id])
                self.selected_camera_id = first_id
                self.save_config()
        except Exception as e:
            logging.error(f"Ошибка сканирования камер: {e}")
            self.camera_combobox.configure(values=["Ошибка сканирования"])

    def on_camera_select(self, choice):
        for id, name in self.camera_names.items():
            if name == choice and id != self.selected_camera_id:
                self.selected_camera_id = id
                self.save_config()
                self.initialize_camera()
                break

    def initialize_camera(self):
        # Останавливаем предыдущее, если было
        try:
            if hasattr(self, 'camera_thread') and self.camera_thread.is_alive():
                self.is_running = False
                self.camera_thread.join(timeout=1.0)
            if hasattr(self, 'processor_thread') and self.processor_thread.is_alive():
                self.processor_thread.join(timeout=1.0)
        except Exception:
            pass

        # Включаем заново
        self.is_running = True
        self.camera_thread = threading.Thread(target=self.video_processing_thread, daemon=True)
        self.processor_thread = threading.Thread(target=self.frame_processing_thread, daemon=True)
        self.camera_thread.start()
        self.processor_thread.start()

    # ------------------------- Методы обновления меток -------------------------
    def update_threshold(self, v):
        if hasattr(self, 'threshold_label'):
            self.threshold_label.configure(text=f"Чувствительность (порог): {float(v):.2f}")
        self.save_config()

    def update_brightness(self, _=None):
        if hasattr(self, 'brightness_target_slider') and hasattr(self, 'brightness_tolerance_slider') and hasattr(self, 'brightness_label'):
            t = int(self.brightness_target_slider.get())
            tol = int(self.brightness_tolerance_slider.get())
            self.brightness_label.configure(text=f"Допустимая яркость: {max(0, t - tol)} - {min(255, t + tol)}")
        self.save_config()

    def update_adv_quality_labels(self, _=None):
        if hasattr(self, 'det_score_label') and hasattr(self, 'det_score_slider'):
            self.det_score_label.configure(text=f"Мин. уверенность детектора: {self.det_score_slider.get():.2f}")
        if hasattr(self, 'face_size_label') and hasattr(self, 'face_size_slider'):
            self.face_size_label.configure(text=f"Мин. размер лица (px): {int(self.face_size_slider.get())}")
        if hasattr(self, 'blur_label') and hasattr(self, 'blur_slider'):
            self.blur_label.configure(text=f"Порог размытости: {self.blur_slider.get():.1f}")
        self.save_config()

    def update_intruder_labels(self, _=None):
        if hasattr(self, 'retries_label') and hasattr(self, 'retries_slider'):
            self.retries_label.configure(text=f"Кол-во повторных проверок: {int(self.retries_slider.get())}")
        if hasattr(self, 'delay_label') and hasattr(self, 'delay_slider'):
            self.delay_label.configure(text=f"Задержка между проверками (сек): {int(self.delay_slider.get())}")
        self.save_config()

    # ------------------------- Удаление, Telegram, закрытие -------------------------
    def delete_enrollment(self):
        if messagebox.askyesno("Подтверждение", "Вы уверены?"):
            if os.path.exists('data/owner_embedding.npy'):
                os.remove('data/owner_embedding.npy')
                messagebox.showinfo("Успех", "Регистрация удалена.")
            else:
                messagebox.showwarning("Внимание", "Файл регистрации не найден.")

    def test_telegram(self):
        self.save_config()
        messagebox.showinfo("Проверка", "Отправляю тестовое сообщение...")
        send_test_message()

    def on_closing(self):
        # корректно завершаем потоки
        self.is_running = False
        try:
            if hasattr(self, 'camera_thread'):
                self.camera_thread.join(timeout=1.0)
            if hasattr(self, 'processor_thread'):
                self.processor_thread.join(timeout=1.0)
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    app = FaceGuardCenter()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
