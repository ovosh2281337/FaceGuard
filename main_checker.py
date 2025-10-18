import cv2
import numpy as np
import insightface
import os
import time
import logging
from config_manager import get_config # Импортируем наш новый менеджер

# --- ВНИМАНИЕ: УБЕДИТЕСЬ, ЧТО telegram_notifier.py НАХОДИТСЯ В ТОЙ ЖЕ ПАПКЕ ---
from telegram_notifier import send_alert 

# --- Настройка логирования для сервиса ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("guard.log", mode='a', encoding='utf-8'), # Запись логов в файл
                        logging.StreamHandler() # Вывод логов в консоль
                    ])

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# --- КОНСТАНТЫ ДЛЯ АДАПТИВНОЙ БАЗЫ ЭТАЛОНОВ ---
# Порог схожести для добавления нового эталона в адаптивную базу (очень высокий!)
# Чем выше значение, тем меньше "ложных" адаптаций, но медленнее адаптация к изменениям.
ADAPTIVE_ADD_THRESHOLD = 0.75

# Максимальное количество адаптивных эталонов, которые будут храниться
# Если база превышает этот размер, самый старый эталон удаляется.
MAX_ADAPTIVE_EMBEDDINGS = 20   

# Путь к файлу, где будут храниться динамически добавляемые эталоны
ADAPTIVE_EMBEDDINGS_PATH = 'data/adaptive_embeddings.npy' 

# --- ОБНОВЛЕННАЯ ФУНКЦИЯ: КОМПЛЕКСНЫЙ АНАЛИЗ КАЧЕСТВА ЛИЦА ---
def is_face_quality_good(face, frame, config):
    """
    Выполняет каскадную проверку качества обнаруженного лица.
    Возвращает True, если лицо прошло все проверки, иначе False.
    """
    try:
        # --- Загрузка параметров из конфига ---
        min_det_score = config.getfloat('QualityCheck', 'min_det_score', fallback=0.9)
        min_face_size = config.getint('QualityCheck', 'min_face_size', fallback=60)
        blur_threshold = config.getfloat('QualityCheck', 'blur_threshold', fallback=100.0)
        target_brightness = config.getint('QualityCheck', 'target_brightness', fallback=120)
        brightness_tolerance = config.getint('QualityCheck', 'brightness_tolerance', fallback=50)
        bright_min = int(max(0, target_brightness - brightness_tolerance))
        bright_max = int(min(255, target_brightness + brightness_tolerance))

        # 1. Проверка уверенности детектора
        if face.det_score < min_det_score:
            logging.warning(f"Качество лица: ПЛОХОЕ. Уверенность детектора ({face.det_score:.2f}) ниже порога ({min_det_score}).")
            return False

        # 2. Проверка размера лица
        bbox = face.bbox.astype(int)
        face_width = bbox[2] - bbox[0]
        face_height = bbox[3] - bbox[1]
        if face_width < min_face_size or face_height < min_face_size:
            logging.warning(f"Качество лица: ПЛОХОЕ. Размер ({face_width}x{face_height}) меньше минимального ({min_face_size}x{min_face_size}).")
            return False

        # --- Извлечение ROI для дальнейших проверок ---
        x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), min(frame.shape[1], bbox[2]), min(frame.shape[0], bbox[3])
        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size == 0:
            logging.warning("ROI лица пуст после обрезки.")
            return False
        
        gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

        # 3. Проверка на размытость (дисперсия Лапласиана)
        laplacian_var = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
        if laplacian_var < blur_threshold:
            logging.warning(f"Качество лица: ПЛОХОЕ. Размытие ({laplacian_var:.2f}) выше порога ({blur_threshold}).")
            return False

        # 4. Проверка на яркость
        brightness = np.mean(gray_roi)
        if not (bright_min <= brightness <= bright_max):
            logging.warning(f"Качество лица: ПЛОХОЕ. Яркость ({brightness:.2f}) вне диапазона ({bright_min}-{bright_max}).")
            return False

        logging.info(f"Качество лица: ХОРОШЕЕ (Уверенность={face.det_score:.2f}, Размер={face_width}x{face_height}, Четкость={laplacian_var:.2f}, Яркость={brightness:.2f})")
        return True

    except Exception as e:
        logging.error(f"Ошибка в функции is_face_quality_good: {e}")
        return False


def perform_check(face_analyzer, all_owner_embeddings, config, cap):
    """
    Основная функция проверки. Захватывает кадр, анализирует лица и
    принимает решение о наличии владельца, а также обновляет адаптивную базу.
    """
    # --- Загрузка порогов из конфига ---
    similarity_threshold = config.getfloat('FaceRecognition', 'threshold')
    intruder_check_retries = config.getint('IntruderCheck', 'retries', fallback=3)
    intruder_check_delay = config.getint('IntruderCheck', 'retry_delay_seconds', fallback=2)
    
    logging.info("Выполняю проверку...")
    
    # --- БЛОК: ЗАХВАТ КАЧЕСТВЕННОГО КАДРА С НЕСКОЛЬКИМИ ПОПЫТКАМИ ---
    MAX_ATTEMPTS = 3 # Количество попыток захвата качественного кадра
    frame = None # Переменная для хранения успешного кадра
    
    for attempt in range(MAX_ATTEMPTS):
        logging.info(f"Попытка захвата кадра #{attempt + 1}...")
        time.sleep(1)  # Пауза между попытками для стабильности камеры
        ret, current_frame = cap.read() # Чтение кадра
        
        if not ret:
            logging.error("Не удалось получить кадр с веб-камеры.")
            continue # Пробуем следующую попытку
            
        # Сначала детектируем лицо, чтобы проверить качество именно ROI лица
        faces = face_analyzer.get(current_frame)
        if not faces:
            logging.warning("На кадре не найдено ни одного лица. Повторная попытка.")
            continue # Если нет лица, кадр не подходит, пробуем снова

        # Ищем хотя бы одно лицо, которое пройдет проверку качества
        good_face_found = False
        for face in faces:
            if is_face_quality_good(face, current_frame, config):
                frame = current_frame  # Сохраняем весь кадр, если найдено хотя бы одно качественное лицо
                good_face_found = True
                logging.info("Найдено качественное лицо. Захватываю этот кадр.")
                break  # Выходим из цикла по лицам, так как нашли подходящий кадр
        
        if good_face_found:
            break # Выходим из цикла попыток, так как нашли хороший кадр
        else:
            logging.warning("Ни одно из найденных лиц не прошло проверку качества. Повторная попытка.")
    
    if frame is None:
        logging.error(f"Не удалось получить кадр с качественным лицом за {MAX_ATTEMPTS} попытки. Проверка отменена.")
        return # Если не удалось получить качественный кадр, выходим
    # --- КОНЕЦ БЛОКА ЗАХВАТА ---

    # --- Анализ снимка и обновление адаптивной базы ---
    try:
        # Повторно детектируем лица на выбранном качественном кадре
        faces = face_analyzer.get(frame) 
        if not faces:
            logging.warning("На качественном снимке не найдено ни одного лица после захвата. Проверка отменена.")
            return

        logging.info(f"На качественном снимке обнаружено лиц: {len(faces)}. Начинаю проверку...")
        
        is_owner_present = False
        highest_similarity = 0.0 # Для отслеживания максимальной схожести
        best_owner_embedding = None # Эмбеддинг лица, давшего максимальную схожесть

        for i, detected_face in enumerate(faces):
            detected_embedding = detected_face.embedding
            logging.info(f"--- Проверка лица #{i+1} со снимка ---")
            
            is_this_face_owner = False
            
            # Сравниваем обнаруженное лицо со ВСЕМИ эталонами (базовыми + адаптивными)
            for j, owner_embedding in enumerate(all_owner_embeddings):
                # Cosine similarity - чем ближе к 1, тем выше схожесть
                similarity = np.dot(owner_embedding, detected_embedding) / (np.linalg.norm(owner_embedding) * np.linalg.norm(detected_embedding))
                logging.info(f"Сравнение с эталоном #{j+1}. Схожесть: {similarity:.4f}")
                
                # Обновляем максимальную схожесть и соответствующий эмбеддинг
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_owner_embedding = detected_embedding 

                if similarity >= similarity_threshold:
                    is_this_face_owner = True
                    logging.info(f"Найдено совпадение с эталоном #{j+1}. Это владелец.")
                    # Продолжаем внутренний цикл, чтобы найти *абсолютно* максимальную схожесть для обновления базы.
                    # Но если вам нужно СРАЗУ выйти после первого совпадения, можно добавить 'break' сюда.
                    # Для адаптивной базы лучше дойти до конца, чтобы найти *лучшее* совпадение.
            
            if is_this_face_owner:
                is_owner_present = True
                # Если владелец найден, можно выйти из цикла по лицам, т.к. цель достигнута.
                break 
                
        if is_owner_present:
            logging.info("Владелец присутствует в кадре. Все в порядке.")

            # --- ЛОГИКА АДАПТИВНОГО ОБНОВЛЕНИЯ БАЗЫ ---
            if highest_similarity >= ADAPTIVE_ADD_THRESHOLD and best_owner_embedding is not None:
                logging.info(f"Высокая схожесть ({highest_similarity:.4f}) обнаружена. Попытка обновить адаптивную базу.")
                
                # Загружаем текущие адаптивные эмбеддинги (или пустой список, если файла нет)
                current_adaptive = []
                if os.path.exists(ADAPTIVE_EMBEDDINGS_PATH):
                    try:
                        current_adaptive = list(np.load(ADAPTIVE_EMBEDDINGS_PATH, allow_pickle=True))
                    except Exception as e:
                        logging.warning(f"Ошибка чтения файла адаптивных эмбеддингов '{ADAPTIVE_EMBEDDINGS_PATH}': {e}. Начинаем с пустой адаптивной базы.")

                # Добавляем новый эмбеддинг в НАЧАЛО списка (самый свежий)
                current_adaptive.insert(0, best_owner_embedding)
                
                # Если количество эмбеддингов превышает лимит, удаляем самый старый (в конце списка)
                if len(current_adaptive) > MAX_ADAPTIVE_EMBEDDINGS:
                    current_adaptive = current_adaptive[:MAX_ADAPTIVE_EMBEDDINGS] # Обрезаем список
                    logging.info(f"Лимит адаптивных эталонов ({MAX_ADAPTIVE_EMBEDDINGS}) достигнут. Самый старый удален.")

                # Сохраняем обновленный список адаптивных эмбеддингов
                np.save(ADAPTIVE_EMBEDDINGS_PATH, np.array(current_adaptive))
                logging.info(f"Адаптивная база успешно обновлена. Текущий размер: {len(current_adaptive)} эталонов.")
            else:
                logging.info(f"Максимальная схожесть ({highest_similarity:.4f}) ниже порога для адаптации ({ADAPTIVE_ADD_THRESHOLD}). Адаптивная база не обновляется.")
            
        else:
            logging.warning("!!! ВЛАДЕЛЕЦ НЕ НАЙДЕН. ЗАПУСКАЮ РЕЖИМ ПОВТОРНОЙ ПРОВЕРКИ... !!!")
            
            # --- БЛОК ПОВТОРНЫХ ПРОВЕРОК ---
            owner_found_on_retry = False
            last_intruder_frame = frame # Сохраняем исходный кадр на случай, если владелец так и не появится
            
            for i in range(intruder_check_retries):
                logging.info(f"Повторная проверка #{i + 1}/{intruder_check_retries}...")
                time.sleep(intruder_check_delay)
                
                ret, retry_frame = cap.read()
                if not ret:
                    logging.error("Не удалось получить кадр при повторной проверке.")
                    continue

                last_intruder_frame = retry_frame # Обновляем последний кадр
                retry_faces = face_analyzer.get(retry_frame)
                
                if not retry_faces:
                    logging.warning("На кадре повторной проверки лиц не найдено.")
                    continue

                # Проверяем, есть ли владелец среди найденных лиц
                for retry_face in retry_faces:
                    for owner_embedding in all_owner_embeddings:
                        similarity = np.dot(owner_embedding, retry_face.embedding) / (np.linalg.norm(owner_embedding) * np.linalg.norm(retry_face.embedding))
                        if similarity >= similarity_threshold:
                            owner_found_on_retry = True
                            logging.info(f"Владелец ОБНАРУЖЕН при повторной проверке! Схожесть: {similarity:.4f}. Тревога отменена.")
                            break
                    if owner_found_on_retry:
                        break
                
                if owner_found_on_retry:
                    break # Выходим из цикла повторных проверок
            
            # --- ОТПРАВКА УВЕДОМЛЕНИЯ, ЕСЛИ ВЛАДЕЛЕЦ ТАК И НЕ НАЙДЕН ---
            if not owner_found_on_retry:
                logging.warning("!!! ВЛАДЕЛЕЦ НЕ БЫЛ НАЙДЕН ДАЖЕ ПОСЛЕ ПОВТОРНЫХ ПРОВЕРОК. ОТПРАВКА УВЕДОМЛЕНИЯ. !!!")
                if not os.path.exists('data/intruders'): 
                    os.makedirs('data/intruders')
                
                intruder_filename = f"intruder_{time.strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
                intruder_filepath = os.path.join('data/intruders', intruder_filename)
                
                cv2.imwrite(intruder_filepath, last_intruder_frame)
                logging.info(f"Фото с неопознанными лицами сохранено в: {intruder_filepath}")
                
                send_alert(intruder_filepath)
            # --- КОНЕЦ БЛОКА ПОВТОРНЫХ ПРОВЕРОК ---

    except Exception as e:
        logging.error(f"Произошла ошибка во время анализа лица или отправки уведомления: {e}")


if __name__ == "__main__":
    logging.info("--- FaceGuard СЕРВИС ЗАПУЩЕН ---")
    
    try:
        config = get_config() # Используем централизованный менеджер
        
        # --- ОБНОВЛЕННАЯ ЛОГИКА ЗАГРУЗКИ ЭТАЛОНОВ ---
        # 1. Загружаем базовые ("золотые") эталоны. Их отсутствие - критическая ошибка.
        base_owner_embedding_path = 'data/owner_embedding.npy'
        if not os.path.exists(base_owner_embedding_path):
            raise FileNotFoundError(f"Файл '{base_owner_embedding_path}' не найден! Запустите Центр Управления для регистрации.")
        
        base_embeddings = list(np.load(base_owner_embedding_path))
        logging.info(f"Загружено {len(base_embeddings)} базовых эталонных эмбеддингов.")

        # 2. Загружаем адаптивные эталоны. Их отсутствие - не ошибка, просто начинаем с пустой базы.
        adaptive_embeddings = []
        if os.path.exists(ADAPTIVE_EMBEDDINGS_PATH):
            try:
                # allow_pickle=True может потребоваться для старых версий numpy или специфических данных
                adaptive_embeddings = list(np.load(ADAPTIVE_EMBEDDINGS_PATH, allow_pickle=True))
                logging.info(f"Загружено {len(adaptive_embeddings)} адаптивных эталонных эмбеддингов.")
            except Exception as e:
                logging.warning(f"Не удалось загрузить файл адаптивных эталонов '{ADAPTIVE_EMBEDDINGS_PATH}': {e}. Начинаем с пустой базы.")
        
        # 3. Объединяем все эталоны в один список для проверок
        all_owner_embeddings = base_embeddings + adaptive_embeddings
        logging.info(f"Всего для проверки используется {len(all_owner_embeddings)} эталонов.")
        
        # Загрузка модели InsightFace
        face_analyzer = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        face_analyzer.prepare(ctx_id=-1, det_size=(640, 640))
        
        logging.info("Все модели и конфиги успешно загружены в память.")
        
    except Exception as e:
        logging.error(f"Критическая ошибка при инициализации сервиса: {e}")
        exit() # Выходим из приложения при критической ошибке инициализации

    # Изначальная задержка перед первой проверкой
    wait_time_seconds = 10
    logging.info(f"Сервис ждет {wait_time_seconds} секунд перед первой проверкой...")
    time.sleep(wait_time_seconds)
    
    # --- Выполнение проверки (одной итерации). Для постоянной работы, обернуть в while True ---
    camera_id = config.getint('Camera', 'device_id', fallback=0)
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        logging.error(f"Критическая ошибка: не удалось открыть веб-камеру с ID {camera_id}.")
        logging.info("Попытка использовать камеру по умолчанию (ID 0)...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logging.error("Критическая ошибка: не удалось открыть и камеру по умолчанию.")
    else:
        perform_check(face_analyzer, all_owner_embeddings, config, cap)
        cap.release() # Освобождаем камеру здесь, в самом конце
    
    logging.info("--- FaceGuard СЕРВИС ЗАВЕРШИЛ РАБОТУ ---")

    # Синхронная "обертка" для удобного вызова из UI. (Удалено, так как это отдельный сервис)
