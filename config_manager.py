import os
from configparser import ConfigParser
import logging

CONFIG_FILE = 'config.ini'

def create_default_config():
    """Создает ConfigParser с настройками по умолчанию."""
    config = ConfigParser()
    
    # Секция 1: Настройки распознавания лиц
    config.add_section('FaceRecognition')
    config.set('FaceRecognition', 'threshold', '0.5') # Порог схожести для идентификации

    # Секция 2: Настройки качества снимков и лиц
    config.add_section('QualityCheck')
    config.set('QualityCheck', 'target_brightness', '120')    # Целевая яркость (0-255)
    config.set('QualityCheck', 'brightness_tolerance', '30') # Допуск по яркости (+/-)
    config.set('QualityCheck', 'min_det_score', '0.9')       # Минимальная уверенность детектора
    config.set('QualityCheck', 'min_face_size', '60')        # Минимальный размер лица в пикселях
    config.set('QualityCheck', 'blur_threshold', '100.0')    # Порог размытости (чем выше, тем более размытые фото отсеиваются)

    # Секция 3: Настройки уведомлений в Telegram
    config.add_section('Telegram')
    config.set('Telegram', 'bot_token', '') # Оставить пустым!
    config.set('Telegram', 'chat_id', '')   # Оставить пустым!

    # Секция 4: Настройки повторных проверок при обнаружении нарушителя
    config.add_section('IntruderCheck')
    config.set('IntruderCheck', 'retries', '3') # Количество повторных проверок
    config.set('IntruderCheck', 'retry_delay_seconds', '2') # Задержка между проверками в секундах

    return config

def get_config():
    """
    Загружает конфигурацию из config.ini.
    Если файл не найден, создает его с настройками по умолчанию.
    """
    config = ConfigParser()
    if not os.path.exists(CONFIG_FILE):
        logging.warning(f"Файл '{CONFIG_FILE}' не найден. Создаю новый с настройками по умолчанию.")
        default_config = create_default_config()
        with open(CONFIG_FILE, 'w') as f:
            default_config.write(f)
        # После создания читаем его, чтобы вернуть объект ConfigParser
        config.read(CONFIG_FILE)
    else:
        config.read(CONFIG_FILE)
    return config

if __name__ == '__main__':
    # Этот блок выполнится, если запустить файл напрямую.
    # Полезно для первоначального создания файла или его сброса.
    print(f"Проверка файла конфигурации '{CONFIG_FILE}'...")
    cfg = get_config()
    print(f"Файл '{CONFIG_FILE}' готов к использованию.")
    # Выводим одну из настроек для проверки
    print(f"Текущий порог чувствительности: {cfg.getfloat('FaceRecognition', 'threshold')}")
