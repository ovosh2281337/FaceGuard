import telegram
from telegram import error
import asyncio
from config_manager import get_config # Импортируем наш новый менеджер
import logging

# --- Настройка базового логирования для этого модуля ---
# Это поможет отлаживать проблемы с отправкой, даже если основной скрипт не запущен.
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("guard.log", mode='a'),
                        logging.StreamHandler()
                    ])

# --------------------------------------------------------------------
# --- ОСНОВНАЯ ФУНКЦИЯ: ОТПРАВКА ТРЕВОЖНОГО УВЕДОМЛЕНИЯ ---
# --------------------------------------------------------------------

async def send_alert_async(bot_token, chat_id, image_path):
    """Асинхронная функция для отправки сообщения и фото."""
    try:
        bot = telegram.Bot(token=bot_token)
        caption_text = "🚨 **Внимание!** 🚨\n\nОбнаружен неопознанный пользователь за вашим ПК."
        
        # Отправляем фото с подписью, используя Markdown для форматирования
        await bot.send_photo(chat_id=chat_id, photo=open(image_path, 'rb'), caption=caption_text, parse_mode='Markdown')
        
        logging.info(f"Уведомление о нарушителе успешно отправлено в Telegram чат {chat_id}.")
    except error.BadRequest as e:
        # Частая ошибка - неправильный chat_id
        logging.error(f"Ошибка запроса Telegram (проверьте ID чата): {e}")
    except error.Unauthorized as e:
        # Частая ошибка - неправильный токен бота
        logging.error(f"Ошибка авторизации Telegram (проверьте токен бота): {e}")
    except Exception as e:
        logging.error(f"Не удалось отправить уведомление о нарушителе в Telegram: {e}")

def send_alert(image_path):
    """
    Синхронная "обертка" для удобного вызова из основного скрипта.
    Загружает конфиг и запускает асинхронную отправку тревоги.
    """
    config = get_config() # Используем централизованный менеджер
    bot_token = config.get('Telegram', 'bot_token', fallback='')
    chat_id = config.get('Telegram', 'chat_id', fallback='')
    
    if not bot_token or bot_token == "ВАШ_ТОКЕН_ОТ_BOTFATHER" or not chat_id:
        logging.error("Токен бота или ID чата не указаны в config.ini! Уведомление о нарушителе не будет отправлено.")
        return
        
    asyncio.run(send_alert_async(bot_token, chat_id, image_path))


# --------------------------------------------------------------------
# --- НОВАЯ ФУНКЦИЯ: ОТПРАВКА ТЕСТОВОГО СООБЩЕНИЯ ---
# --------------------------------------------------------------------

async def send_test_message_async(bot_token, chat_id):
    """Асинхронная функция для отправки простого текстового сообщения."""
    try:
        bot = telegram.Bot(token=bot_token)
        text = "✅ Тестовое сообщение от FaceGuard. Связь установлена!"
        await bot.send_message(chat_id=chat_id, text=text)
        logging.info(f"Тестовое сообщение успешно отправлено в Telegram чат {chat_id}.")
        return True
    except error.BadRequest:
        logging.error("Ошибка запроса при отправке тестового сообщения. Скорее всего, неверный ID чата.")
        return False
    except error.Unauthorized:
        logging.error("Ошибка авторизации при отправке тестового сообщения. Скорее всего, неверный токен бота.")
        return False
    except Exception as e:
        logging.error(f"Не удалось отправить тестовое сообщение в Telegram: {e}")
        return False

def send_test_message():
    """
    Синхронная "обертка" для удобного вызова из UI.
    Загружает конфиг и запускает асинхронную отправку теста.
    """
    config = get_config() # Используем централизованный менеджер
    bot_token = config.get('Telegram', 'bot_token', fallback='')
    chat_id = config.get('Telegram', 'chat_id', fallback='')
    
    if not bot_token or bot_token == "ВАШ_ТОКЕН_ОТ_BOTFATHER" or not chat_id:
        logging.error("Токен бота или ID чата не указаны в config.ini! Тестовое сообщение не будет отправлено.")
        # В UI мы уже показываем messagebox, здесь достаточно лога.
        return
        
    # Запускаем асинхронную функцию
    asyncio.run(send_test_message_async(bot_token, chat_id))
