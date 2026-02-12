## Создание проекта
### Install Python 3.10.11
    D:/Opt/python/python310/python -m venv venv

### Установка зависимостей
    pip install -r requirements.txt
    
### Запуск вирт. окружения
    venv/Scripts/activate
    

### Запуск проекта
    Запустить файл main.py (Проследить что запуск скриптов осущ. из папки venv/Scripts/python)
    Будет поднят uvicorn и доступен Swagger UI для тестирования API.


## Результат
### Интерфейс висит на [http://localhost:8000/docs](http://localhost:8000/docs)
    Загрузить файлы PDF в базу через роут /upload_file

## Окончание работы скрипта
    Выйти Ctrl+C