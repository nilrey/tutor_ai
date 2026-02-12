## Создание проекта
### Install Python 3.10.11
    под Windows:

    D:/Opt/python/python310/python -m venv venv
    
### Запуск вирт. окружения
    под Windows:
    
    venv/Scripts/activate

### Установка зависимостей
    pip install -r requirements.txt
    

### Запуск проекта
    Запустить файл main.py (важно, что запуск скриптов осущ. из папки venv/Scripts/python)
    Будет поднят uvicorn и доступен Swagger UI для тестирования API.


## Результат
### Интерфейс висит на [http://localhost:8000/docs](http://localhost:8000/docs)
    Загрузить файлы PDF в базу через роут /upload_file

## Окончание работы скрипта
    Выйти Ctrl+C