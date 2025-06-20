# SiameseNet Flask Проєкт

## Опис

Цей проєкт реалізує веб-сервіс на Flask для порівняння зображень за допомогою Siamese Neural Network.  
Модель базується на ResNet50, що вбудовується у кастомну мережу для отримання ембеддингів зображень, після чого порівнює їх схожість.

---

## Можливості

- Завантаження зображень через веб-інтерфейс.
- Генерація ембеддингів за допомогою натренованої моделі.
- Обчислення подібності між двома зображеннями.
- Збереження і сортування зображень у різні категорії (Positive, Negative, CUSTOM).
- Використання pre-trained ResNet50 як базової мережі.

---

## Встановлення

1. Клонувати репозиторій:
   ```bash
   git clone https://github.com/твій_нікнейм/імʼя_репозиторію.git
   cd імʼя_репозиторію

    Створити та активувати віртуальне середовище (рекомендується):

python3 -m venv venv
source venv/bin/activate  # Linux/MacOS
venv\Scripts\activate     # Windows

Встановити залежності:

    pip install -r requirements.txt

Запуск

Запустити Flask сервер командою:

python app.py

Перейти у браузері за адресою:

http://127.0.0.1:5000

Використання

    На головній сторінці можна завантажити два зображення.

    Натиснути кнопку «Порівняти» для обчислення схожості.

    Результат відображається у відсотках.

Структура проекту

/app.py            - основний серверний файл Flask
/models            - папка з моделями нейронних мереж
/static            - статичні файли (css, js, зображення)
/templates         - html-шаблони для відображення сторінок
/requirements.txt   - список необхідних бібліотек
