# MNIST REST API (FastAPI + Google Colab)

Простой проект, демонстрирующий, как обучить модель для распознавания рукописных цифр MNIST и развернуть её как REST API с помощью FastAPI прямо в Google Colab.

## Возможности

- Обучение модели на датасете MNIST
- Запуск REST API на FastAPI прямо в Google Colab
- Эндпоинты для предсказания, истории запросов и её очистки
- Возможность публикации API в интернет через Serveo

## Требования
- Google Colab
- Python 3.9+
- TensorFlow
- FastAPI
- Uvicorn
- Serveo (для публикации)
---

## Этапы выполнения

### 1. Обучение модели

- Загружаем датасет MNIST (рукописные цифры).
- Нормализуем значения (0–1).
- Переводим метки в формат one-hot.
- Строим простую нейросеть: Flatten → Dense(128) → Dense(10).
- Обучаем модель 10 эпох и сохраняем в файл mnist_model.h5.
```python
# Для загрузки и дальнейшей работы, создаём простейшую модель классификации цифр MNIST

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# загрузка и нормализация
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

# one-hot
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# простая модель
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, validation_split=0.1)
model.save("mnist_model.h5")
```

### 2. Установка зависимостей
- FastAPI — фреймворк для API.
- uvicorn — сервер для запуска API.
- python-multipart — для обработки форм с файлами(необходимо для FAST API).
```bash
!pip install fastapi uvicorn python-multipart
```

### 3. Создание API (main.py)

Маршруты:
- GET /health — проверка, работает ли API
- POST /predict — принимает изображение PNG + метку, возвращает предсказание
- GET /history — показывает все предсказания
- DELETE /history — очищает историю

Ключевые моменты:
- Загрузка модели mnist_model.h5
- Предобработка изображения: 28x28, оттенки серого, нормализация
- Используется request_history — список словарей с предсказаниями
```python
%%writefile main.py

import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from tensorflow.keras.models import load_model
from PIL import Image
import io

app = FastAPI(
    title="MNIST Predictor API",
    description="REST API для предсказания цифр с изображений MNIST и ведения истории запросов")

# Загружаем обученную модель
model = load_model("mnist_model.h5")

# Примитивная "база данных" — история запросов
request_history = []

class PredictionRecord(BaseModel):
    filename: str
    true_label: int
    predicted: int

@app.get("/health",
         summary="Проверка работоспособности",
         description="Простой тест, чтобы убедиться, что API запущен и работает")
async def health():
    return {"status": "API работает корректно"}

@app.post("/predict",
          summary="Предсказание по изображению",
          description="Принимает PNG-изображение цифры 0-9 и возвращает предсказание модели")
async def predict(file: UploadFile = File(...), true_label: int = Form(...)):
    try:
        image = Image.open(file.file).convert("L").resize((28, 28))
        image = np.array(image) / 255.0
        image = image.reshape(1, 28, 28, 1)
        prediction = model.predict(image)
        predicted_label = int(np.argmax(prediction))

        # Запись в историю
        request_history.append({
            "filename": file.filename,
            "true_label": true_label,
            "predicted": predicted_label
        })

        return {"predicted": predicted_label}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.get("/history",
         response_model=List[PredictionRecord],
         summary="История предсказаний",
         description="Возвращает историю всех запросов к /predict")
async def get_history():
    return request_history

@app.delete("/history",
            summary="Очистка истории предсказаний",
            description="Удаляет все предыдущие предсказания из истории")
async def clear_history():
    request_history.clear()
    return {"message": "История успешно очищена"}
```

### 4. Локальный запуск в Colab

- nohup и & — запускают сервер в фоне (чтобы не блокировал ячейки).
- --reload — автообновление при изменении кода.
```bash
# локальный запуск
!nohup uvicorn main:app --reload &
# nohup и & - прописываются для запуска процесса в фоне (чтобы не блокировать Colab)
# параметр --reload позволяет автоматически перезапускать uvicorn при изменениях в файле main.py
!cat nohup.out
```

### 5. Тестирование

- Проверка /health
- Отправка случайного изображения из тестовой выборки на /predict
- Отображение изображения с фактической меткой
- Получение истории /history
- Очистка истории /history (DELETE)
```python
import requests

res = requests.get("http://127.0.0.1:8000/health")
print(res.json())
```

Отправка случайного изображения:

```python
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random

# ПРОВЕРКА МОДЕЛИ НА СЛУЧАЙНОМ ЧИСЛЕ
# Загружаем данные
(_, _), (x_test, y_test) = mnist.load_data()

# Выбираем случайное изображение
idx = np.random.randint(0, len(x_test))
img_array = (x_test[idx] * 255).astype(np.uint8)
img = Image.fromarray(img_array)
img.save("test_digit.png")

# Отображаем изображение
plt.imshow(img_array, cmap="gray")
plt.title(f"Ожидаемая метка: {y_test[idx]}")
plt.axis("off")
plt.show()

# Отправка изображения и метки на сервер
with open("test_digit.png", "rb") as f:
    res = requests.post(
        "http://127.0.0.1:8000/predict",
        files={"file": f},
        data={"true_label": int(y_test[idx])}
    )

# Вывод результата
print("\nПредсказание от модели:", res.json())
print("Ожидаемая метка (из теста):", int(y_test[idx]))
```

Получение и очистка истории:

```python
# Запрос истории предсказаний
history = requests.get("http://127.0.0.1:8000/history").json()
print("\nИстория предсказаний:")
for item in history:
    print(f"Ожидаемая метка: {item['true_label']} — Предсказание: {item['predicted']}")

# Очиcтить историю можно следующим образом
res = requests.delete("http://127.0.0.1:8000/history")
print("Очистка истории:", res.json())
```

### 6. Публикация через Serveo
- Устанавливает SSH-туннель localhost:8000 → https://***.serveo.net
- Автоматически извлекает ссылку из вывода
- Показывает ссылку на:
- - документацию: https://xxxx.serveo.net/docs
- - историю: https://xxxx.serveo.net/history
```python
# запуск с помощью serveo.net
#!ssh -o "StrictHostKeyChecking no" -R 80:localhost:8000 serveo.net
# !ssh -o "StrictHostKeyChecking no" -R 80:localhost:8000 nokey@localhost.run # альтернатива

import subprocess
import re
import time

# Запускаем SSH в фоне
proc = subprocess.Popen(
    ["ssh", "-o", "StrictHostKeyChecking=no", "-R", "80:localhost:8000", "serveo.net"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

# Ждем и читаем вывод
print("Ожидание ссылки от Serveo...")

url = None
for i in range(30):  # максимум ~30 секунд ожидания
    line = proc.stdout.readline()
    print(line.strip())

    match = re.search(r"https://[a-z0-9]+\.serveo\.net", line)
    if match:
        url = match.group(0)
        break
    time.sleep(1)

# Выводим ссылки на документацию и историю запросов
if url:
    print(f"\nДокументация доступна по адресу: {url}/docs")
    print(f"История запросов: {url}/history")
else:
    print("Не удалось найти ссылку Serveo.")
```

---

## Эндпоинты

| Метод  | Путь       | Описание                              |
| ------ | ---------- | ------------------------------------- |
| GET    | `/health`  | Проверка, работает ли API             |
| POST   | `/predict` | Отправка изображения для предсказания |
| GET    | `/history` | Получение истории всех запросов       |
| DELETE | `/history` | Очистка истории                       |

---

## Результат

- Обученная модель
- Запущенный API
- Публичная ссылка (через Serveo)
- Swagger-документация: `/docs`

---

## Полезные ссылки

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [Serveo](https://serveo.net)

---

Готово! Вы можете развернуть свой REST API прямо в браузере, не выходя из Google Colab — быстро, просто и понятно 

