# MNIST REST API (FastAPI + Google Colab)

Простой проект, демонстрирующий, как обучить модель для распознавания рукописных цифр MNIST и развернуть её как REST API с помощью FastAPI прямо в Google Colab.

## ✨ Возможности

- Обучение модели на датасете MNIST
- Запуск REST API на FastAPI прямо в Google Colab
- Эндпоинты для предсказания, истории запросов и её очистки
- Возможность публикации API в интернет через Serveo

---

## 📅 Этапы выполнения

### 1. Обучение модели

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

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

```bash
!pip install fastapi uvicorn python-multipart
```

### 3. Создание API (main.py)

```python
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from typing import List
from pydantic import BaseModel

app = FastAPI(title="MNIST Predictor API")
model = load_model("mnist_model.h5")
request_history = []

class PredictionRecord(BaseModel):
    filename: str
    true_label: int
    predicted: int

@app.get("/health")
async def health():
    return {"status": "API работает корректно"}

@app.post("/predict")
async def predict(file: UploadFile = File(...), true_label: int = Form(...)):
    try:
        image = Image.open(file.file).convert("L").resize((28, 28))
        image = np.array(image) / 255.0
        image = image.reshape(1, 28, 28, 1)
        prediction = model.predict(image)
        predicted_label = int(np.argmax(prediction))

        request_history.append({
            "filename": file.filename,
            "true_label": true_label,
            "predicted": predicted_label
        })

        return {"predicted": predicted_label}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.get("/history", response_model=List[PredictionRecord])
async def get_history():
    return request_history

@app.delete("/history")
async def clear_history():
    request_history.clear()
    return {"message": "История успешно очищена"}
```

### 4. Локальный запуск в Colab

```bash
!nohup uvicorn main:app --reload &
!cat nohup.out
```

### 5. Тестирование

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

(_, _), (x_test, y_test) = mnist.load_data()
idx = np.random.randint(0, len(x_test))
img_array = (x_test[idx] * 255).astype(np.uint8)
img = Image.fromarray(img_array)
img.save("test_digit.png")

with open("test_digit.png", "rb") as f:
    res = requests.post("http://127.0.0.1:8000/predict", files={"file": f}, data={"true_label": int(y_test[idx])})
    print(res.json())
```

Получение и очистка истории:

```python
requests.get("http://127.0.0.1:8000/history").json()
requests.delete("http://127.0.0.1:8000/history").json()
```

### 6. Публикация через Serveo

```python
import subprocess
import re
import time

proc = subprocess.Popen([
    "ssh", "-o", "StrictHostKeyChecking=no", "-R", "80:localhost:8000", "serveo.net"
], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

url = None
for i in range(30):
    line = proc.stdout.readline()
    print(line.strip())
    match = re.search(r"https://[a-z0-9]+\.serveo\.net", line)
    if match:
        url = match.group(0)
        break
    time.sleep(1)

if url:
    print(f"\nДокументация API: {url}/docs")
    print(f"История запросов: {url}/history")
else:
    print("Не удалось получить ссылку от Serveo")
```

---

## 🔎 Эндпоинты

| Метод  | Путь       | Описание                              |
| ------ | ---------- | ------------------------------------- |
| GET    | `/health`  | Проверка, работает ли API             |
| POST   | `/predict` | Отправка изображения для предсказания |
| GET    | `/history` | Получение истории всех запросов       |
| DELETE | `/history` | Очистка истории                       |

## 📊 Структура репозитория

```
mnist-api/
├── mnist_model.h5         # обученная модель
├── main.py                # код FastAPI
├── test_digit.png         # пример изображения
├── README.md              # текущий файл
```

---

## 🎉 Результат

- 📊 Обученная модель
- 🚀 Запущенный API
- 🌐 Публичная ссылка (через Serveo)
- 🏆 Swagger-документация: `/docs`

---

## ✨ Полезные ссылки

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [Serveo](https://serveo.net)

---

Готово! Вы можете развернуть свой REST API прямо в браузере, не выходя из Google Colab — быстро, просто и понятно ✨

