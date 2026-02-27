# Image Similarity Search - поиск дубликатов и похожих изображений

Пет-проект: сервис, который по входной картинке находит:
- точные дубликаты (перезаливы, другое сжатие)
- near-duplicates (кроп, поворот, легкие правки)
- top-k похожих изображений

## Что внутри
- извлечение эмбеддингов (CLIP через transformers или ResNet50 fallback)
- построение индекса (FAISS, если установлен; иначе sklearn NearestNeighbors)
- поиск top-k по cosine similarity
- генерация near-duplicates аугментациями (для проверки качества)
- оценка качества (Precision@K, Recall@K, mAP@K) на синтетических парах

## Быстрый старт

### 1) Установка
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
```

Если FAISS не ставится:
- удалите `faiss-cpu` из requirements.txt
- проект автоматически переключится на sklearn NearestNeighbors

### 2) Данные
Положите изображения в `data/images/` (любые jpg/png/webp).

### 3) Построить эмбеддинги и индекс
```bash
python -m src.build_index --images_dir data/images --out_dir artifacts --encoder clip
```

### 4) Поиск похожих
```bash
python -m src.search --query path/to/query.jpg --artifacts_dir artifacts --top_k 10 --out_dir results
```

Скрипт сохранит:
- `results/topk.json` - список top-k с similarity
- `results/montage.jpg` - коллаж (query + top-k)

### 5) Генерация near-duplicates и оценка (опционально)
```bash
python -m src.augment_pairs --images_dir data/images --out_dir data/pairs --per_image 3
python -m src.evaluate --pairs_dir data/pairs --artifacts_dir artifacts --k 10
```

## Лицензия
MIT
