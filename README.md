# FraudTest

Прототип системы обнаружения аномальных транзакций с комбинацией ML и правил. Поддерживает загрузку CSV/ZIP, генерацию синтетических данных, вычисление метрик и интерфейс FastAPI.

## Возможности
- Предобработка временных рядов: извлечение часа/дня недели, агрегаты по истории клиента (mean/std/max/velocity).
- ML-модель IsolationForest с пайплайном (импьютация + one-hot кодирование).
- Правила: высокий чек, рискованная география/категория, высокая скорость транзакций.
- Объяснимость: итоговый score, флаги правил и топ-важности признаков (permutation importance).
- Экспорт результатов в CSV с `transaction_id, fraud_score, is_suspicious, explanation`.
- Метрики: ROC-AUC, PR-AUC, recall@10%, доля помеченных.
- Интерфейсы: CLI (Typer) и FastAPI (`/score`, `/health`).

## Установка
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Быстрый старт
Сгенерировать синтетические данные и обучить модель:
```bash
python -m fraud_detection.cli generate-synthetic --output data/synth.csv --rows 20000
python -m fraud_detection.cli train --input-csv data/synth.csv --model-path artifacts/model.pkl
```

Скоринг нового файла (CSV или ZIP с CSV):
```bash
python -m fraud_detection.cli score --input-csv your_transactions.csv --model-path artifacts/model.pkl --output-csv data/results.csv --threshold 0.6
```
В результате получите CSV с полями `transaction_id, fraud_score, is_suspicious, explanation`.

### Как запустить на `Synthetic_Financial_datasets_log.csv.zip` (~200 МБ)
```bash
# 1. Создать и активировать окружение
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. (Опционально) обучить модель один раз и сохранить артефакт
python -m fraud_detection.cli train --model-path artifacts/model.pkl

# 3. Прогнать скоринг на архиве
python -m fraud_detection.cli score \
  --input-csv Synthetic_Financial_datasets_log.csv.zip \
  --model-path artifacts/model.pkl \
  --output-csv data/results.csv \
  --threshold 0.6 \
  --compute-importance

# 4. Результаты будут в data/results.csv
head -n 5 data/results.csv
```
Примечания:
- Архив ZIP читается напрямую, распаковывать не нужно.
- Если пропустить `--model-path`, CLI обучит IsolationForest на синтетических данных и сразу применит.
- При ограниченной памяти запускайте на машине с ≥4 ГБ RAM; 200 МБ CSV поместится в память Pandas.

Оценка качества:
```bash
python -m fraud_detection.cli evaluate --input-csv data/synth.csv
```

## FastAPI
Запуск сервера:
```bash
uvicorn fraud_detection.api:app --host 0.0.0.0 --port 8000
```
Затем отправляйте POST `/score` с файлом CSV (form-data поле `file`). Параметры: `threshold`, `compute_importance`.

## Примечания
- При отсутствии входного файла CLI использует синтетический датасет.
- Для больших файлов (например, `Synthetic_Financial_datasets_log.csv.zip` ~200 МБ) используйте команду `score`, указав путь к архиву; чтение ZIP поддерживается автоматически.
- Порог аномальности и список правил настраиваются в `config.py`.

## Устранение неполадок при установке зависимостей
Если `pip install -r requirements.txt` завершается ошибками сети/прокси (например, `ProxyError: Tunnel connection failed: 403 Forbidden`):
- Установите зависимости из локального кэша/зеркала PyPI или перенесите нужные `*.whl` файлы (pandas, numpy, scikit-learn, fastapi, uvicorn) на машину и выполните `pip install *.whl`.
- Проверьте переменные окружения прокси (`HTTP_PROXY`, `HTTPS_PROXY`) или временно отключите их для прямого доступа к PyPI.
- При отсутствии интернета используйте предварительно собранный виртуальныйenv/conda-окружение с нужными пакетами и активируйте его перед запуском CLI/серверов.
