# RAG-пайплайн по базе знаний из 5 документов

Полный цикл: загрузка PDF → очистка → нормализация → структурирование → чанкинг → embeddings →
индексация в Qdrant с валидацией → retrieval (top-k) → сборка контекста → промпт → **YandexGPT 5 Lite** →
ответ с показом использованных фрагментов. Проверяется на наборе из 5 тестовых вопросов.

Проект вырос из конвейера подготовки данных и индексации
(`pipeline → chunk_pipeline → embed_pipeline → vector_pipeline`): описания этапов, схемы JSONL
и семантика артефактов индексации сохранены целиком, значения приведены к текущей базе знаний.
Что именно изменилось относительно той версии — раздел 13.

Все этапы настраиваются через YAML-конфиги, воспроизводимы и не требуют внешних сервисов,
кроме API генерации.

---

## 1. Стек и параметры

| Компонент | Значение | Где настраивается |
|---|---|---|
| Загрузка | PDF (PyPDF2); поддерживаются также `.txt`, `.json`, `.csv`, `.fb2` | `config/config.yaml` |
| Очистка | управляющие символы, лишние пробелы, пустые строки, склейка разрывов PDF | `config/config.yaml` |
| Нормализация | Unicode NFKC, унификация переводов строк, регистр не меняется | `config/config.yaml` |
| Чанкинг | стратегия `sentence`, chunk_size **700** символов, overlap **120** | `config/chunking.yaml` |
| Количество чанков | **58** на текущей базе (5 PDF, 19 страниц, 32 837 символов) | — |
| Embeddings | `intfloat/multilingual-e5-small`, 384 измерения, префиксы `passage:` / `query:` | `config/embeddings.yaml` |
| Vector store | Qdrant в локальном режиме (без сервера), коллекция `rag_documents`, метрика Cosine | `config/vector_store.yaml` |
| Retrieval | top-k, по умолчанию **5**, параметр `--top-k` | `config/rag.yaml` |
| LLM | **YandexGPT 5 Lite** (`yandexgpt-5-lite/latest`) через OpenAI-совместимый эндпоинт Yandex Cloud, temperature 0 | `config/rag.yaml` + `.env` |

Почему `multilingual-e5-small`, а не `all-MiniLM-L6-v2` из прошлой версии: база знаний на русском,
англоязычная MiniLM на кириллице даёт заметно худший retrieval. Размерность у обеих 384,
поэтому `vector_size` в конфиге индексации менять не пришлось.

---

## 2. Возможности

- **Поддержка форматов**: `.txt`, `.json`, `.csv`, `.pdf` (через PyPDF2), `.fb2` (FictionBook).
  Текущая база знаний — только PDF.
- **Очистка текста**: удаление управляющих символов, лишних пробелов, пустых строк;
  отдельная опция `unwrap_pdf_line_breaks` склеивает строки, разорванные вёрсткой PDF.
- **Нормализация**: Unicode (NFKC), унификация переводов строк.
- **Структурирование**: единый объект с `text` и `metadata` (источник, тип, `document_id`, секция,
  дата обработки) плюс прикладные поля из sidecar: `document_name`, `category`, `version`, `source_type`.
- **Чанкинг**: стратегии `sentence` или `paragraph`, настраиваемый размер и overlap;
  `chunk_size` — жёсткий предел (длинные предложения режутся по словам).
- **Стабильные идентификаторы**: каждый чанк получает `chunk_id` в формате UUID, однозначно
  связывающий его с документом.
- **Embeddings**: локальная модель `sentence-transformers` (размерность 384), пакетная обработка,
  префиксы семейства E5, offline-заглушка для смоук-теста.
- **Индексация**: Qdrant (локальный режим) с автоматическим созданием коллекции, проверкой
  параметров, загрузкой векторов и payload (метаданных).
- **Полная валидация**: проверка количества записей, размерности, наличия metadata, payload
  в Qdrant, возможности выполнения поиска.
- **Тестовый поиск**: два режима — 1) по существующим векторам (без пересчёта), 2) по текстовым
  запросам с вычислением эмбеддингов той же моделью. Результаты сохраняются с оценкой релевантности.
- **Артефакты индексации**: `manifest.json`, `validation.json`, `search_results.json` в `data/vector_store/`.
- **RAG-слой**: retrieval по вопросу, сборка контекста с лимитами, промпт «только по контексту»,
  генерация в YandexGPT 5 Lite, вывод использованных фрагментов со score и метаданными.
- **Артефакты прогона**: `results/qa_results.json` и `results/qa_report.md` по 5 тестовым вопросам.

---

## 3. Структура проекта

```
KB products/
├── config/
│   ├── config.yaml           # подготовка данных
│   ├── chunking.yaml         # чанкинг
│   ├── embeddings.yaml       # расчёт эмбеддингов
│   ├── vector_store.yaml     # индексация в Qdrant
│   ├── rag.yaml              # retrieval + контекст + промпт + LLM
│   └── questions.yaml        # 5 тестовых вопросов с ожидаемыми ответами
├── data/
│   ├── raw/                  # 5 PDF базы знаний + metadata.yaml (входные данные)
│   ├── prepared/             # очищенные документы (.jsonl)
│   ├── chunks/               # чанки (.jsonl)
│   ├── embeddings/           # векторы (.jsonl)
│   ├── vector_store/         # артефакты индексации
│   │   ├── manifest.json
│   │   ├── validation.json
│   │   └── search_results.json
│   └── qdrant_storage/       # данные Qdrant (создаются автоматически)
├── results/                  # qa_results.json, qa_report.md (создаются rag_eval.py)
├── src/
│   ├── loader.py             # загрузка файлов (TXT, JSON, CSV, PDF, FB2)
│   ├── cleaner.py            # очистка текста, склейка разрывов строк PDF
│   ├── normalizer.py         # нормализация (Unicode, переносы)
│   ├── structurer.py         # структурирование + метаданные из data/raw/metadata.yaml
│   ├── exporter.py           # экспорт подготовленных документов
│   ├── pipeline.py           # ЭТАП 1: пайплайн подготовки данных
│   ├── chunk_loader.py       # загрузка подготовленных документов
│   ├── chunker.py            # разбиение на чанки
│   ├── chunk_exporter.py     # экспорт чанков
│   ├── chunk_pipeline.py     # ЭТАП 2: пайплайн чанкинга
│   ├── embed_loader.py       # загрузка чанков для эмбеддингов
│   ├── embedder.py           # расчёт эмбеддингов (sentence-transformers)
│   ├── embed_exporter.py     # экспорт эмбеддингов
│   ├── embed_pipeline.py     # ЭТАП 3: пайплайн эмбеддингов
│   ├── vector_loader.py      # загрузка эмбеддингов с валидацией
│   ├── vector_indexer.py     # создание коллекции, загрузка в Qdrant
│   ├── vector_validator.py   # проверка индекса (кол-во, payload, поиск)
│   ├── vector_searcher.py    # тестовый поиск (два режима)
│   ├── vector_manifest.py    # генерация manifest.json
│   ├── vector_pipeline.py    # ЭТАП 4: пайплайн индексации
│   ├── retriever.py          # retrieval по вопросу (top-k) из Qdrant
│   ├── context_builder.py    # сборка контекста из найденных чанков
│   ├── llm_client.py         # клиент LLM (OpenAI-совместимый / нативный YandexGPT)
│   ├── rag_answer.py         # ЭТАП 5: CLI «вопрос → ответ»
│   ├── rag_eval.py           # прогон 5 тестовых вопросов, отчёт
│   ├── llm_check.py          # проверка доступа к LLM без Qdrant и индекса
│   └── check_results.py      # просмотр JSONL-артефактов
├── tools/md_to_pdf.py        # опционально: сборка PDF из data/source_md/*.md
├── .env.example
├── .gitignore
└── requirements.txt
```

---

## 4. База знаний

`data/raw/` — 5 PDF-документов, по одному на продукт. Это входные данные пайплайна,
они лежат в репозитории и никак не генерируются.

| Файл | Стр. | Документ | Категория | Версия |
|---|---|---|---|---|
| `01_ergo_business_simulator.pdf` | 3 | ERGO — бизнес-симулятор принятия стратегических решений | products/leadership_development | 2025-09 |
| `02_sips_investment_planning.pdf` | 4 | SIPS — Strategic Investment Planning Suite | products/strategic_investment_planning | 2025-09 |
| `03_aims_asset_management.pdf` | 4 | AIMS — Asset Intelligence Management System 2.0 | products/asset_management | 2025-09 |
| `04_ibp_integrated_business_planning.pdf` | 4 | IBP — Интегрированное бизнес-планирование | products/integrated_planning | 2025-06 |
| `05_integral360_infrastructure_design.pdf` | 4 | Integral 360 — концептуальное проектирование индустриально-логистических объектов | products/infrastructure_projects | 2025-09 |

Метаданные (`document_name`, `category`, `version`, `source_type`) задаются в `data/raw/metadata.yaml`
и проходят через все этапы: попадают в метаданные чанка, в embeddings и в payload Qdrant,
поэтому видны в результатах поиска и в ответе.

**Файл `data/raw/metadata.yaml` обязателен.** Без него пайплайн не падает, но в payload уходит
`document_name` = имя файла, `category` = `uncategorized`, `version` = `n/a` — и требование
«метаданные видны в ответе» перестаёт выполняться. Ключ записи — имя PDF-файла.
Добавили документ в `data/raw/` — добавьте запись и сюда.

```yaml
defaults:
  document_name: "Без названия"
  category: "uncategorized"
  version: "n/a"
  source_type: "internal_product_doc"

documents:
  01_ergo_business_simulator.pdf:
    document_name: "ERGO — бизнес-симулятор принятия стратегических решений"
    category: "products/leadership_development"
    version: "2025-09"
    source_type: "internal_product_doc"
  ...
```

### Как получены документы

Исходники — презентации в PDF с версткой, схемами, логотипами и контактами. Для RAG они не годятся:
экстрактор вытаскивает подписи к картинкам вперемешку с текстом, а лишние сущности (названия компаний,
ФИО, телефоны, e-mail) в базе знаний не нужны и создают риски по персональным данным.

Что сделано:
1. содержимое слайдов переписано в «плоский» текст с заголовками разделов и списками;
2. удалена вся графика, схемы, дэшборды и логотипы;
3. удалены названия компаний — как поставщика решений, так и клиентов и упомянутых публичных компаний
   (заменены на обезличенные описания: «международная авиакомпания», «горно-металлургическая компания» и т. п.);
4. полностью удалены контакты и персональные данные (ФИО, телефоны, e-mail, ссылки на соцсети);
5. в самих PDF оставлены только заголовок, строки «Документ» и «Категория» и содержательный текст;
   версия документа вынесена в `data/raw/metadata.yaml` и в тело PDF не попадает;
6. все проверяемые факты и числа сохранены — иначе не на чем проверять качество retrieval.

Обезличенные PDF — конечный результат этой подготовки, менять их не нужно.
Утилита `tools/md_to_pdf.py` (reportlab, шрифт DejaVu Sans) остаётся в проекте на случай,
когда базу знаний нужно пересобрать: положите тексты в `data/source_md/*.md` и запустите

```bash
python tools/md_to_pdf.py            # data/source_md/*.md -> data/raw/*.pdf
```

В обычном сценарии («вход — готовые PDF») этот шаг не выполняется, а `reportlab`
из `requirements.txt` не нужен.

---

## 5. Установка и зависимости

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # и вписать свои значения
```

```
PyYAML==6.0.1
sentence-transformers>=2.7
torch>=1.9.0
PyPDF2>=3.0.0
qdrant-client>=1.10,<2.0
reportlab>=4.0        # нужен только для tools/md_to_pdf.py
```

Требуется Python 3.9+. `torch` весит 0,8–2,5 ГБ в зависимости от платформы — на медленном канале
установка занимает 5–15 минут. `src/llm_client.py` и `src/llm_check.py` внешних зависимостей
не имеют, работают на stdlib.

---

## 6. Подключение YandexGPT 5 Lite

Используется OpenAI-совместимый эндпоинт Yandex Cloud: `https://llm.api.cloud.yandex.net/v1`,
тот же формат `/chat/completions`, что у OpenAI. Нативный API Foundation Models остаётся
запасным вариантом (`llm.provider: "yandex"`).

### Порядок действий в Yandex AI Studio

1. **Биллинг.** Аккаунт Yandex Cloud с привязанным платёжным аккаунтом. Без него Foundation Models
   возвращают 403 даже при корректных ролях; стартовый грант подходит.
2. **Folder ID.** Консоль → нужный каталог → поле **ID** вида `b1g...` (20 символов). Это `YC_FOLDER_ID`.
   Он же виден в адресной строке: `https://console.yandex.cloud/folders/b1g...`.
   Не перепутайте с ID облака — префикс тот же `b1g`, но в URL сегмент `/cloud/`, а не `/folders/`.
3. **Точный идентификатор модели.** Каталог моделей → карточка **YandexGPT 5 Lite** → у нужного
   инстанса кнопка **«URI инстанса»** копирует готовый `gpt://<folder_id>/yandexgpt-5-lite/latest`.
   Не набирайте идентификатор руками: название в каталоге («YandexGPT 5 Lite») и идентификатор
   (`yandexgpt-5-lite/latest`) различаются, а старое имя `yandexgpt-lite/latest` — это другая модель.
4. **Выбор инстанса** — четыре варианта в карточке не равнозначны:

   | Инстанс | Суффикс URI | Когда брать |
   |---|---|---|
   | Базовый | закреплённая версия | прод и зачётный прогон `rag_eval.py`: версия не поедет под вами |
   | Latest | `/latest` | разработка и отладка |
   | RC | `/rc` | обкатка следующей версии, для боевого RAG не нужен |
   | Deprecated | `/deprecated` | только для отката, отключается по расписанию |

5. **Сервисный аккаунт и роль.** «Сервисные аккаунты» → «Создать» → назначить на каталоге роль
   **`ai.languageModels.user`**. Именно её, не `editor`. Роли `ai.assistants.user`
   и `ai.foundationModels.editor` нужны только для агентов и дообучения — здесь не используются.
6. **API-ключ.** Карточка сервисного аккаунта → «Создать новый ключ» → **«Создать API-ключ»**.
   Если предлагается ограничить **область действия (scope)** — выберите генерацию текста
   (`yc.ai.languageModels.execute`); ключ с чужим scope даёт 403, который легко принять за
   проблему с ролью. Секрет показывается один раз, начинается с `AQVN` — это `LLM_API_KEY`.
7. **Проверка в Playground до кода.** «Попробовать в Playground» → вставить системный промпт
   из `config/rag.yaml` (`prompt.system`), один тестовый вопрос и готовый контекст;
   выставить `temperature = 0` (дефолт модели 0.3 для фактологического RAG избыточен).
   Кнопка **«Посмотреть код»** отдаёт рабочий сниппет с точным эндпоинтом и телом запроса.
8. **Заполнить `.env`** и запустить `python src/llm_check.py`.

### Файл `.env`

```
YC_FOLDER_ID=<ваш-folder-id>
LLM_API_KEY=<ваш-API-ключ>
LLM_BASE_URL=https://llm.api.cloud.yandex.net/v1
LLM_MODEL=yandexgpt-5-lite/latest
```

`.env` в репозиторий не коммитится (`.gitignore`), в код и артефакты ключи не попадают —
`src/llm_client.py` читает только переменные окружения, а в выводе показывает 4 последних символа ключа.

Клиент сам достраивает короткое имя модели до полного идентификатора
`gpt://<YC_FOLDER_ID>/<LLM_MODEL>` — Yandex ожидает в поле `model` именно его.
Можно вставить в `LLM_MODEL` сразу полный URI из шага 3, тогда `YC_FOLDER_ID` для
OpenAI-совместимого пути не потребуется (для нативного провайдера он нужен всегда:
уходит отдельным заголовком `x-folder-id`).

Схема авторизации определяется автоматически: API-ключ (`AQVN...`) → `Api-Key`,
IAM-токен (`t1....`) → `Bearer`. Переопределяется переменной `LLM_AUTH_SCHEME`.

### Диагностика ошибок доступа

| Код | Причина в большинстве случаев |
|---|---|
| 400 | в поле `model` ушло короткое имя без префикса `gpt://<folder_id>/` |
| 401 | не та схема авторизации: `Bearer` с API-ключом вместо `Api-Key` |
| 403 | нет роли `ai.languageModels.user`, не привязан биллинг или ключ выпущен с другим scope |
| 404 | опечатка в идентификаторе — например старый `yandexgpt-lite` вместо `yandexgpt-5-lite` |
| 429 | превышена квота каталога по запросам или токенам («Квоты» в консоли) |

Те же подсказки клиент печатает сам — см. `_post()` в `src/llm_client.py`.

### Выбор модели

| Модель | Когда брать |
|---|---|
| `yandexgpt-5-lite/latest` (YandexGPT 5 Lite) | по умолчанию: ответ по готовому контексту, дёшево и быстро; контекст 32 768 токенов |
| `yandexgpt/latest` (YandexGPT 5 Pro) | если Lite фантазирует на вопросе q5 или теряет пункты в длинных перечислениях |
| `gpt-oss-120b` | открытая модель, альтернатива Pro; полезна для сравнения качества на том же контексте |

Для RAG-задачи «ответь строго по контексту» размер модели решает меньше, чем промпт:
разница между Lite и Pro видна в основном на q5 (вопрос без ответа в базе) и на полноте
перечислений. Начните с Lite, переключайтесь одной строкой в `.env`.

Function calling, Responses API и background-режим, заявленные в карточке модели, здесь
не задействованы: клиент шлёт обычный синхронный `/chat/completions` без инструментов.

### Другие провайдеры

Для OpenAI / DeepSeek / Ollama оставьте `provider: "openai_compatible"` и заполните:

```
LLM_API_KEY=...
LLM_BASE_URL=https://api.openai.com/v1     # Ollama: http://localhost:11434/v1
LLM_MODEL=gpt-4o-mini
```

Локальная LLM без ключа тоже подходит: `LLM_BASE_URL=http://localhost:11434/v1`, `LLM_MODEL=llama3.1`.

---

## 7. Этапы пайплайна

### Этап 1. Подготовка данных (загрузка, очистка, нормализация, структурирование)

Исходные файлы лежат в `data/raw/` (поддерживаются `.txt`, `.json`, `.csv`, `.pdf`, `.fb2`),
прикладные метаданные — в `data/raw/metadata.yaml`.

```bash
python src/pipeline.py
```

Результат: `data/prepared/prepared_documents.jsonl` — каждая строка:

```json
{
  "text": "очищенный текст документа",
  "metadata": {
    "source": "data/raw/01_ergo_business_simulator.pdf",
    "file_name": "01_ergo_business_simulator.pdf",
    "file_type": "pdf",
    "document_id": "e4bf815c4dec968a",
    "section": "main",
    "processed_at": "2026-09-04T10:47:28",
    "document_name": "ERGO — бизнес-симулятор принятия стратегических решений",
    "category": "products/leadership_development",
    "version": "2025-09",
    "source_type": "internal_product_doc"
  }
}
```

`document_id` — усечённый SHA-256 от текста и источника: стабилен между запусками,
пока сам файл не изменился.

### Этап 2. Чанкинг (разбиение на фрагменты)

Параметры — в `config/chunking.yaml` (стратегия, размер, overlap, обработка длинных предложений).

`chunk_size` — жёсткий верхний предел длины чанка. Если отдельное предложение длиннее `chunk_size`,
при `split_long_sentences: true` оно дополнительно режется по словам (до `chunk_size - chunk_overlap`,
чтобы чанк вместе с перекрытием тоже уложился в лимит). При `split_long_sentences: false`
длинное предложение попадает в чанк целиком и `chunk_size` перестаёт быть реальным ограничением.

Стратегия `sentence` выбрана осознанно: `paragraph` для PDF не подходит, потому что вёрстка
рвёт абзацы на произвольные строки.

```bash
python src/chunk_pipeline.py
```

Результат: `data/chunks/chunks.jsonl` — каждая строка содержит стабильный `chunk_id` (UUID)
и все метаданные документа:

```json
{
  "text": "текст чанка",
  "metadata": {
    "source": "data/raw/01_ergo_business_simulator.pdf",
    "file_name": "01_ergo_business_simulator.pdf",
    "file_type": "pdf",
    "document_id": "e4bf815c4dec968a",
    "section": "main",
    "processed_at": "2026-09-04T10:47:28",
    "document_name": "ERGO — бизнес-симулятор принятия стратегических решений",
    "category": "products/leadership_development",
    "version": "2025-09",
    "source_type": "internal_product_doc",
    "chunk_id": "b9a71db8-79b6-5cd5-bebe-d4b5a66781c1",
    "chunk_index": 0,
    "chunk_size": 609,
    "chunk_overlap": 120,
    "strategy": "sentence"
  }
}
```

На текущей базе получается **58 чанков**: мин. 157 / средн. 593 / макс. 713 символов.

### Этап 3. Расчёт эмбеддингов

Модель и параметры — в `config/embeddings.yaml`.

```bash
python src/embed_pipeline.py
```

При первом запуске модель (~130 МБ) скачивается с HuggingFace.

Результат: `data/embeddings/embeddings.jsonl` — каждая строка:

```json
{
  "text": "текст чанка",
  "embedding": [0.123, -0.456, "... всего 384 значения"],
  "metadata": {
    "...": "все поля чанка, плюс:",
    "embedding_model": "intfloat/multilingual-e5-small",
    "embedding_dimensions": 384
  }
}
```

Модель семейства E5 требует префиксов: документы кодируются с `passage: `, запросы — с `query: `.
Префикс подставляется автоматически и в текст чанка в артефакте не попадает.

### Этап 4. Индексация в Qdrant

Убедитесь, что `data/embeddings/embeddings.jsonl` существует. Настройте `config/vector_store.yaml`
(имя коллекции, размерность, метрика, режим тестового поиска).

```bash
python src/vector_pipeline.py
```

Можно передать другой конфиг: `python src/vector_pipeline.py config/vector_store.yaml`.
Скрипт возвращает код выхода `0`, если `index_is_valid: true`, и `1`, если валидация
не прошла — это позволяет использовать этап в CI.

**Storage Qdrant** (local mode, без сервера): `data/qdrant_storage/`.
При `recreate_collection: true` коллекция `rag_documents` удаляется и создаётся заново.
При `recreate_collection: false` параметры существующей коллекции (размерность и метрика)
сверяются с конфигом, и при расхождении пайплайн **останавливается с понятной ошибкой**,
а не пишет данные в несовместимую коллекцию.

Недопустимое значение `distance` (не Cosine / Dot / Euclid) также приводит к явной ошибке:
молчаливой подмены метрики на Cosine нет.

#### Что делает этот этап

- **Загружает эмбеддинги** с жёсткой валидацией до обращения к Qdrant: наличие `text`, `embedding`,
  `metadata`, `chunk_id`, `document_id`, `embedding_model`, `embedding_dimensions`; непустой вектор;
  отсутствие NaN/Infinity; совпадение фактической размерности с `vector_size` из конфига
  и с `embedding_dimensions` из metadata. Пустые строки, отклонённые записи и валидные записи
  считаются раздельно и все попадают в артефакты.
- **Создаёт коллекцию** в Qdrant в локальном режиме (если она уже существует — проверяет
  её параметры, при необходимости пересоздаёт).
- **Загружает векторы** пакетами: `PointStruct` формируются порциями по `batch_size`,
  полный список точек не держится в памяти целиком. Идентификатор точки = `chunk_id`.
- **Проверяет**, что все точки загружены, их payload содержит обязательные поля
  (`text`, `chunk_id`, `document_id` и др.).
- **Выполняет тестовый поиск** в двух режимах и сохраняет результаты с оценкой релевантности.
- **Генерирует три артефакта** в `data/vector_store/`.

#### Результаты

Данные загружены в локальное хранилище Qdrant. Все файлы Qdrant (коллекции, векторы, индексы)
сохраняются в папке из конфигурации:

```yaml
paths:
  qdrant_storage: "data/qdrant_storage"
```

Папка создаётся автоматически при первом запуске `vector_pipeline.py`. Если удалить её, все данные
будут потеряны — при следующем запуске коллекция будет создана заново (при `recreate_collection: true`).

Созданы артефакты:

- `manifest.json` — параметры запуска, статистика, пути к артефактам;
- `validation.json` — детальные результаты проверок (количество, размерность, payload, поиск);
- `search_results.json` — результаты тестовых поисков с `relevance_judgment` (high/medium/low).

Подробный разбор всех трёх — раздел 10.

### Этап 5. Ответ на вопрос (RAG)

```bash
python src/rag_answer.py "Что входит в цифровое ядро AIMS?"
```

1. **Retrieval** (`src/retriever.py`): вопрос кодируется той же моделью с префиксом `query: `,
   в Qdrant ищутся top-k ближайших векторов. Проверяется, что размерность модели совпадает
   с размерностью коллекции, иначе — явная ошибка, а не тихий мусор в выдаче.
2. **Context builder** (`src/context_builder.py`): чанки собираются в пронумерованный контекст,
   каждый фрагмент отделён явно и подписан источником:

```
[Фрагмент 1]
Источник: IBP — Интегрированное бизнес-планирование (версия 2025-06, категория products/integrated_planning)
Файл: 04_ibp_integrated_business_planning.pdf | чанк 9 | score 0.8712
Текст: ...
```

   Действуют лимиты `max_chars_per_chunk` (2000) и `max_context_chars` (12000): если общий контекст
   не влезает, лишние фрагменты отбрасываются и в «использованные» не попадают.
3. **Промпт** (`config/rag.yaml`, ключ `prompt.system`): отвечать только по контексту, не выдумывать
   факты и числа, при отсутствии ответа выдать ровно «Недостаточно информации в предоставленных
   документах.», в конце перечислить использованные фрагменты.
4. **LLM** (`src/llm_client.py`): POST на `{base_url}/chat/completions` в формате OpenAI,
   `temperature: 0`. Для Yandex в поле `model` уходит `gpt://<folder>/<модель>`, заголовок —
   `Authorization: Api-Key <ключ>`. Запасной провайдер `yandex` использует нативный API
   Foundation Models (`modelUri`, `completionOptions`, сообщения `{role, text}`, эндпоинт
   `.../foundationModels/v1/completion`).
5. **Вывод**: ответ + список фрагментов со score, названием документа, версией, именем файла,
   `chunk_index` и `chunk_id`.

#### Бюджет контекста

Окно YandexGPT 5 Lite — 32 768 токенов на весь запрос (system + контекст + вопрос + ответ).
Для русского текста ориентир ~3–4 символа на токен:

| Часть запроса | Символов | ≈ токенов |
|---|---|---|
| системный промпт | ~700 | ~200 |
| контекст (5 чанков + метаданные), лимит | 12 000 | 3 500–4 000 |
| вопрос | ~150 | ~50 |
| ответ (`max_tokens`) | — | 700 |
| **итого** | — | **~5 000 из 32 768** |

Запас шестикратный, вся база знаний целиком — 32 837 символов (~9–11 тыс. токенов) — тоже влезла бы
в окно. Но заливать все 58 чанков в промпт не нужно: на длинном контексте Lite теряет пункты
в середине сильнее, чем на 5 отобранных фрагментах, а счёт идёт по токенам ввода. Узкое место
здесь — качество retrieval, а не размер окна.

---

## 8. Конфигурационные файлы

`config/config.yaml` — подготовка данных

```yaml
paths:
  raw_dir: "data/raw"                       # папка с исходными файлами базы знаний (PDF)
  prepared_dir: "data/prepared"             # папка для очищенных документов
  metadata_file: "data/raw/metadata.yaml"   # sidecar с document_name / category / version

cleaning:
  remove_extra_spaces: true         # удалять множественные пробелы и табуляции
  remove_empty_lines: true          # удалять пустые строки
  remove_control_characters: true   # удалять управляющие символы (кроме перевода строки)
  unwrap_pdf_line_breaks: true      # склеивать строки, разорванные вёрсткой PDF

normalization:
  unicode_form: "NFKC"              # форма нормализации Unicode (NFKC, NFC, NFD, NFKD)
  normalize_line_endings: true      # приводить переносы строк к \n
  lowercase: false                  # регистр не трогаем: E5-модели чувствительны к нему

export:
  format: "jsonl"
  filename: "prepared_documents.jsonl"
```

`config/chunking.yaml` — чанкинг

```yaml
paths:
  input_jsonl: "data/prepared/prepared_documents.jsonl"
  output_jsonl: "data/chunks/chunks.jsonl"

chunking:
  strategy: sentence            # sentence — устойчиво к переносам строк из PDF
  chunk_size: 700               # жёсткий верхний предел длины чанка в символах (~150-180 слов)
  chunk_overlap: 120            # перекрытие в символах: удерживает контекст на границе абзацев
  split_long_sentences: true    # длинные перечисления режутся по словам, лимит остаётся реальным
```

`config/embeddings.yaml` — эмбеддинги

```yaml
paths:
  input_jsonl: "data/chunks/chunks.jsonl"
  output_jsonl: "data/embeddings/embeddings.jsonl"

embedding:
  model_name: "intfloat/multilingual-e5-small"  # мультиязычная модель: база знаний на русском
  provider: "sentence-transformers"
  dimensions: 384                               # сверяется с фактической размерностью модели
  batch_size: 32
  max_tokens: 512
  passage_prefix: "passage: "                   # требование семейства E5 для документов
  normalize_embeddings: true                    # нормируем векторы -> cosine == dot

export:
  include_text: true
```

`config/vector_store.yaml` — индексация

```yaml
paths:
  input_jsonl: "data/embeddings/embeddings.jsonl"
  output_dir: "data/vector_store"
  qdrant_storage: "data/qdrant_storage"

vector_store:
  type: "qdrant"
  collection_name: "rag_documents"
  vector_size: 384                 # multilingual-e5-small
  distance: "Cosine"               # Cosine, Dot или Euclid
  recreate_collection: true
  batch_size: 64

search:
  top_k: 5
  test_queries:
    - "Сколько сегментов рынка микроэлектроники в игре ERGO?"
    - "Какие эффекты даёт внедрение IBP?"
    - "Что входит в цифровое ядро AIMS?"
  use_existing_vectors: true       # true — self-match по существующим векторам
  model_name: "intfloat/multilingual-e5-small"
```

`config/rag.yaml` — retrieval, контекст, промпт, LLM

```yaml
retrieval:
  collection_name: "rag_documents"
  top_k: 5                                      # переопределяется параметром --top-k
  score_threshold: 0.0                          # 0.0 = не отсекать
  model_name: "intfloat/multilingual-e5-small"  # ОБЯЗАТЕЛЬНО та же модель, что на этапе embeddings
  query_prefix: "query: "                       # префикс E5 для запросов
  normalize_embeddings: true

context:
  max_chars_per_chunk: 2000
  max_context_chars: 12000
  include_metadata: true

llm:
  provider: "openai_compatible"    # openai_compatible | yandex | none
  model: "yandexgpt-5-lite/latest"
  base_url: "https://llm.api.cloud.yandex.net/v1"
  temperature: 0.0
  max_tokens: 700
  timeout_seconds: 60

prompt:
  system: |
    Ты — ассистент, отвечающий на вопросы строго по предоставленному контексту.
    ...
```

`config/questions.yaml` — 5 тестовых вопросов с полем `expected` для сверки (раздел 11).

---

## 9. Запуск пайплайна пошагово

Все команды выполняются из корня проекта.

1. Подготовьте исходные данные — PDF в `data/raw/`, метаданные в `data/raw/metadata.yaml`.
2. Установите зависимости: `pip install -r requirements.txt`
3. Заполните `.env` и проверьте доступ к LLM:

```bash
python src/llm_check.py
```

4. Подготовка данных: `python src/pipeline.py`
5. Чанкинг: `python src/chunk_pipeline.py`
6. Эмбеддинги: `python src/embed_pipeline.py`
7. Индексация: `python src/vector_pipeline.py`

Полный прогон одной строкой:

```bash
python src/pipeline.py && python src/chunk_pipeline.py && python src/embed_pipeline.py && python src/vector_pipeline.py
```

Затем вопросы:

```bash
# один вопрос
python src/rag_answer.py "Что входит в цифровое ядро AIMS?"

# другое количество извлекаемых чанков
python src/rag_answer.py "Какие эффекты даёт внедрение IBP?" --top-k 8

# только retrieval, без обращения к LLM (отладка поиска)
python src/rag_answer.py "Какие эффекты даёт внедрение IBP?" --no-llm

# машиночитаемый вывод
python src/rag_answer.py "Что такое SIPS?" --json

# прогон всех 5 тестовых вопросов и генерация отчёта
python src/rag_eval.py
```

`rag_eval.py` сохраняет `results/qa_results.json` (вопрос, найденные чанки, контекст, ответ)
и `results/qa_report.md` (тот же материал в читаемом виде с полями «Оценка» и «Комментарий»
под каждым ответом — их заполняете вручную после прогона).

### Контрольные точки прогона

Значения получены на текущей базе знаний — сверяйтесь с ними.

| Этап | Что должно быть в выводе |
|---|---|
| `llm_check.py` | идентификатор начинается с `gpt://` и содержит ваш folder ID, `схема авт.: Api-Key`, ответ модели непустой. Коды возврата: 0 — работает, 1 — HTTP-ошибка API, 2 — не заполнен `.env` |
| `pipeline.py` | `Всего документов: 5`, `Форматы: {'pdf'}`; строки `Предупреждение: файл метаданных ... не найден` быть не должно |
| `chunk_pipeline.py` | `Создано 58 чанков` |
| `embed_pipeline.py` | `Размерность вектора: 384`, `Всего чанков: 58`; строки про `RAG_FAKE_EMBEDDINGS` быть не должно |
| `vector_pipeline.py` | `загружено в Qdrant: 58`, `размерность коллекции: 384 (метрика COSINE)`, `search_works: True`, `index_is_valid: True` |

Проверка retrieval отдельно от генерации — так видно, чья вина при плохом ответе:

```bash
python src/rag_answer.py "Какие три цифровых решения входят в цифровое ядро AIMS?" --no-llm
```

В топе должен стоять фрагмент из `03_aims_asset_management.pdf`. Для e5 на попадающем вопросе
score первого фрагмента обычно 0,80–0,90; ниже 0,75 при формально верном документе — модель нашла
документ «на грани», это аргумент за гибридный поиск.

### Смоук-тест без интернета

Если нет доступа к HuggingFace, связку «чанк → вектор → Qdrant → retrieval» можно проверить
на детерминированной заглушке вместо настоящей модели:

```bash
RAG_FAKE_EMBEDDINGS=1 python src/embed_pipeline.py
RAG_FAKE_EMBEDDINGS=1 python src/vector_pipeline.py
RAG_FAKE_EMBEDDINGS=1 python src/rag_answer.py "Что входит в цифровое ядро AIMS?" --no-llm
```

Режим проверяет только работоспособность конвейера — качество поиска в нём нерелевантно.
Перед реальным использованием удалите `data/embeddings/`, `data/vector_store/`, `data/qdrant_storage/`
и переиндексируйте настоящей моделью.

### Очистка данных и перезапуск проекта

Чтобы запустить пайплайн с нуля (после замены исходных файлов или изменения конфигурации),
удалите все промежуточные и итоговые данные — иначе возможны конфликты с устаревшими файлами:

```bash
rm -rf data/prepared data/chunks data/embeddings data/vector_store data/qdrant_storage
```

- `data/prepared/` — подготовленные документы;
- `data/chunks/` — чанки;
- `data/embeddings/` — эмбеддинги;
- `data/vector_store/` — артефакты индексации (validation, search_results, manifest);
- `data/qdrant_storage/` — локальные данные Qdrant (векторы, индексы, payload).

**Важно:** не удаляйте `data/raw/` — там исходные PDF и `metadata.yaml`.

---

## 10. Проверка результатов и артефакты индексации

**Подготовка данных.** Откройте `data/prepared/prepared_documents.jsonl` — текст очищен,
структурирован, в metadata заполнены `document_name`, `category`, `version`.

**Чанкинг.** Проверьте количество чанков, наличие `chunk_id` (UUID) и полноту метаданных.

**Эмбеддинги.** Все векторы одинаковой размерности, в метаданных есть `embedding_model`
и `embedding_dimensions`.

Быстрый просмотр обоих файлов:

```bash
python src/check_results.py                            # чанки и эмбеддинги по умолчанию
python src/check_results.py data/chunks/chunks.jsonl 5 # произвольный файл, 5 записей
```

**Индексация.** Достаточно одного флага `"index_is_valid": true` в `validation.json` — он агрегирует
`count_match`, `vector_size_match`, `distance_metric_match`, `payload_fields_present`,
`search_works`, `all_input_records_valid` и пустой `errors`.

Одной командой:

```bash
python -c "import json;d=json.load(open('data/vector_store/validation.json'));print({k:d[k] for k in ('actual_points_in_qdrant','vector_size_match','distance_metric_match','payload_fields_present','search_works','index_is_valid')});print('errors:',d['errors'])"
```

### 1. `validation.json` — детальная проверка индекса

Фактический вывод на текущей базе знаний:

```json
{
  "total_lines_in_file": 58,
  "blank_lines_in_file": 0,
  "data_lines_in_file": 58,
  "valid_records_loaded": 58,
  "error_records_in_file": 0,
  "counters_consistent": true,
  "all_input_records_valid": true,
  "actual_points_in_qdrant": 58,
  "count_match": true,
  "count_match_vs_data_lines": true,
  "vector_size_expected": 384,
  "vector_size_actual": 384,
  "vector_size_match": true,
  "distance_metric_expected": "Cosine",
  "distance_metric_actual": "COSINE",
  "distance_metric_match": true,
  "all_embeddings_have_same_dimension": true,
  "payload_checked_points": 5,
  "payload_fields_present": true,
  "payload_required_fields": ["chunk_id", "document_id", "text", "source", "section", "embedding_model", "embedding_dimensions"],
  "search_works": true,
  "search_returned_hits": 1,
  "errors": [],
  "index_is_valid": true
}
```

Как читать счётчики: `total_lines_in_file = blank_lines_in_file + data_lines_in_file`,
`data_lines_in_file = valid_records_loaded + error_records_in_file`. Отклонённые записи
не исчезают из статистики, поэтому `count_match: true` при `error_records_in_file > 0`
не выдаёт индексацию за полную — для этого есть отдельный флаг `all_input_records_valid`.
Поля `*_actual` читаются из самой коллекции Qdrant, `*_expected` — из конфига.

Что проверяется:

- `counters_consistent` / `all_input_records_valid` — сходимость счётчиков по входному файлу
  (`total = blank + data`, `data = valid + errors`) и отсутствие отклонённых записей.
- `count_match` — совпадает ли количество загруженных записей с количеством точек в Qdrant.
- `vector_size_actual` — фактическая размерность векторов в коллекции (извлекается из Qdrant).
- `distance_metric_actual` — фактическая метрика коллекции.
- `payload_fields_present` — точки поднимаются из Qdrant по `chunk_id` (`client.retrieve`)
  и проверяется, что payload существует, содержит `chunk_id`, `document_id`, `text`, `source`,
  `section`, `embedding_model`, `embedding_dimensions`, и что `payload.chunk_id` совпадает с id точки.
- `search_works` — выполняется тестовый поиск по первому вектору из файла (`query_points`,
  с fallback на устаревший `search` для клиентов старше 1.10) и проверяется, что Qdrant вернул
  хотя бы один результат.
- `errors` — список любых обнаруженных проблем (например, несовпадение размерностей, отсутствие полей).

### 2. `search_results.json` — результаты тестового поиска

Пример структуры (режим `use_existing_vectors: true`):

```json
[
  {
    "test_id": "b9a71db8-79b6-5cd5-bebe-d4b5a66781c1",
    "query_type": "existing_vector",
    "source_chunk_id": "b9a71db8-79b6-5cd5-bebe-d4b5a66781c1",
    "query_text": "ERGO — бизнес-симулятор принятия стратегических решений...",
    "self_match_found": true,
    "self_match_score": 1.0,
    "best_other_score": 0.708,
    "relevance_judgment": "high",
    "relevance_comment": "Исходный чанк найден первым (self-match), лучший сторонний результат score=0.7080 — вердикт 'high'.",
    "results": [
      {"id": "b9a71db8-...", "score": 1.0, "chunk_id": "b9a71db8-...", "document_id": "e4bf815c...", "payload": {"text": "...", "source": "...", "section": "main"}}
    ]
  }
]
```

Поля:

- `test_id` — идентификатор теста: в режиме существующих векторов это **chunk_id исходного чанка**,
  в режиме текстовых запросов — `query_1`, `query_2`, …
- `query_type` — `existing_vector` или `text_query`.
- `source_chunk_id` — id чанка, вектор которого использовался как запрос (для `text_query` — `null`).
- `query_text` — текст, которому соответствует вектор запроса (в режиме существующих векторов это
  текст того же самого чанка, а не произвольная строка из конфига).
- `self_match_found` / `self_match_score` — найден ли сам чанк по своему вектору; прямая проверка
  связи `chunk_id -> point`.
- `best_other_score` — лучший score среди результатов, исключая self-match.
- `relevance_judgment` — вердикт high/medium/low по `best_other_score`.
- `results` — массив найденных точек по убыванию score: `id`, `score`, `chunk_id`, `document_id`
  и полный payload.

Про оценку релевантности. В режиме `use_existing_vectors: true` запросом служит вектор
конкретного чанка, поэтому первым результатом всегда возвращается сам этот чанк со score ≈ 1.0.
Такой self-match ничего не доказывает о качестве поиска, но доказывает целостность связи
`chunk_id -> point` — он фиксируется отдельным полем `self_match_found`. Вердикт
`relevance_judgment` выносится по `best_other_score`, то есть по лучшему результату
**без учёта самого запроса**: `>= 0.60` — high, `>= 0.40` — medium, иначе low.

Режим `use_existing_vectors: false` считает embedding текстовых запросов из `search.test_queries`
моделью `search.model_name` (должна совпадать с моделью этапа embeddings — при расхождении
пайплайн печатает предупреждение). В этом режиме `test_id` = `query_N`, а найденный чанк
фиксируется в поле `top_chunk_id`.

### 3. `manifest.json` — информация о запуске

```json
{
  "run_id": "484e3908-d66f-463b-a4ad-066998d4f6ff",
  "timestamp": "2026-09-04T10:35:20Z",
  "config_file": "config/vector_store.yaml",
  "config": { },
  "vector_store_type": "qdrant",
  "collection_name": "rag_documents",
  "input_file": "data/embeddings/embeddings.jsonl",
  "input_total_lines": 58,
  "input_blank_lines": 0,
  "input_data_lines": 58,
  "input_valid_records": 58,
  "input_error_records": 0,
  "input_error_details": [],
  "indexed_points": 58,
  "validation": { },
  "index_is_valid": true,
  "search_tests_count": 3,
  "search_relevance_summary": [{"test_id": "b9a71db8-...", "relevance_judgment": "medium"}],
  "validation_file": "data/vector_store/validation.json",
  "search_results_file": "data/vector_store/search_results.json"
}
```

- `run_id` — уникальный идентификатор запуска (UUID4).
- `timestamp` — время запуска в UTC (`time.gmtime()`, суффикс `Z` соответствует действительности).
- `config_file` / `config` — путь к использованному конфигу и его полная копия.
- `input_total_lines`, `input_blank_lines`, `input_data_lines`, `input_valid_records`,
  `input_error_records` — счётчики по **реальному входному файлу**, а не по уже отфильтрованному
  набору. `input_error_details` содержит до 20 конкретных причин отбраковки с номерами строк.
- `indexed_points` — фактическое число точек, отправленных в Qdrant (должно совпадать
  с `input_valid_records`).
- `index_is_valid` — итоговый флаг успешности запуска.
- `validation_file` / `search_results_file` — пути к артефактам (относительные, в posix-формате).

### Как понять, что индексация прошла корректно

| Признак | Что должно быть |
|---|---|
| `index_is_valid` | `true` — агрегирующий флаг, закрывает все проверки ниже |
| `counters_consistent` | `true` — счётчики по входному файлу сходятся |
| `all_input_records_valid` | `true` — ни одна входная запись не отброшена |
| `count_match` | `true` — число точек в Qdrant равно числу валидных записей (58) |
| `vector_size_match` / `distance_metric_match` | `true` — фактические параметры коллекции совпадают с конфигом |
| `payload_fields_present` | `true` — у точек есть все обязательные поля и `payload.chunk_id == id точки` |
| `search_works` | `true` — поиск выполняется и возвращает результаты |
| `errors` | `[]` |
| `self_match_found` в `search_results.json` | `true` — чанк находится по собственному вектору |
| `relevance_judgment` | `high` или `medium` для большинства тестов (по `best_other_score`, без учёта self-match) |

Быстрая проверка одной командой:

```bash
python src/vector_pipeline.py && echo "INDEX OK"
```

Если все условия выполнены — индекс готов к использованию в RAG.

---

## 11. Тестовые вопросы

Набор в `config/questions.yaml`, состав по требованиям задания:

| # | Тип | Вопрос |
|---|---|---|
| q1 | ответ есть в документах | Сколько сегментов рынка микроэлектроники моделируется в ERGO и какие это сегменты? |
| q2 | ответ есть в документах | Какие экономические эффекты получены при внедрении интегрированного планирования в горно-металлургической компании? |
| q3 | ответ есть в документах | Какие три цифровых решения входят в цифровое ядро AIMS? |
| q4 | поиск в конкретном документе/разделе | В описании SIPS: какие критерии и ограничения заданы в примере оптимизации инвестпрограммы ж/д сети и как изменился объём электрификации путей? |
| q5 | ответа в документах нет | Сколько стоит годовая лицензия на SIPS и внедрение для компании из 500 сотрудников? |

Ожидаемые ответы указаны в том же файле в поле `expected` — по ним удобно ставить оценку
«корректно / частично корректно / некорректно» в `results/qa_report.md`.

Что считать успехом по каждому вопросу:

| # | Критерий |
|---|---|
| q1 | названо число сегментов микроэлектроники и их перечень — ровно как в документе |
| q2 | экономические эффекты приведены с исходными цифрами, без округлений и добавлений |
| q3 | названы все **три** решения цифрового ядра AIMS, не два |
| q4 | критерии, ограничения и изменение объёма электрификации; в контексте — чанки именно `02_sips...` |
| q5 | ровно «Недостаточно информации в предоставленных документах.» и ни одной цифры |

Отдельно про q5: правильное поведение — не ответ, а честное «Недостаточно информации
в предоставленных документах». Ценовой информации в базе знаний нет вообще, так что этот вопрос
проверяет дисциплину промпта и отсутствие галлюцинаций. Любая придуманная сумма означает,
что промпт не удерживает модель: поднимайте `score_threshold` с 0.0 до 0.75–0.80
или переключайтесь на Pro одной строкой в `.env`.

---

## 12. Вывод о качестве работы пайплайна (подробнее в `results/qa_report.md`)

## Использованные параметры

- **Embedding-модель:** `intfloat/multilingual-e5-small` (384 измерений, префиксы `passage:` / `query:`)
- **Vector store:** Qdrant в локальном режиме, коллекция `rag_documents`, метрика Cosine, 58 точек
- **Chunk size:** 700 символов, overlap 120, стратегия `sentence`
- **top-k:** 5
- **LLM:** yandexgpt-5-lite/latest @ https://llm.api.cloud.yandex.net/v1 (…qONL), temperature 0.0, max_tokens 700

## Сводка по вопросам

| # | Тип | Документ в топе выдачи | Лучший score | Отказ от ответа | Оценка |
|---|---|---|---|---|---|
| q1 | факт есть в документах | `01_ergo_business_simulator.pdf` | 0.8764 | нет | корректно |
| q2 | факт есть в документах | `04_ibp_integrated_business_planning.pdf` | 0.875 | нет | некорректно |
| q3 | факт есть в документах | `03_aims_asset_management.pdf` | 0.8847 | нет | корректно |
| q4 | поиск в конкретном документе/разделе | `02_sips_investment_planning.pdf` | 0.895 | нет | корректно |
| q5 | ответа в документах нет | `02_sips_investment_planning.pdf` | 0.8469 | да | корректно |


## Какие вопросы отработали хорошо

Все вопросы кроме q2

## Где пайплайн ошибся или дал неполный ответ

Вопрос q2. Проблема в настройках чанкинга. Заголовок 8.1. Горно-металлургическая компания» ушёл в предыдущий чанк, а цифры эффектов оказались во Фрагменте 1 без привязки к компании.

**Известные ограничения и что улучшать:**
- **Chunking:** текущий размер 700 символов не очень удачен, это видно на вопросе 2; 
  варианты — поднять `chunk_size` до 900-1000, хотя надо проверить уложатся ли все разделы в это количество;
  или можно перейти на разбиение по заголовкам разделов, что лучше;
- **top-k:** сейчас 5; увеличение повышает полноту, но тянет в контекст
  слабые фрагменты; увеличивать не нужно, так как на текущих вопросах верные ответы содержались в top 5;
- **Metadata:** `document_name` / `category` / `version` уже доходят до payload;
  следующий шаг — фильтр `query_filter` по `document_name` для вопросов вида
  «в документе X…», чтобы retrieval не смешивал продукты; это видно в вопросе 4, где ссылка на конкретный документ;
- **Документы:** база из 5 обезличенных PDF описывает продукты, но не содержит
  коммерческих предложений с ценами; вопрос 5 как раз это проверял;
- **Гибридный поиск:** гипотетически, чисто векторный retrieval слабее на точных терминах
  и аббревиатурах — связка BM25 + dense закрывает этот класс запросов.
- **Prompt:** правило про отказ отработало (вопрос 5 — корректный ответ, без выдуманных сумм),
  но на вопросе 2 сработало избыточно: цифры эффектов присутствовали во Фрагменте 1, однако
  модель не связала их с компанией из вопроса и сообщила о недостатке информации.
  Можно добавить «если в контексте есть числовые показатели,
  относящиеся к теме вопроса, приведи их; отказ уместен только тогда,
  когда данных нет вовсе»; «перечисли все пункты, встречающиеся в контексте, не сокращая список»

---

## 13. Что добавлено и изменено относительно прошлой версии пайплайна

Новое:
- `src/retriever.py` — retrieval по вопросу с параметром top-k и проверкой совместимости модели и коллекции;
- `src/context_builder.py` — сборка контекста с явным разделением фрагментов и лимитами по длине;
- `src/llm_client.py` — клиент на stdlib с двумя провайдерами: OpenAI-совместимый
  (OpenAI / Yandex Cloud / Ollama / DeepSeek) и нативный YandexGPT; ключи только из `.env`;
- `src/rag_answer.py` — CLI: вопрос → ответ + использованные фрагменты (`--top-k`, `--no-llm`, `--json`);
- `src/rag_eval.py` — прогон 5 тестовых вопросов, артефакты `results/qa_results.json` и `results/qa_report.md`;
- `src/llm_check.py` — проверка доступа к LLM без Qdrant и индекса;
- `config/rag.yaml`, `config/questions.yaml`, `.env.example`, `.gitignore`;
- `tools/md_to_pdf.py` — опциональная пересборка базы знаний в PDF;
- `data/raw/metadata.yaml` — прикладные метаданные документов.

Изменено:
- `src/structurer.py` — подтягивает `document_name` / `category` / `version` / `source_type` из sidecar;
- `src/chunk_pipeline.py` — пробрасывает все метаданные документа в чанк, а не фиксированный список полей;
- `src/cleaner.py` — опция `unwrap_pdf_line_breaks` для склейки строк, разорванных вёрсткой PDF;
- `src/embedder.py` — поддержка префиксов E5, устойчивое определение размерности, offline-режим
  для смоук-теста;
- `config/embeddings.yaml` — `all-MiniLM-L6-v2` → `intfloat/multilingual-e5-small` (размерность
  та же, 384, поэтому `vector_size` не менялся);
- `config/chunking.yaml` — chunk_size 300 → 700, overlap 50 → 120 (короткие чанки рвали смысловые блоки);
- `config/config.yaml` — добавлены `metadata_file` и `unwrap_pdf_line_breaks`;
- `config/vector_store.yaml` — тестовые запросы под новую базу знаний;
- `requirements.txt` — `sentence-transformers>=2.7`, `qdrant-client>=1.10,<2.0`, добавлен `reportlab`.

Подключение YandexGPT 5 Lite:
- `config/rag.yaml` — модель `yandexgpt-lite/latest` → **`yandexgpt-5-lite/latest`**;
  `top_k` 3 → 5; `max_context_chars` 6000 → 12000; `max_chars_per_chunk` 1200 → 2000
  (окно модели 32 768 токенов это позволяет, прежние лимиты выставлялись «вслепую»);
- `src/llm_client.py` — схема авторизации выбирается автоматически (`Api-Key` для ключа `AQVN...`,
  `Bearer` для IAM-токена `t1....`) вместо жёсткого `Bearer`, который давал 401 на API-ключе;
- `src/llm_client.py` — для `provider: "yandex"` эндпоинт больше не берётся из `llm.base_url`:
  нативный API живёт по пути `.../foundationModels/v1/completion`, и подстановка `.../v1`
  из конфига OpenAI-совместимого слоя приводила к 404;
- `src/llm_client.py` — развёрнутые подсказки по кодам 400/401/403/404/429;
- восстановлены `data/raw/metadata.yaml` и `.gitignore`, отсутствовавшие в поставке с PDF.

---

## 14. Краткий вывод (пункт 10 задания)

Готовый вывод генерируется в `results/qa_report.md` при запуске `python src/rag_eval.py`:
раздел «Краткий вывод» в конце отчёта. 

Зафиксированные параметры (дублируются в шапке отчёта):

| Параметр | Значение |
|---|---|
| Embedding-модель | `intfloat/multilingual-e5-small`, 384 измерения, префиксы `passage:` / `query:` |
| Vector store | Qdrant (local mode), коллекция `rag_documents`, метрика Cosine, 58 точек |
| Chunk size / overlap | 700 / 120 символов, стратегия `sentence` |
| top-k | 5 |
| score_threshold | 0.0 |
| Лимиты контекста | 12 000 символов всего, 2 000 на фрагмент |
| LLM | YandexGPT 5 Lite (`yandexgpt-5-lite/latest`), temperature 0, max_tokens 700 |
| Промпт | `config/rag.yaml`, ключ `prompt.system` — полный текст печатается в отчёт |

---

## 15. Цикл оптимизации: baseline → улучшения → сравнение

См. `results/OPTIMIZATION_REPORT.md`
