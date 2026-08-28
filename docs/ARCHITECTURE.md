# Архитектура AI Mentor

Локальный ИИ-наставник для начинающих инженеров по LLM: полностью офлайн,
RAG поверх локального Qdrant, инференс GGUF через llama.cpp (CUDA).
Единственный внешний сетевой вызов — скачивание GGUF-модели по кнопке
на первом экране (встроенный загрузчик).

## Структура workspace

```
mentor-core (src/)          ядро, без GUI-зависимостей
├── config.rs               config.toml: загрузка, пути, toml_edit-запись
│                           с сохранением комментариев
├── rag.rs                  Qdrant (gRPC) + fastembed (ONNX e5);
│                           format_context() — единый форматтер фрагментов
├── inference.rs            llama-cpp-2: GGUF, полный GPU-оффлоад,
│                           KV Q4_0 + FlashAttention, генерация
│                           токен-за-токеном (generate_with_callback)
├── generator.rs            промпт (system+RAG+вопрос, ChatML/raw),
│                           ThinkRouter — инкрементальная разметка стрима
│                           thinking/answer при разрезе <think> между токенами
├── downloader.rs           HTTP-загрузчик GGUF: Range-докачка (.part +
│                           meta.json), SHA-256, магия GGUF, отмена,
│                           read/total таймауты
└── bin/                    CLI: retrieval_test, gen_bench, inference_smoke,
                            download_test

src-tauri/                  Tauri v2 оболочка
├── src/lib.rs              AppState, команды, события, provisioning, pre-flight
├── src/qdrant.rs           жизненный цикл Qdrant sidecar
└── build.rs                tauri-build + авто-загрузка бинарных зависимостей

frontend/                   статические HTML/CSS/JS, терминальная эстетика
```

## Поток запроса

```
вопрос → embed(e5, "query: ") → Qdrant top_k → format_context
      → ChatML-промпт → llama.cpp (GPU) → [стрим gen-token: think|answer]
      → ChatReply{thinking, answer, sources, prompt} → канонический рендер
```

Генерация стримится во вебвью событием `gen-token` и параллельно копится в
снимок `gen_stream` (команда `get_gen_progress`) — запасной канал опроса,
тот же паттерн, что у прогресса скачивания.

## Ключевые инварианты (не сломать)

- **Паритет промпта с Python-эталоном**: SYSTEM_PROMPT, формат фрагментов и
  e5-префикс `query: ` зафиксированы; менять только через
  `rag::format_context` / `generator::build_prompt_parts`.
- **Детерминизм**: `DEFAULT_SEED=1234` для воспроизводимости бенчей.
- **Один декодер UTF-8 на генерацию** (кириллица рвётся между токенами).
- **BOS**: у nanbeige `<|im_start|>` и есть BOS — второй не добавлять
  (детект в `Inference::generate_with_callback`).
- **n_batch >= длина промпта** (иначе GGML_ASSERT внутри llama.cpp);
  клампы бюджета — в `generate()`.
- **Комментарии config.toml сохраняются** при записи (toml_edit).
- **XSS**: пользовательский ввод / статусные тексты / источники — только
  `textContent`/`createTextNode`. Модельный текст (thinking, answer) —
  Markdown-пайплайн: `markdown-it` (html:false, breaks:true) +
  `DOMPurify.sanitize` → `innerHTML`. Это единственная точка HTML-вставки.
  Локальные копии библиотек в `frontend/vendor/` (CSP `default-src 'self'`).

## Решения, принятые в L3–L5

### L3 — единый рендерер стрима и финала
Два пути построения DOM ответа (стрим в `.md-body`, финал в `msg__text`)
давали разный вид. Единая фабрика `buildAiText()` в `frontend/main.js`:
и `startLiveMessage`, и `addAiAnswer` строят одинаковую структуру
(think-details + .md-body), рендер везде `applyModelHtml`.

### L4 — стабильность (P0/P1 из аудита)
- **LlamaBackend во владении AppState** (`Arc<LlamaBackend>`), а не в
  `static OnceLock`: статика дропается недетерминированно, CUDA-контекст
  не освобождал VRAM. `init_backend()` — единственная точка
  `LlamaBackend::init()`; Tauri гарантирует Drop состояния при выходе.
- **Таймауты downloader**: `read_timeout(60s)` + общий `timeout(7200s)` —
  без них молчащий сервер вешал `chunk().await` навечно.
- **parking_lot вместо std::sync::Mutex** для коротких секций (нет
  «отравления»); `tokio::sync::Mutex` только там, где гвард держится
  через `.await` (rag-поиск, llm_load_lock).
- **Async-инициализация RAG**: ONNX-инициализация и чтение `*.jsonl`
  базы знаний — в `tokio::task::spawn_blocking`, main-поток не блокируется.

### L5 — self-contained установка
- **Бинарные зависимости не в git**: build.rs скачивает CUDA runtime и
  cuBLAS из официального redist NVIDIA (SHA-256 из официального манифеста),
  ONNX Runtime и Qdrant — с GitHub releases (pinned SHA-256); кэш в
  `target/deps_cache/`.
- **Qdrant sidecar**: externalBin в бандле; запуск `std::process` (spawn
  через shell-плагин привязан к event loop и дедлочится при init до окна);
  `CREATE_NO_WINDOW`.
- **Динамический порт**: connect-пробой (bind-проверка на Windows
  «пропускает» порт, занятый на 0.0.0.0); адрес подменяется только в
  памяти, config.toml на диске не меняется; готовность — polling `/readyz`
  (у Qdrant 1.19 `/readiness` не существует), таймаут 30 с.
- **Все пользовательские данные в `%APPDATA%\ai-mentor`**: конфиг, модели,
  кэш эмбеддингов, стор Qdrant. Первый запуск разворачивает их из бандла
  (эти данные мейнтейнера в git не хранятся — копируются build.rs из
  локального дерева проекта).
- **Pre-flight драйвера**: проверка `nvcuda.dll` до инициализации инференса;
  без NVIDIA-драйвера — понятная ошибка со ссылкой на nvidia.com/drivers
  (сознательная деградация «блокировка», не CPU: полный оффлоад на CPU —
  минуты на токен).
