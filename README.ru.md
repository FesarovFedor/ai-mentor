# AI Mentor

[EN](README.md) | **RU**

![Release](https://img.shields.io/github/v/release/FesarovFedor/ai-mentor)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Rust](https://img.shields.io/badge/Rust-stable-orange.svg)
![Platform](https://img.shields.io/badge/platform-Windows%2010%2F11%20x64-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-12-76B900.svg)

Полностью офлайн десктоп-приложение «ИИ-наставник» для начинающих
LLM-инженеров: RAG поверх локального Qdrant + инференс GGUF через llama.cpp
на вашей NVIDIA-видеокарте. Из машины не уходит ничего: ни вопросы, ни база
знаний, ни генерация.

![AI Mentor — чат с готовым ответом](assets/hero.png)

## ✨ Возможности

- **Установка в один клик** — внутри MSI уже всё: CUDA runtime, cuBLAS,
  ONNX Runtime, Qdrant и база знаний. Единственное скачивание — GGUF-модель
  одной кнопкой через встроенный загрузчик.
- **Ноль облака** — эмбеддинги, поиск и генерация на 100% локальные.
- **Стриминг рассуждений** — блок `<think>` модели показывается живьём в
  сворачиваемой панели; финальный ответ печатается по токенам.

## 🖥 Технологии

| Слой | Технология |
| --- | --- |
| Оболочка | Tauri v2 (Rust + системный WebView2) |
| Ядро | Rust-крейт `mentor-core` |
| Инференс | llama.cpp (`llama-cpp-2`) на CUDA 12, KV Q4_0 + FlashAttention |
| Векторная БД | Qdrant 1.19 (gRPC), работает как sidecar приложения |
| Эмбеддинги | intfloat/multilingual-e5-small через fastembed (ONNX) |
| Фронтенд | Vanilla HTML/CSS/JS, markdown-it + DOMPurify (CSP `default-src 'self'`) |

## 📦 Установка

### Пользователям

1. Скачайте `ai-mentor-1.1.0-x64.msi` со страницы
   [Releases](https://github.com/FesarovFedor/ai-mentor/releases/latest).
2. Установите (внутри уже всё: не нужен ни CUDA Toolkit, ни ручной Qdrant,
   ни правка конфигов).
3. Запустите **AI Mentor**. При первом запуске приложение само поднимет
   Qdrant, развернёт данные в `%APPDATA%\ai-mentor` и предложит скачать
   GGUF-модель — URL уже заполнен, нажмите кнопку скачивания.

### Разработчикам

```powershell
git clone https://github.com/FesarovFedor/ai-mentor.git
cd ai-mentor
cargo tauri build
```

`build.rs` сам скачает бинарные зависимости (официальные источники,
проверка SHA-256, кэш в `target/deps_cache/`). Подробности — в
[CONTRIBUTING.md](CONTRIBUTING.md).

## 📋 Системные требования

- Windows 10/11 x64
- NVIDIA GPU с 8+ ГБ VRAM и свежим драйвером — **CUDA Toolkit НЕ нужен**
  (приложение несёт CUDA runtime с собой; от драйвера нужен только
  `nvcuda.dll`)
- ~6 ГБ диска: приложение (~900 МБ) + модель (~2,7 ГБ) + запас

Если драйвер NVIDIA не обнаружен, инференс блокируется с понятным
сообщением и ссылкой на [nvidia.com/drivers](https://www.nvidia.com/drivers) —
молчаливого «фоллбэка на CPU» (минуты на токен) нет.

## 🏗 Архитектура

Rust-workspace: `mentor-core` (конфиг, RAG, инференс, генератор, загрузчик)
+ оболочка `src-tauri` (команды, жизненный цикл Qdrant sidecar,
provisioning первого запуска, pre-flight драйвера). Qdrant стартует
автоматически на свободном порту и останавливается при выходе.

Подробно: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — карта модулей,
поток запроса, ключевые инварианты и архитектурные решения L3–L5.

## 📊 Замеры

RTX 4070 Laptop (8 ГБ VRAM), nanbeige4.2-3B Q4_K_M, seed 1234:

| Метрика | Значение |
| --- | --- |
| Скорость генерации | ~41 токен/с (GPU, полный оффлоад) |
| Пик VRAM | ~3,6 ГиБ (n_ctx 12288, KV Q4_0 + FlashAttention) |
| Окно контекста | 12 288 токенов (RAG-промпт ~2 750 + бюджет 5 500) |
| Загрузка модели | ~1,6 с (mmap) |

## 💾 Хранение данных

Все пользовательские данные — в `%APPDATA%\ai-mentor\`:

| Путь | Содержимое |
| --- | --- |
| `config.toml` | конфиг пользователя (путь к модели, параметры генерации) |
| `.models\downloaded\` | скачанные GGUF-модели |
| `.models\fastembed\` | кэш модели эмбеддингов |
| `qdrant\storage\` | векторная база (база знаний) |
| `kb_chunks\` | тексты чанков для RAG-контекста |

**Сброс**: закройте приложение и удалите `%APPDATA%\ai-mentor\` — при
следующем запуске всё развернётся заново из установщика (модель придётся
скачать повторно).

## 🔧 Решение проблем

**«NVIDIA-драйвер не найден (nvcuda.dll)»**
Установите/обновите драйвер с
[nvidia.com/drivers](https://www.nvidia.com/drivers) и перезапустите
приложение. CUDA Toolkit никогда не требуется.

**«Qdrant не поднялся» / ошибки порта**
Приложение выбирает свободный порт автоматически (диапазон начинается с
6333/6334). Если конфликт всё же случился — завис посторонний локальный
Qdrant: закройте его или перезагрузитесь. Выбранный порт виден в
статус-баре.

**Загрузка модели оборвалась**
Нажмите «скачать» ещё раз — загрузчик продолжит с последнего байта
(`.part` + метаданные), проверит SHA-256 (если задан) и магию GGUF.

**Загрузка или генерация «зависли»**
Загрузка обрывается по таймаутам (60 с сетевой тишины, 2 часа общий бюджет).
Генерация стримится по токенам — если пульсирует панель размышлений,
модель работает.

**Приложение вообще не запускается**
Загляните в `%TEMP%\ai-mentor-init-error.log` — ошибки старта (включая
сбой Qdrant) пишутся туда и показываются диалогом.

## 🤝 Участие в разработке

[CONTRIBUTING.md](CONTRIBUTING.md). Баги и идеи — через
[Issues](https://github.com/FesarovFedor/ai-mentor/issues) (шаблоны
прилагаются).

## 📄 История изменений

[CHANGELOG.md](CHANGELOG.md) (формат Keep a Changelog).

## ⚖ Лицензия

[MIT](LICENSE) © 2026 Fedor Fesarov

## 🙏 Благодарности

- [llama.cpp](https://github.com/ggml-org/llama.cpp) и Rust-байндинги
  [llama-cpp-2](https://crates.io/crates/llama-cpp-2)
- [Qdrant](https://qdrant.tech/) — локальная векторная БД
- [fastembed-rs](https://github.com/Anush008/fastembed-rs) — ONNX-эмбеддинги
- [Tauri](https://tauri.app/) — оболочка приложения
- [markdown-it](https://github.com/markdown-it/markdown-it) +
  [DOMPurify](https://github.com/cure53/DOMPurify) — безопасный рендер

## 📮 Контакты

**fedorfesarov@gmail.com** — вопросы, отзывы, баг-репорты
(или [issue](https://github.com/FesarovFedor/ai-mentor/issues)).
