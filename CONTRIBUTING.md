# Contributing to AI Mentor

Спасибо за интерес к проекту! Полностью офлайн-приложение для начинающих
LLM-инженеров: RAG + локальный Qdrant + GGUF-инференс на NVIDIA GPU.

**Вопросы и баги:** fedorfesarov@gmail.com или
[Issues](https://github.com/FesarovFedor/ai-mentor/issues).

## Сборка из исходников

Требования: Windows 10/11 x64, Rust (stable, MSVC), Node не нужен,
для CUDA-сборки llama.cpp — CUDA Toolkit 12.x (nvcc) и libclang.

```powershell
git clone https://github.com/FesarovFedor/ai-mentor.git
cd ai-mentor
cargo tauri build
```

`build.rs` сам скачает бинарные зависимости (CUDA runtime, cuBLAS, ONNX
Runtime, qdrant.exe) с официальных источников, проверит SHA-256 и положёт
в `src-tauri/resources/` и `src-tauri/binaries/`. Кэш — `target/deps_cache/`,
повторные сборки ничего не качают.

### libclang (bindgen для llama-cpp-sys-2)

`llama-cpp-sys-2` генерирует биндинги через bindgen, которому нужен libclang.
Файл `.cargo/config.toml` (машинно-специфичный путь) не хранится в git —
скопируйте шаблон и укажите свой путь:

```powershell
Copy-Item .cargo/config.toml.example .cargo/config.toml
# затем пропишите LIBCLANG_PATH на каталог bin/ вашего libclang.dll
```

Варианты: portable LLVM (распакованный релиз llvm-project) или системная
установка LLVM; если libclang доступен в PATH, секцию `[env]` можно опустить.

Данные базы знаний (векторный стор, тексты чанков и кэш embedding-модели
fastembed) в git не хранятся: build.rs копирует их из локального дерева
(`tools_bin/qdrant_server/storage`, `../kb_chunks`, `.models/fastembed`).
Для сборки без них задайте пути (или пустые каталоги) через env
`QDRANT_STORAGE_SRC` / `KB_CHUNKS_SRC` / `FASTEMBED_SRC` — приложение
соберётся, но RAG потребует собственной базы, а embedding-модель
(~465 МБ) будет скачана с HuggingFace при первом запуске.

## Запуск в dev-режиме

```powershell
cargo tauri dev
```

## Тесты

```powershell
cargo test --workspace   # unit-тесты чистых функций ядра
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
cargo run --bin retrieval_test   # требует запущенный Qdrant
```

Релизные проверки: `cargo tauri build` должен собирать MSI, приложение —
поднимать Qdrant sidecar и отвечать на вопрос из базы знаний.

## Стиль коммитов

- `feat:`, `fix:`, `docs:`, `chore:`, `build:` — обычные префиксы.
- Один коммит — одна логическая перемена.
- Коммиты должны быть подписаны вашим именем; для атрибуции на GitHub
  используйте email, привязанный к аккаунту.

## Pull Requests

1. Форк → ветка `feature/<краткое-имя>`.
2. `cargo fmt` + `cargo clippy -- -D warnings` без ошибок.
3. Описание: что меняется и зачем; скриншоты для изменений UI.
4. Шаблон PR заполнится автоматически.

## Что менять осторожно

См. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — раздел «Ключевые
инварианты (не сломать)»: паритет промпта с эталоном, детерминизм seed,
один UTF-8 декодер на генерацию, BOS, XSS-политика фронтенда.
