//! Tauri-команды: мост между фронтендом (frontend/) и ядром mentor-core.
//!
//! Этап L5: self-contained установка — Qdrant поднимается как sidecar
//! (модуль qdrant), данные живут в AppData, порт выбирается динамически.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

pub mod qdrant;

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use parking_lot::Mutex;
use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager, State};
// AsyncMutex — только для гвардов, удерживаемых через .await (rag-поиск,
// сериализация загрузки модели). Короткие синхронные секции — parking_lot.
use tokio::sync::Mutex as AsyncMutex;

use mentor_core::config::{save_generation_fields, save_string_field, AppConfig};
use mentor_core::downloader::{filename_from_url, CancelToken, DownloadSpec, Downloader};
use mentor_core::generator::{generate_response_streaming, HistoryTurn, StreamKind};
use mentor_core::inference::{gguf_display_name, Inference, LlamaBackend};
use mentor_core::rag::{format_context, Rag, SearchHit};

/// Разделяемое состояние приложения: RAG-ядро + конфиг + загрузчик модели.
///
/// Владелец единственного LlamaBackend: Arc дублируется в каждую Inference,
/// поэтому бэкенд живёт, пока жива хоть одна модель, и гарантированно
/// дропается вместе с AppState при завершении приложения (RunEvent::Exit) —
/// CUDA-контекст освобождает VRAM (фикс P0 утечки из аудита, этап L4).
pub struct AppState {
    /// LlamaBackend (единственная точка init_backend()). Option из-за F-025:
    /// на машине без NVIDIA-драйвера бэкенд не инициализируется вовсе —
    /// иначе CUDA-инициализация падает раньше дружелюбного сообщения о
    /// драйвере. None -> инференс блокируется (ensure_llm), RAG/скачивание
    /// работают.
    pub backend: Option<Arc<LlamaBackend>>,
    /// RAG держится через .await (поиск по Qdrant) — AsyncMutex.
    pub rag: AsyncMutex<Rag>,
    /// Конфиг под мьютексом: обновляется после выбора/скачивания модели.
    /// Короткие секции без .await — parking_lot.
    pub cfg: Mutex<AppConfig>,
    /// Путь к config.toml в AppData (запись model_path после выбора/скачивания).
    pub cfg_path: PathBuf,
    /// Загруженная LLM и путь, по которому она загружена (для перезагрузки
    /// при смене model_path). Модель лениво грузится на первом вопросе.
    pub llm: Mutex<Option<(PathBuf, Arc<Inference>)>>,
    /// Сериализует загрузку/перезагрузку модели: без него два параллельных
    /// send_message при смене модели загрузили бы её дважды. Гвард держится
    /// через .await (там грузится модель) — AsyncMutex.
    pub llm_load_lock: AsyncMutex<()>,
    /// Отмена активной загрузки.
    pub download_cancel: Mutex<CancelToken>,
    /// Рдёт ли сейчас загрузка (защита от повторного запуска).
    pub download_active: AtomicBool,
    /// Последний снимок прогресса загрузки для опроса фронтом (страховка на
    /// случай, если шина событий недоступна во вебвью). Короткие секции,
    /// нужен и из синхронного колбэка — parking_lot::Mutex.
    pub download_progress: Mutex<Option<DownloadEvent>>,
    /// Накопленный поток генерации (thinking/answer) для опроса фронтом —
    /// тот же запасной канал, что у download_progress. Абсолютные значения:
    /// фронт сверяет длины и дозабирает недостающее.
    pub gen_stream: Mutex<GenStreamSnapshot>,
    /// Флаг отмены активной генерации (F-002, аналог CancelToken для
    /// скачивания): проверяется в колбэке каждого токена.
    pub gen_cancel: AtomicBool,
    /// Причина неудачной предзагрузки модели (F-004): без этого текст уходил
    /// только в eprintln, которого в release-сборке не видно, и пользователь
    /// узнавал причину лишь при первой отправке сообщения.
    pub llm_error: Mutex<Option<String>>,
    /// Дочерний процесс Qdrant sidecar: останавливается в RunEvent::Exit.
    pub qdrant: qdrant::QdrantProc,
    /// Pre-flight (Этап L5, шаг 6): есть ли NVIDIA-драйвер в системе.
    /// false -> инференс блокируется с дружелюбной ошибкой (деградация
    /// выбрана как "блокировка", не CPU: полный оффлоад 3B-модели на CPU
    /// занял бы минуты на токен и выглядел бы как зависание).
    pub gpu_ready: AtomicBool,
}

#[derive(Serialize)]
pub struct ChatReply {
    pub answer: String,
    /// Ход рассуждений модели (<think>…</think>); пуст у нон-reasoning
    /// моделей. Фронтенд рендерит его отдельным сворачиваемым блоком.
    pub thinking: String,
    pub sources: Vec<SearchHit>,
    /// Полный промпт, который получил(а бы) LLM.
    pub prompt_for_model: String,
}

#[derive(Serialize)]
pub struct StatusInfo {
    pub qdrant_url: String,
    pub collection: String,
    pub points: u64,
    pub top_k: u32,
    pub embedding_model: String,
    pub model_path_set: bool,
    /// true — реальная LLM загружена (переименование F-018: старое имя
    /// generator_stub осталось с этапа заглушки и было семантически
    /// инвертировано).
    pub llm_loaded: bool,
    /// Причина, по которой модель не загрузилась (F-004): ошибка предзагрузки
    /// из init_state; None — если предзагрузка не проводилась/прошла успешно.
    pub llm_error: Option<String>,
}

/// Состояние модели для стартового экрана (читает конфиг с диска — всегда свежий).
#[derive(Serialize, Clone)]
pub struct ModelStatus {
    /// Модель готова: путь задан и файл существует.
    pub found: bool,
    pub path: String,
    /// Плейсхолдер URL из config.toml (пуст -> скачивание недоступно).
    pub download_url: String,
    /// Задана ли контрольная сумма для проверки после скачивания.
    pub sha256_set: bool,
}

/// Событие прогресса загрузки в вебвью (канал "download-progress").
#[derive(Serialize, Clone)]
pub struct DownloadEvent {
    pub downloaded: u64,
    /// 0 = размер неизвестен, процент не посчитать.
    pub total: u64,
    pub resumed_from: u64,
    pub done: bool,
    pub error: Option<String>,
}

/// Событие потоковой генерации в вебвью (канал "gen-token"): один кусок
/// текста, размеченный по блоку (thinking/answer) на бэкенде.
#[derive(Serialize, Clone)]
pub struct GenTokenEvent {
    /// "think" | "answer"
    pub kind: String,
    pub text: String,
}

/// Снимок накопленного стрима генерации (запасной канал для опроса).
#[derive(Serialize, Clone, Default)]
pub struct GenStreamSnapshot {
    pub thinking: String,
    pub answer: String,
    /// Генерация завершена (фронт может прекратить опрос).
    pub done: bool,
}

/// Снимок статуса модели из загруженного конфига (F-015: единый источник —
/// AppState.cfg; перечитывание с диска нужно только после внешней правки
/// model_download_url, что делает start_model_download/run_download).
fn model_status_from(cfg: &AppConfig) -> ModelStatus {
    ModelStatus {
        found: cfg.model_ready(),
        path: cfg.model_file_path().to_string_lossy().into_owned(),
        download_url: cfg.model_download_url.clone(),
        sha256_set: !cfg.model_sha256.trim().is_empty(),
    }
}

/// Каталог пользовательских данных: %APPDATA%\ai-mentor (Этап L5, шаг 7).
/// Все пользовательские данные (config.toml, модели, кэш эмбеддингов,
/// storage векторной БД) живут здесь — каталог установки может быть
/// read-only (Program Files).
fn app_data_dir() -> PathBuf {
    let base = std::env::var("APPDATA")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir());
    base.join("ai-mentor")
}

/// Первый запуск: если config.toml ещё нет в AppData — разворачиваем его из
/// бандла (пути kb_chunks переписываются с ../kb_chunks на AppData) и
/// копируем данные базы знаний (тексты чанков + векторный стор с
/// коллекцией mentor_kb). Повторные запуски ничего не перезатирают.
fn provision_first_run(resource_dir: &Path) -> Result<PathBuf, anyhow::Error> {
    let dir = app_data_dir();
    fs::create_dir_all(&dir).with_context(|| format!("mkdir {}", dir.display()))?;
    let cfg_path = dir.join("config.toml");
    if !cfg_path.exists() {
        let template_path = resource_dir.join("defaults").join("config.toml");
        let template = fs::read_to_string(&template_path)
            .with_context(|| format!("в бандле нет шаблона конфига {}", template_path.display()))?;
        // kb_chunks в бандле лежат рядом с config.toml в AppData.
        let rewritten = template.replace("../kb_chunks", "kb_chunks");
        fs::write(&cfg_path, rewritten)
            .with_context(|| format!("запись {}", cfg_path.display()))?;
    }
    copy_missing(
        &resource_dir.join("kb_chunks"),
        &dir.join("kb_chunks"),
        "тексты чанков",
    )?;
    copy_missing(
        &resource_dir.join("qdrant-storage"),
        &dir.join("qdrant").join("storage"),
        "векторный стор Qdrant",
    )?;
    // Кэш embedding-модели (F-026): разворачивается из бандла, чтобы первый
    // запуск не качал ~465 МБ с HuggingFace (обещание «Zero cloud» в README).
    copy_missing(
        &resource_dir.join("fastembed-cache"),
        &dir.join(".models").join("fastembed"),
        "кэш embedding-модели fastembed",
    )?;
    Ok(cfg_path)
}

fn copy_missing(src: &Path, dst: &Path, what: &str) -> Result<(), anyhow::Error> {
    if !src.exists() {
        anyhow::bail!("{what}: в бандле нет {}", src.display());
    }
    if dst.exists() {
        return Ok(());
    }
    copy_tree(src, dst).with_context(|| format!("копирование {what} в {}", dst.display()))
}

fn copy_tree(src: &Path, dst: &Path) -> std::io::Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let to = dst.join(entry.file_name());
        if entry.path().is_dir() {
            copy_tree(&entry.path(), &to)?;
        } else {
            fs::copy(entry.path(), &to)?;
        }
    }
    Ok(())
}

/// Перечитывает config.toml с диска в AppState, СОХРАНЯЯ динамический
/// адрес Qdrant: порт выбирается в рантайме и на диск не пишется (L5,
/// шаг 5), поэтому после каждого reload его нужно накладывать заново.
fn reload_cfg_preserving_port(app: &AppState) -> Result<(), String> {
    let mut fresh = AppConfig::load(&app.cfg_path).map_err(|e| format!("config.toml: {e:#}"))?;
    fresh.qdrant.url = app.cfg.lock().qdrant.url.clone();
    *app.cfg.lock() = fresh;
    Ok(())
}

/// Pre-flight (шаг 6): проверяем наличие NVIDIA-драйвера (nvcuda.dll —
/// клиентская библиотека CUDA ставится ТОЛЬКО вместе с драйвером NVIDIA).
pub fn gpu_driver_available() -> bool {
    let system32 = std::env::var("SystemRoot").map_or_else(
        |_| PathBuf::from(r"C:\Windows"),
        |root| PathBuf::from(root).join("System32"),
    );
    system32.join("nvcuda.dll").is_file()
}

/// Каталог ресурсов бандла: Tauri кладёт ресурсы (DLL, qdrant-storage,
/// kb_chunks, defaults/) рядом с основным бинарем — и в MSI, и при dev-запуске.
fn resource_dir() -> Result<PathBuf, anyhow::Error> {
    let exe = std::env::current_exe().context("не удалось определить каталог приложения")?;
    exe.parent()
        .with_context(|| format!("exe без родительского каталога: {}", exe.display()))
        .map(Path::to_path_buf)
}

/// Рнициализация состояния: первый запуск -> Qdrant sidecar -> RAG -> модель.
/// Вызывается из setup-хука ДО создания окна: пользователь не видит
/// "зависшего" окна, а любые ошибки показываются диалогом (не молча).
async fn init_state() -> Result<AppState, String> {
    let resource_dir = resource_dir().map_err(|e| format!("каталог ресурсов: {e:#}"))?;
    // Тяжёлое копирование данных первого запуска (до ~600 МБ стор Qdrant) —
    // в блокирующем пуле, main-поток не занят файловым I/O.
    let res_for_provision = resource_dir.clone();
    let cfg_path =
        tauri::async_runtime::spawn_blocking(move || provision_first_run(&res_for_provision))
            .await
            .map_err(|e| format!("поток первого запуска: {e}"))?
            .map_err(|e| format!("первый запуск: {e:#}"))?;

    let mut cfg = AppConfig::load(&cfg_path).map_err(|e| format!("config.toml: {e:#}"))?;

    // Pre-flight драйвера: модель грузим только если есть NVIDIA.
    // F-025: init_backend вызывается ТОЛЬКО при gpu_ready — CUDA-инициализация
    // на машине без nvcuda.dll падала бы техническим диалогом раньше
    // задокументированного дружелюбного сообщения о драйвере.
    let gpu_ready = gpu_driver_available();
    let backend = if gpu_ready {
        Some(Arc::new(mentor_core::inference::init_backend().map_err(
            |e| format!("инициализация llama.cpp backend: {e:#}"),
        )?))
    } else {
        None
    };

    // Qdrant sidecar: динамический порт + AppData-стор; ждём readiness.
    let qproc = qdrant::QdrantProc::new();
    let storage = app_data_dir().join("qdrant").join("storage");
    let sidecar_exe = qdrant::sidecar_exe_path().map_err(|e| format!("Qdrant: {e:#}"))?;
    let (http_port, grpc_port) =
        qdrant::start_qdrant(&qproc, &sidecar_exe, &storage).map_err(|e| {
            qdrant::stop_qdrant(&qproc);
            format!("запуск Qdrant: {e:#}")
        })?;
    qdrant::wait_for_ready(http_port, Duration::from_secs(30))
        .await
        .map_err(|e| {
            qdrant::stop_qdrant(&qproc);
            format!("Qdrant: {e:#}")
        })?;
    // Порт — величина времени исполнения: переопределяем адрес ТОЛЬКО в
    // памяти (config.toml на диске остаётся с дефолтным 6334).
    cfg.qdrant.url = format!("http://127.0.0.1:{grpc_port}");

    let rag = Rag::new(cfg.clone())
        .await
        .map_err(|e| format!("инициализация RAG: {e:#}"))?;

    let mut llm_error: Option<String> = None;
    let llm = match backend.as_ref() {
        Some(backend) if cfg.model_ready() => {
            let path_str = cfg.model_file_path().to_string_lossy().into_owned();
            let backend = backend.clone();
            match tauri::async_runtime::spawn_blocking(move || {
                mentor_core::inference::load_model(&path_str, &backend)
            })
            .await
            .map_err(|e| format!("поток предзагрузки модели: {e}"))?
            {
                Ok(inf) => Some((cfg.model_file_path(), Arc::new(inf))),
                Err(e) => {
                    // F-004: причина неудачной предзагрузки должна быть видна
                    // пользователю сразу (в release eprintln никуда не пишется).
                    let text = format!("{e:#}");
                    eprintln!("предзагрузка модели не удалась: {text}");
                    llm_error = Some(text);
                    None
                }
            }
        }
        _ => None,
    };
    Ok(AppState {
        backend,
        rag: AsyncMutex::new(rag),
        cfg: Mutex::new(cfg),
        cfg_path,
        llm: Mutex::new(llm),
        llm_load_lock: AsyncMutex::new(()),
        download_cancel: Mutex::new(CancelToken::new()),
        download_active: AtomicBool::new(false),
        download_progress: Mutex::new(None),
        gen_stream: Mutex::new(GenStreamSnapshot::default()),
        gen_cancel: AtomicBool::new(false),
        llm_error: Mutex::new(llm_error),
        qdrant: qproc,
        gpu_ready: AtomicBool::new(gpu_ready),
    })
}

/// Возвращает загруженную модель, при необходимости грузит её в фоне
/// (лениво на первом вопросе) или перезагружает после смены model_path.
/// Вся последовательность "проверка -> выгрузка -> загрузка -> запись"
/// держит llm_load_lock: параллельные запросы не могут загрузить модель
/// дважды или увидеть промежуточное None.
async fn ensure_llm(app: &Arc<AppState>, model_path: PathBuf) -> Result<Arc<Inference>, String> {
    // Pre-flight (L5, шаг 6): без NVIDIA-драйвера не пытаемся грузить
    // модель с полным GPU-оффлоадом — честная ошибка с ссылкой на драйвер.
    if !app.gpu_ready.load(Ordering::SeqCst) {
        return Err(
            "NVIDIA-драйвер не найден (nvcuda.dll отсутствует). Рнференс требует \
             GPU: установите драйвер с https://www.nvidia.com/drivers и \
             перезапустите приложение."
                .into(),
        );
    }
    let _load_guard = app.llm_load_lock.lock().await;
    {
        let slot = app.llm.lock();
        if let Some((loaded_path, inf)) = &*slot {
            if *loaded_path == model_path {
                return Ok(inf.clone());
            }
        }
    }
    // Смена модели: сбрасываем прежний движок и грузим новый. In-flight
    // запросы держат собственный Arc старой модели и честно досчитывают на
    // ней; новые запросы получат уже новую.
    *app.llm.lock() = None;
    let path_str = model_path.to_string_lossy().into_owned();
    let backend = app.backend.clone().ok_or_else(|| {
        "llama.cpp backend не инициализирован: нет NVIDIA-драйвера \
         (см. https://www.nvidia.com/drivers)"
            .to_string()
    })?;
    let inf = tauri::async_runtime::spawn_blocking(move || {
        mentor_core::inference::load_model(&path_str, &backend)
    })
    .await
    .map_err(|e| format!("поток загрузки модели: {e}"))?
    .map_err(|e| format!("загрузка модели: {e:#}"))?;
    let arc = Arc::new(inf);
    *app.llm.lock() = Some((model_path, arc.clone()));
    Ok(arc)
}

/// Вопрос пользователя -> контекст из базы -> ответ реальной LLM.
/// Генерация стримится во фронтенд событиями "gen-token" (по куску на токен,
/// размеченный thinking/answer); параллельно копится снимок gen_stream для
/// опроса как запасного канала — тот же паттерн, что у download-progress.
#[tauri::command]
async fn send_message(
    app_handle: AppHandle,
    state: State<'_, Arc<AppState>>,
    question: String,
    history: Option<Vec<HistoryTurn>>,
) -> Result<ChatReply, String> {
    if question.trim().is_empty() {
        return Err("пустой вопрос".into());
    }
    let app = state.inner().clone();
    let cfg = app.cfg.lock().clone();
    if !cfg.model_ready() {
        return Err(
            "модель не подключена: укажи .gguf вручную или скачай на стартовом экране".into(),
        );
    }
    let llm = ensure_llm(&app, cfg.model_file_path()).await?;

    // 1. ретрив
    let mut rag = app.rag.lock().await;
    let k = cfg.qdrant.top_k as usize;
    let hits = rag
        .search(&question, k)
        .await
        .map_err(|e| format!("поиск по базе знаний: {e:#}"))?;
    // Генерация может занимать секунды — не держим RAG-мьютекс.
    drop(rag);

    // 2. контекст для промпта (единый форматтер — этап K)
    let context = format_context(&hits);

    // 3. реальный инференс вне async-потоков (блокирующая CPU-работа);
    //    каждый токен уходит во вебвью событием и в снимок для опроса.
    {
        let mut snap = app.gen_stream.lock();
        snap.thinking.clear();
        snap.answer.clear();
        snap.done = false;
    }
    // Новый запрос сбрасывает флаг отмены прошлой генерации (F-002).
    app.gen_cancel.store(false, Ordering::SeqCst);
    let generation = cfg.generation.clone();
    let q = question.trim().to_string();
    // История диалога (F-001): фронтенд присылает предыдущие ходы активного
    // треда; тримминг по бюджету делает trim_history внутри format_prompt_with_history.
    let history = history.unwrap_or_default();
    let emitter = app_handle.clone();
    let snap_state = app.clone();
    let generated = tauri::async_runtime::spawn_blocking(move || {
        generate_response_streaming(&llm, &q, &context, &history, &generation, |piece| {
            // Пользовательский «стоп» (F-002): колбэк возвращает false —
            // цикл генерации прерывается, частичный текст сохраняется.
            if snap_state.gen_cancel.load(Ordering::SeqCst) {
                return false;
            }
            {
                let mut snap = snap_state.gen_stream.lock();
                match piece.kind {
                    StreamKind::Thinking => snap.thinking.push_str(&piece.text),
                    StreamKind::Answer => snap.answer.push_str(&piece.text),
                }
            }
            let event = GenTokenEvent {
                kind: match piece.kind {
                    StreamKind::Thinking => String::from("think"),
                    StreamKind::Answer => String::from("answer"),
                },
                text: piece.text,
            };
            if let Err(e) = emitter.emit("gen-token", event) {
                eprintln!("gen: emit: {e}");
            }
            true
        })
    })
    .await
    .map_err(|e| format!("поток инференса: {e}"))?
    .map_err(|e| format!("инференс: {e:#}"))?;
    {
        let mut snap = app.gen_stream.lock();
        snap.done = true;
    }

    Ok(ChatReply {
        answer: generated.answer,
        thinking: generated.thinking,
        sources: hits,
        prompt_for_model: generated.prompt,
    })
}

/// Служебная информация для статус-бара фронтенда.
#[tauri::command]
async fn get_status(state: State<'_, Arc<AppState>>) -> Result<StatusInfo, String> {
    let app = state.inner().clone();
    let points = app
        .rag
        .lock()
        .await
        .verify_collection()
        .await
        .map_err(|e| format!("Qdrant: {e:#}"))?;
    let cfg = app.cfg.lock();
    let llm_loaded = app.llm.lock().is_some();
    let llm_error = app.llm_error.lock().clone();
    Ok(StatusInfo {
        qdrant_url: cfg.qdrant.url.clone(),
        collection: cfg.qdrant.collection.clone(),
        points,
        top_k: cfg.qdrant.top_k,
        embedding_model: cfg.embedding.model.clone(),
        model_path_set: !cfg.model_path.trim().is_empty(),
        llm_loaded,
        llm_error,
    })
}

/// Проверка модели для решения "чат или экран настройки".
/// F-015: источник — AppState.cfg (обновляется после каждой записи на диск),
/// а не повторное чтение config.toml с диска.
#[tauri::command]
async fn get_model_status(state: State<'_, Arc<AppState>>) -> Result<ModelStatus, String> {
    let cfg = state.inner().cfg.lock().clone();
    Ok(model_status_from(&cfg))
}

/// Последний снимок прогресса загрузки (фронтенд опрашивает как запасной
/// канал, если события не доходят).
#[tauri::command]
async fn get_download_progress(
    state: State<'_, Arc<AppState>>,
) -> Result<Option<DownloadEvent>, String> {
    Ok(state.inner().download_progress.lock().clone())
}

/// Снимок накопленного стрима генерации (запасной канал к событиям
/// "gen-token", если они не доходят до вебвью). Абсолютные значения —
/// фронт сверяет длины и дозабирает недостающее.
#[tauri::command]
async fn get_gen_progress(state: State<'_, Arc<AppState>>) -> Result<GenStreamSnapshot, String> {
    Ok(state.inner().gen_stream.lock().clone())
}

/// Диалог выбора готового .gguf; путь сохраняется в config.toml.
/// Возвращает None, если пользователь закрыл диалог без выбора.
#[tauri::command]
async fn pick_model_file(state: State<'_, Arc<AppState>>) -> Result<Option<ModelStatus>, String> {
    let app = state.inner().clone();
    let picked = tauri::async_runtime::spawn_blocking(|| {
        rfd::FileDialog::new()
            .set_title("Выбери GGUF-файл модели")
            .add_filter("GGUF-модели", &["gguf"])
            .add_filter("Все файлы", &["*"])
            .pick_file()
    })
    .await
    .map_err(|e| format!("поток диалога: {e}"))?;

    let Some(path) = picked else {
        return Ok(None); // диалог закрыт — не ошибка
    };
    let path_str = path.to_string_lossy().into_owned();
    save_string_field(&app.cfg_path, "model_path", &path_str)
        .map_err(|e| format!("запись config.toml: {e:#}"))?;
    reload_cfg_preserving_port(&app)?;
    let status = model_status_from(&app.cfg.lock());
    Ok(Some(status))
}

/// Фоновая задача скачивания: события прогресса в канал "download-progress",
/// по успехе путь сохраняется в config.toml. Сигнал done шлётся только
/// после записи конфига, чтобы фронтенд не увидел промежуточное состояние.
async fn run_download(
    app_handle: AppHandle,
    app: Arc<AppState>,
    url: String,
    cancel: CancelToken,
) -> Result<(), String> {
    let cfg = AppConfig::load(&app.cfg_path).map_err(|e| format!("config.toml: {e:#}"))?;
    let download_dir = cfg.download_dir();
    let dest = download_dir.join(filename_from_url(&url));
    // Защитная проверка (F-007, belt-and-braces к фильтру сепараторов в
    // filename_from_url): каталог назначения обязан остаться внутри каталога
    // загрузки — иначе URL уводит запись произвольных файлов.
    if let Some(parent) = dest.parent() {
        if let (Ok(base), Ok(actual)) = (download_dir.canonicalize(), parent.canonicalize()) {
            if !actual.starts_with(&base) {
                return Err(
                    "имя файла из URL уводит запись за пределы каталога загрузки — \
                     проверь model_download_url"
                        .into(),
                );
            }
        }
    }
    let spec = DownloadSpec {
        url: url.clone(),
        expected_sha256: {
            let sha = cfg.model_sha256.trim();
            (!sha.is_empty()).then(|| sha.to_string())
        },
        dest,
    };

    let downloader = Downloader::new().map_err(|e| format!("HTTP-клиент: {e:#}"))?;
    let snapshot = app.clone();
    let emit_handle = app_handle.clone();
    let result = downloader
        .download(&spec, &cancel, move |p| {
            let event = DownloadEvent {
                downloaded: p.downloaded,
                total: p.total,
                resumed_from: p.resumed_from,
                done: false,
                error: None,
            };
            if let Err(e) = emit_handle.emit("download-progress", event.clone()) {
                eprintln!("download: emit: {e}");
            }
            *snapshot.download_progress.lock() = Some(event);
        })
        .await;

    match result {
        Ok(()) => {
            let saved = spec.dest.to_string_lossy().into_owned();
            save_string_field(&app.cfg_path, "model_path", &saved)
                .map_err(|e| format!("запись config.toml: {e:#}"))?;
            reload_cfg_preserving_port(&app)?;
            let done_event = DownloadEvent {
                downloaded: 0,
                total: 0,
                resumed_from: 0,
                done: true,
                error: None,
            };
            *app.download_progress.lock() = Some(done_event.clone());
            let _ = app_handle.emit("download-progress", done_event);
            Ok(())
        }
        Err(e) => {
            let text = format!("{e:#}");
            let err_event = DownloadEvent {
                downloaded: 0,
                total: 0,
                resumed_from: 0,
                done: false,
                error: Some(text),
            };
            *app.download_progress.lock() = Some(err_event.clone());
            let _ = app_handle.emit("download-progress", err_event);
            Err(String::from("загрузка не завершена"))
        }
    }
}

/// Старт скачивания модели по URL из config.toml (model_download_url).
#[tauri::command]
async fn start_model_download(
    app_handle: AppHandle,
    state: State<'_, Arc<AppState>>,
) -> Result<(), String> {
    let app = state.inner().clone();
    if app.download_active.swap(true, Ordering::SeqCst) {
        return Err("загрузка уже идёт".into());
    }
    // Свежий конфиг с диска: URL могли только что отредактировать.
    let cfg = AppConfig::load(&app.cfg_path).map_err(|e| {
        app.download_active.store(false, Ordering::SeqCst);
        format!("config.toml: {e:#}")
    })?;
    let url = cfg.model_download_url.trim().to_string();
    if url.is_empty() {
        app.download_active.store(false, Ordering::SeqCst);
        return Err(
            "model_download_url в config.toml пуст — заполни его реальным адресом .gguf".into(),
        );
    }
    // Новый токен отмены на каждую загрузку (прошлая отмена не «висит»).
    let cancel = CancelToken::new();
    *app.download_cancel.lock() = cancel.clone();
    *app.download_progress.lock() = None;

    tauri::async_runtime::spawn(async move {
        let result = run_download(app_handle.clone(), app.clone(), url, cancel).await;
        app.download_active.store(false, Ordering::SeqCst);
        if let Err(e) = result {
            eprintln!("download: {e}");
        }
    });
    Ok(())
}

/// Отмена активной загрузки (.part сохраняется — следующий запуск докачает).
#[tauri::command]
async fn cancel_model_download(state: State<'_, Arc<AppState>>) -> Result<(), String> {
    state.inner().download_cancel.lock().cancel();
    Ok(())
}

/// Прерывание активной генерации (F-002): флаг проверяется в колбэке
/// каждого токена; частичный ответ сохраняется в чате и в gen_stream.
#[tauri::command]
async fn stop_generation(state: State<'_, Arc<AppState>>) -> Result<(), String> {
    state.inner().gen_cancel.store(true, Ordering::SeqCst);
    Ok(())
}

/// Снимок настроек генерации для окна настроек фронтенда.
#[derive(Serialize)]
pub struct SettingsInfo {
    /// Путь к .gguf из config.toml (может быть пуст).
    pub model_path: String,
    /// Отображаемое имя из GGUF-метаданных (general.name); если метаданных
    /// нет — имя файла; если файла нет — пустая строка.
    pub model_name: String,
    pub temperature: f32,
    pub max_tokens: u32,
    pub n_ctx: u32,
}

/// Снимок настроек из уже загруженного конфига (F-015: единый источник —
/// AppState.cfg; перечитывание с диска нужно только после внешней правки).
fn settings_from(cfg: &AppConfig) -> SettingsInfo {
    let path = cfg.model_file_path();
    let model_name = if cfg.model_ready() {
        gguf_display_name(&path).unwrap_or_else(|| {
            path.file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default()
        })
    } else {
        String::new()
    };
    SettingsInfo {
        model_path: cfg.model_path.clone(),
        model_name,
        temperature: cfg.generation.temperature,
        max_tokens: cfg.generation.max_tokens,
        n_ctx: cfg.generation.n_ctx,
    }
}

/// Текущие настройки: путь/имя модели + параметры [generation].
#[tauri::command]
async fn get_settings(state: State<'_, Arc<AppState>>) -> Result<SettingsInfo, String> {
    Ok(settings_from(&state.inner().cfg.lock()))
}

/// Сохраняет temperature/max_tokens в config.toml (с сохранением комментариев)
/// и обновляет конфиг в состоянии. Возвращает свежий снимок настроек.
#[tauri::command]
async fn set_settings(
    state: State<'_, Arc<AppState>>,
    temperature: f64,
    max_tokens: u32,
) -> Result<SettingsInfo, String> {
    if !(0.0..=2.0).contains(&temperature) {
        return Err("temperature должен быть в диапазоне 0.0–2.0".into());
    }
    if !(128..=32768).contains(&max_tokens) {
        return Err("max_tokens должен быть в диапазоне 128–32768".into());
    }
    let app = state.inner().clone();
    let cfg_path = app.cfg_path.clone();
    tauri::async_runtime::spawn_blocking(move || {
        save_generation_fields(&cfg_path, temperature, max_tokens)
    })
    .await
    .map_err(|e| format!("поток записи конфига: {e}"))?
    .map_err(|e| format!("запись config.toml: {e:#}"))?;
    reload_cfg_preserving_port(&app)?;
    let snapshot = settings_from(&app.cfg.lock());
    Ok(snapshot)
}

pub fn run() {
    // Рнициализация (первый запуск, Qdrant sidecar, RAG, модель) до старта
    // event loop: окна ещё нет, замерзать нечему; ошибки показываются
    // понятным диалогом — не молча (L5, шаг 4).
    let state = match tauri::async_runtime::block_on(init_state()) {
        Ok(s) => s,
        Err(e) => {
            // Диагностика: текст ошибки остаётся в файле (в
            // windows_subsystem=windows eprintln некуда писать).
            let _ = std::fs::write(
                std::env::temp_dir().join("ai-mentor-init-error.log"),
                format!("{e}\n"),
            );
            let description = format!(
                "Не удалось инициализировать приложение:\n\n{e}\n\n\
                 Приложение будет закрыто."
            );
            rfd::MessageDialog::new()
                .set_title("AI Mentor — ошибка запуска")
                .set_description(&description)
                .set_buttons(rfd::MessageButtons::Ok)
                .set_level(rfd::MessageLevel::Error)
                .show();
            std::process::exit(1);
        }
    };
    tauri::Builder::default()
        .manage(Arc::new(state))
        .setup(|app| {
            // Окно создаём только когда сервисы готовы.
            use tauri::webview::WebviewWindowBuilder;
            use tauri::WebviewUrl;
            WebviewWindowBuilder::new(app.handle(), "main", WebviewUrl::default())
                .title("AI Mentor — локальный наставник")
                .inner_size(1180.0, 780.0)
                .min_inner_size(720.0, 540.0)
                .build()?;
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            send_message,
            get_status,
            get_model_status,
            pick_model_file,
            start_model_download,
            cancel_model_download,
            get_download_progress,
            get_gen_progress,
            get_settings,
            set_settings,
            stop_generation
        ])
        .build(tauri::generate_context!())
        .expect("ошибка сборки Tauri-приложения")
        .run(|app_handle, event| {
            // ОБЯЗАТЕЛЬНО (L5, шаг 4): при выходе останавливаем sidecar
            // процессы, иначе qdrant.exe останется висеть после закрытия окна.
            if let tauri::RunEvent::Exit = event {
                let state: State<Arc<AppState>> = app_handle.state();
                qdrant::stop_qdrant(&state.qdrant);
            }
        });
}
