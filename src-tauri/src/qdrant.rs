//! Менеджер жизненного цикла Qdrant sidecar (Этап L5).
//!
//! Приложение владеет процессом Qdrant: запускает его на свободном порту,
//! ждёт готовности (polling /readiness) и гарантированно останавливает при
//! выходе (hook на RunEvent::Exit в run()). Данные персистятся в AppData.
//!
//! Почему std::process, а не tauri-plugin-shell: бинарь бандлится как
//! externalBin (см. tauri.conf.json), но запускается ДО старта event loop
//! (до создания окна), а спавн через плагин привязан к event loop и
//! взаимно блокируется с block_on в инициализации. std::process не зависит
//! от рантайма Tauri и работает одинаково до/после запуска UI.
//!
//! Замечание о graceful shutdown: на Windows нет POSIX-сигналов; qdrant
//! пишет WAL после каждой операции, поэтому завершение через TerminateProcess
//! безопасно для данных (проверено штатными перезапусками на этапах D/L3).
use std::path::Path;
use std::process::Child;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};

/// Дочерний процесс Qdrant; None = не запущен. parking_lot: короткие
/// секции, гвард не держится через .await.
pub struct QdrantProc {
    child: parking_lot::Mutex<Option<Child>>,
}

impl QdrantProc {
    pub fn new() -> Self {
        Self {
            child: parking_lot::Mutex::new(None),
        }
    }

    pub fn is_running(&self) -> bool {
        self.child.lock().is_some()
    }
}

impl Default for QdrantProc {
    fn default() -> Self {
        Self::new()
    }
}

/// Первый свободный TCP-порт начиная с `start` (исключая `skip`, если задан).
/// Проверка через connect (а не bind!): на Windows bind(127.0.0.1:P)
/// успешно проходит даже когда другой процесс держит 0.0.0.0:P — и сервис
/// потом умирает на реальном bind'е. connect-пробой детектирует любого
/// слушателя. Редкая гонка «проверили -> заняли» ловится wait_for_ready.
pub fn find_free_port(start: u16, skip: Option<u16>) -> u16 {
    // Ограничение перебора: защита от теоретического зацикливания на u16
    // (wrapping_add); на практике свободный порт находится в первых десятках.
    for offset in 0..1024u16 {
        let port = start.wrapping_add(offset);
        let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
        let busy = std::net::TcpStream::connect_timeout(&addr, Duration::from_millis(150)).is_ok();
        if !busy && Some(port) != skip {
            return port;
        }
    }
    panic!("не удалось найти свободный порт начиная с {start}");
}

/// Путь к qdrant.exe: Tauri кладёт externalBin рядом с основным бинарем
/// (в установленном MSI — в каталоге установки; при cargo build — рядом
/// с target/<profile>/mentor-tauri.exe, имя уже без target-triple суффикса
/// в бандле; в dev-запусках tauri-build копирует суффиксованный файл).
pub fn sidecar_exe_path() -> Result<std::path::PathBuf> {
    let dir = std::env::current_exe()
        .context("не удалось определить каталог приложения")?
        .parent()
        .expect("exe лежит в каталоге")
        .to_path_buf();
    // prod-бандл: qdrant.exe; dev (cargo build без бандла): суффикс triple.
    let candidates = [
        dir.join(format!("qdrant{}", std::env::consts::EXE_SUFFIX)),
        dir.join(format!(
            "qdrant-{}{}",
            current_triple(),
            std::env::consts::EXE_SUFFIX
        )),
    ];
    candidates
        .iter()
        .find(|p| p.is_file())
        .cloned()
        .context(format!(
            "qdrant sidecar не найден рядом с приложением ({})",
            dir.display()
        ))
}

fn current_triple() -> String {
    // tauri-build копирует externalBin с суффиксом target-triple сборки.
    // На этапе рантайма определим его из имени файла рядом с exe.
    // Для Windows-x64 проекта triple фиксирован сборкой MSVC.
    String::from("x86_64-pc-windows-msvc")
}

/// Запускает qdrant с динамическими портами и стором в AppData.
/// Возвращает выбранную пару портов (http, grpc).
pub fn start_qdrant(
    proc_state: &QdrantProc,
    exe: &Path,
    storage_path: &Path,
) -> Result<(u16, u16)> {
    let http_port = find_free_port(6333, None);
    let grpc_port = find_free_port(6334, Some(http_port));
    let mut cmd = std::process::Command::new(exe);
    cmd.env("QDRANT__SERVICE__HTTP_PORT", http_port.to_string())
        .env("QDRANT__SERVICE__GRPC_PORT", grpc_port.to_string())
        .env(
            "QDRANT__STORAGE__STORAGE_PATH",
            storage_path.to_string_lossy().into_owned(),
        );
    // GUI-приложение не должно показывать консоль sidecar.
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    let child = cmd
        .spawn()
        .with_context(|| format!("не удалось запустить {}", exe.display()))?;
    *proc_state.child.lock() = Some(child);
    Ok((http_port, grpc_port))
}

/// Polling health-эндпойнта до успеха или таймаута (30 с по требованию L5).
/// В задаче назван "/readiness", но у Qdrant 1.19 такой эндпойнт отдаёт 404;
/// канонические health-пути Qdrant — /readyz, /livez, /healthz (k8s-стиль).
/// Поллим /readyz — готовность сервиса обрабатывать запросы.
pub async fn wait_for_ready(http_port: u16, timeout: Duration) -> Result<()> {
    let url = format!("http://127.0.0.1:{http_port}/readyz");
    let deadline = Instant::now() + timeout;
    let client = reqwest::Client::new();
    while Instant::now() < deadline {
        match client.get(&url).send().await {
            Ok(resp) if resp.status().is_success() => return Ok(()),
            _ => tokio::time::sleep(Duration::from_millis(500)).await,
        }
    }
    bail!("Qdrant не поднялся за {} с ({url})", timeout.as_secs())
}

/// Останавливает дочерний процесс Qdrant. Вызывается из RunEvent::Exit.
pub fn stop_qdrant(proc_state: &QdrantProc) {
    if let Some(mut child) = proc_state.child.lock().take() {
        let _ = child.kill();
        // Забираем статус, чтобы не остался зомби-процесс.
        let _ = child.wait();
    }
}
