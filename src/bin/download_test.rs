//! CLI-проверка загрузчика модели (этап C): прогресс, докачка по Range,
//! отмена, проверка SHA-256. Не является частью приложения.
//!
//! Примеры:
//!   download_test <url> <dest> [stop_after_mb] [expected_sha256]
//! stop_after_mb > 0 — имитация обрыва: как только скачано указанное число
//! МиБ, процесс отменяет загрузку и завершается, оставляя .part на диске.
//! Следующий запуск без этого аргумента продолжает с того же места.
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use mentor_core::config::AppConfig;
use mentor_core::downloader::{CancelToken, DownloadSpec, Downloader};

fn default_config_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("config.toml")
}

#[tokio::main]
async fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("использование: download_test <url> <dest> [stop_after_mb] [expected_sha256]");
        return ExitCode::from(2);
    }
    let url = args[1].clone();
    let dest = PathBuf::from(&args[2]);
    let stop_after_bytes: u64 = args
        .get(3)
        .and_then(|v| v.parse::<f64>().ok())
        .map(|mb| (mb * 1024.0 * 1024.0) as u64)
        .unwrap_or(0);
    let expected_sha = args
        .get(4)
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    let cfg = match AppConfig::load(&default_config_path()) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("config.toml: {e:#}");
            return ExitCode::FAILURE;
        }
    };
    let _ = cfg; // конфиг нужен загрузчику косвенно (здесь только для валидации чтения)

    let spec = DownloadSpec {
        url,
        dest,
        expected_sha256: expected_sha,
    };

    let downloader = match Downloader::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("{e:#}");
            return ExitCode::FAILURE;
        }
    };

    let cancel = CancelToken::new();
    let observed = Arc::new(AtomicU64::new(0));
    let resumed = Arc::new(AtomicBool::new(false));
    let started = Instant::now();

    // Наблюдатель: имитирует обрыв на stop_after_bytes.
    let watcher_cancel = cancel.clone();
    let watcher_observed = observed.clone();
    let watcher = tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_millis(100)).await;
            let cur = watcher_observed.load(Ordering::Relaxed);
            if stop_after_bytes > 0 && cur >= stop_after_bytes {
                eprintln!("STOP-TEST: достигнуто {cur} байт — имитируем обрыв (отмена загрузки)");
                watcher_cancel.cancel();
                return;
            }
        }
    });

    let cb_observed = observed.clone();
    let cb_resumed = resumed.clone();
    let mut last_printed_pct: i64 = -1;
    let result = downloader
        .download(&spec, &cancel, move |p| {
            cb_observed.store(p.downloaded, Ordering::Relaxed);
            if p.resumed_from > 0 && !cb_resumed.swap(true, Ordering::Relaxed) {
                eprintln!(
                    "RESUME: сервер подтвердил докачку с байта {} ({:.1} МиБ)",
                    p.resumed_from,
                    p.resumed_from as f64 / (1024.0 * 1024.0)
                );
            }
            if p.done {
                eprintln!(
                    "DONE: получено {} байт (resume с {})",
                    p.downloaded, p.resumed_from
                );
                return;
            }
            if p.total > 0 {
                let pct = (p.downloaded as f64 / p.total as f64 * 100.0) as i64;
                if pct != last_printed_pct {
                    last_printed_pct = pct;
                    eprintln!("progress: {pct}%  ({}/{})", p.downloaded, p.total);
                }
            } else {
                eprintln!("progress: {} байт", p.downloaded);
            }
        })
        .await;

    watcher.abort();
    let _ = watcher.await;

    match result {
        Ok(()) => {
            eprintln!(
                "OK: {} готов за {:.1}s",
                spec.dest.display(),
                started.elapsed().as_secs_f64()
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            if e.chain()
                .any(|c| c.is::<mentor_core::downloader::Cancelled>())
            {
                eprintln!(
                    "CANCELLED после {} байт; .part сохранён для следующего запуска",
                    observed.load(Ordering::Relaxed)
                );
                if stop_after_bytes > 0 {
                    return ExitCode::SUCCESS; // ожидаемый исход теста обрыва
                }
                return ExitCode::from(3);
            }
            eprintln!("FAIL: {e:#}");
            ExitCode::FAILURE
        }
    }
}
