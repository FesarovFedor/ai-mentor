//! Скачивание GGUF-модели по HTTP(S): прогресс, докачка после обрыва
//! (resume через Range), отмена, проверка места на диске и SHA-256.
//!
//! Докачка: частичный файл хранится как `<dest>.part`, рядом метаданные
//! `<dest>.part.meta.json` (url/total/etag). При повторном старте шлётся
//! `Range: bytes=<have>-`; серверный 206 -> дописываем, 200 -> начинаем
//! заново (ресурс изменился или Range не поддерживается), 416 при полном
//! `.part` -> сразу финализация без тела (обрыв случился между концом
//! загрузки и переименованием). Дополнительная защита от подмены ресурса:
//! тотал из `Content-Range` обязан совпасть с сохранённым в метаданных.
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Сигнал отмены, безопасен для передачи между задачами.
#[derive(Clone, Default)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

/// Пользовательская отмена (не считается ошибкой сети).
#[derive(Debug)]
pub struct Cancelled;

impl std::fmt::Display for Cancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("загрузка отменена пользователем")
    }
}
impl std::error::Error for Cancelled {}

/// Снимок прогресса для UI (передаётся троттлингом ~раз в 100мс и в конце).
#[derive(Debug, Clone, Copy, Serialize)]
pub struct DownloadProgress {
    pub downloaded: u64,
    /// 0 = сервер не дал Content-Length, процент неизвестен.
    pub total: u64,
    /// С какого байта стартовал этот запуск (0 — качали с нуля).
    pub resumed_from: u64,
    pub done: bool,
}

/// Что и куда качать.
#[derive(Debug, Clone)]
pub struct DownloadSpec {
    pub url: String,
    pub dest: PathBuf,
    pub expected_sha256: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct PartMeta {
    url: String,
    total: u64,
    #[serde(default)]
    etag: Option<String>,
}

/// Как писать тело ответа.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WriteMode {
    /// Дописывать существующий .part (подтверждённый 206).
    Append,
    /// Писать с нуля (truncate).
    Fresh,
    /// Тела нет и оно не нужно: .part уже полный, сразу проверки+rename.
    FinalizeOnly,
}

const PROGRESS_INTERVAL: Duration = Duration::from_millis(100);
/// Запас поверх размера файла при проверке свободного места.
const DISK_MARGIN_BYTES: u64 = 256 * 1024 * 1024;

fn part_paths(dest: &Path) -> (PathBuf, PathBuf) {
    let mut name = dest.file_name().unwrap_or_default().to_os_string();
    name.push(".part");
    let part = dest.with_file_name(name);
    let mut meta_name = part.file_name().unwrap_or_default().to_os_string();
    meta_name.push(".meta.json");
    let meta = dest.with_file_name(meta_name);
    (part, meta)
}

/// Имя файла по URL: последний сегмент без query/fragment, декодированный
/// процент-энкодинг и очищенный от запрещённых в Windows символов.
///
/// Сепараторы `/` и `\` заменяются вместе с прочими запрещёнными символами:
/// без этого percent-encoded `..%5c..%5c` после декодирования превращается
/// в путь с обходом каталога загрузки (path traversal, фикс F-007 аудита).
pub fn filename_from_url(url: &str) -> String {
    let raw = url.split(['#', '?']).next().unwrap_or(url);
    let raw = raw.rsplit('/').next().unwrap_or(raw);
    let decoded = percent_decode_minimal(raw);
    let cleaned: String = decoded
        .chars()
        .map(|c| {
            if c.is_control() || "<>:\"|?*/\\".contains(c) {
                '_'
            } else {
                c
            }
        })
        .collect();
    let trimmed = cleaned.trim().trim_matches('.').to_string();
    if trimmed.is_empty() {
        String::from("model.gguf")
    } else {
        trimmed
    }
}

fn percent_decode_minimal(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let hex_pair = if bytes[i] == b'%' && i + 2 < bytes.len() {
            bytes[i + 1..i + 3]
                .iter()
                .map(|b| (*b as char).to_digit(16))
                .collect::<Option<Vec<_>>>()
                .filter(|d| d.len() == 2)
        } else {
            None
        };
        match hex_pair {
            Some(digits) => {
                out.push((digits[0] * 16 + digits[1]) as u8);
                i += 3;
            }
            None => {
                out.push(bytes[i]);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

pub struct Downloader {
    client: reqwest::Client,
}

impl Downloader {
    pub fn new() -> Result<Self> {
        // Таймауты критичны для продакшена: с одним connect_timeout сервер,
        // принявший TCP-соединение, но переставший слать данные (зависший
        // прокси, уснувший CDN, разорванный NAT), вешает response.chunk().await
        // НАВСЕГДА — поток загрузки и UI замирают без ошибки и без прогресса
        // (P0 из аудита).
        // - read_timeout(60s): максимальная пауза МЕЖДУ пакетами. Порция
        //   данных любой живой связи приходит чаще; тишина дольше минуты —
        //   соединение мертво, докачка (Range) продолжит со следующей попытки.
        // - timeout(7200s): общий бюджет одной попытки, защита от бесконечного
        //   «капельного» скачивания гигабайтных файлов. Двух часов достаточно
        //   даже для 100+ GiB на быстром канале; прерывание не теряет данные —
        //   .part + meta.json позволяют докачать.
        let client = reqwest::Client::builder()
            .user_agent("ai-mentor-downloader/0.1")
            .connect_timeout(Duration::from_secs(30))
            .read_timeout(Duration::from_secs(60))
            .timeout(Duration::from_secs(7200))
            .redirect(reqwest::redirect::Policy::limited(10))
            .build()
            .context("не удалось создать HTTP-клиент")?;
        Ok(Self { client })
    }

    /// Качает spec.dest; по завершении проверяет SHA-256 (если задан) и
    /// магию GGUF, затем атомарно переименовывает .part в целевой файл.
    /// Функция async и вызывается из фоновой задачи — UI не блокируется.
    pub async fn download(
        &self,
        spec: &DownloadSpec,
        cancel: &CancelToken,
        mut on_progress: impl FnMut(DownloadProgress),
    ) -> Result<()> {
        if !spec.url.starts_with("http://") && !spec.url.starts_with("https://") {
            bail!("URL должен начинаться с http:// или https://: {}", spec.url);
        }
        let dest_dir = spec
            .dest
            .parent()
            .ok_or_else(|| anyhow!("у пути нет родительского каталога: {}", spec.dest.display()))?;
        std::fs::create_dir_all(dest_dir)
            .with_context(|| format!("не удалось создать каталог {}", dest_dir.display()))?;

        let (part, meta_path) = part_paths(&spec.dest);
        // F-024: битые/нераспарсиваемые метаданные (.part.meta.json) раньше
        // молча сбрасывали загрузку на ноль. BOM толерируем, прочий мусор —
        // с предупреждением в stderr.
        let saved_meta: Option<PartMeta> = match std::fs::read(&meta_path) {
            Ok(raw) => {
                let stripped = raw.strip_prefix(&[0xEF, 0xBB, 0xBF]).unwrap_or(&raw);
                match serde_json::from_slice::<PartMeta>(stripped) {
                    Ok(m) => Some(m),
                    Err(e) => {
                        eprintln!(
                            "предупреждение: метаданные {} повреждены ({e}) — \
                             загрузка начнётся с нуля",
                            meta_path.display()
                        );
                        None
                    }
                }
            }
            Err(_) => None,
        }
        .filter(|m| m.url == spec.url);
        // total из прошлой сессии: 0 = неизвестен/нет метаданных
        let meta_total = saved_meta.as_ref().map(|m| m.total).unwrap_or(0);

        let mut have: u64 = 0;
        if saved_meta.is_some() && part.exists() {
            have = std::fs::metadata(&part)?.len();
        } else {
            cleanup_part(&part, &meta_path);
        }

        let mut request = self.client.get(&spec.url);
        if have > 0 {
            request = request.header(reqwest::header::RANGE, format!("bytes={have}-"));
        }
        let response = match request.send().await {
            Ok(r) => r,
            Err(e) => return Err(preserve_cancel(e, cancel)),
        };
        if cancel.is_cancelled() {
            return Err(anyhow::Error::new(Cancelled));
        }
        let status = response.status();
        let mut response = response;

        // Режим записи и ожидаемый полный размер ресурса.
        let (mode, mut expected_total) = match status {
            reqwest::StatusCode::PARTIAL_CONTENT => {
                let cr = content_range_total(&response).unwrap_or(0);
                if cr > 0 && cr >= have && (meta_total == 0 || cr == meta_total) {
                    (WriteMode::Append, cr)
                } else if cr == 0 && have == 0 {
                    // 206 без Content-Range на свежий запрос: размер возьмём
                    // из Content-Length ниже
                    (WriteMode::Fresh, 0)
                } else {
                    // часть длиннее ресурса или ресурс подменён — докачивать нельзя
                    cleanup_part(&part, &meta_path);
                    (WriteMode::Fresh, 0)
                }
            }
            reqwest::StatusCode::OK => {
                if have > 0 {
                    // сервер проигнорировал Range — начинаем заново
                    cleanup_part(&part, &meta_path);
                }
                (WriteMode::Fresh, content_length(&response).unwrap_or(0))
            }
            reqwest::StatusCode::RANGE_NOT_SATISFIABLE => {
                if meta_total != 0 && have == meta_total {
                    // .part уже полный, обрыв случился до переименования
                    (WriteMode::FinalizeOnly, meta_total)
                } else {
                    have = 0;
                    cleanup_part(&part, &meta_path);
                    (WriteMode::Fresh, 0)
                }
            }
            other => {
                let hint = if other.is_client_error() {
                    "ошибка запроса или доступа (4xx); проверь URL"
                } else if other.is_server_error() {
                    "временная ошибка сервера (5xx); попробуй позже"
                } else {
                    "неожиданный ответ сервера"
                };
                bail!("сервер вернул HTTP {other}: {hint}");
            }
        };

        // 206 без распарсенного Content-Range: остаёмся на Content-Length.
        if expected_total == 0 {
            if let Some(len) = content_length(&response) {
                expected_total = if mode == WriteMode::Append {
                    have + len
                } else {
                    len
                };
            }
        }
        if mode != WriteMode::FinalizeOnly && expected_total > 0 {
            check_disk_space(dest_dir, expected_total.saturating_sub(have))?;
        }

        let resumed_from = have;
        emit(
            &mut on_progress,
            if mode == WriteMode::FinalizeOnly {
                expected_total
            } else {
                have
            },
            expected_total,
            resumed_from,
            false,
        );

        if mode != WriteMode::FinalizeOnly {
            // Метаданные .part нужны пережить перезапуск процесса: url фиксирует
            // ресурс, total отсекает подмену/битый .part при следующем старте.
            let meta_to_store = PartMeta {
                url: spec.url.clone(),
                total: expected_total,
                etag: response
                    .headers()
                    .get(reqwest::header::ETAG)
                    .and_then(|v| v.to_str().ok())
                    .map(str::to_owned),
            };
            std::fs::write(
                &meta_path,
                serde_json::to_vec(&meta_to_store)
                    .context("не удалось сериализовать метаданные .part")?,
            )
            .with_context(|| format!("не удалось записать {}", meta_path.display()))?;
        }

        let mut downloaded = have;
        if mode != WriteMode::FinalizeOnly {
            let mut writer = if mode == WriteMode::Append && have > 0 {
                tokio::fs::OpenOptions::new()
                    .append(true)
                    .open(&part)
                    .await
                    .with_context(|| {
                        format!("не удалось открыть для дозаписи {}", part.display())
                    })?
            } else {
                tokio::fs::OpenOptions::new()
                    .create(true)
                    .write(true)
                    .truncate(true)
                    .open(&part)
                    .await
                    .with_context(|| format!("не удалось открыть на запись {}", part.display()))?
            };

            let mut last_emit = Instant::now() - PROGRESS_INTERVAL;
            use tokio::io::AsyncWriteExt;
            loop {
                if cancel.is_cancelled() {
                    writer.flush().await.ok();
                    return Err(anyhow::Error::new(Cancelled));
                }
                let chunk = match response.chunk().await {
                    Ok(Some(c)) => c,
                    Ok(None) => break,
                    Err(e) => return Err(preserve_cancel(e, cancel)),
                };
                writer.write_all(&chunk).await.with_context(|| {
                    format!("ошибка записи в {} (место на диске?)", part.display())
                })?;
                downloaded += chunk.len() as u64;
                if last_emit.elapsed() >= PROGRESS_INTERVAL {
                    last_emit = Instant::now();
                    emit(
                        &mut on_progress,
                        downloaded,
                        expected_total,
                        resumed_from,
                        false,
                    );
                }
            }
            writer
                .flush()
                .await
                .context("ошибка сброса буфера записи")?;
            writer
                .sync_all()
                .await
                .context("ошибка sync_all при записи файла модели")?;

            if expected_total != 0 && downloaded < expected_total {
                bail!(
                    "соединение оборвалось раньше конца: получено {downloaded} из {expected_total} байт; \
                     нажми «скачать» ещё раз — загрузка продолжится с этого места"
                );
            }
        } else {
            downloaded = expected_total;
        }

        emit(
            &mut on_progress,
            downloaded,
            expected_total,
            resumed_from,
            true,
        );

        // Контрольная сумма / магия GGUF до переименования.
        if let Some(expected_sha) = &spec.expected_sha256 {
            let actual = sha256_hex_file(&part)?;
            if !actual.eq_ignore_ascii_case(expected_sha.trim()) {
                cleanup_part(&part, &meta_path);
                bail!(
                    "контрольная сумма не совпала: ожидалось sha256={expected_sha}, \
                     фактически {actual}; файл удалён"
                );
            }
        }
        // F-010 (аналог S4 audit-v4): магия проверяется ВСЕГДА, независимо от
        // расширения имени (URL мог отдать model.bin, query съел расширение и
        // т.п.). Без этого HTML-страница ошибки проходила бы как «модель».
        let magic_ok = std::fs::File::open(&part)
            .and_then(|mut f| {
                let mut head = [0u8; 4];
                f.read_exact(&mut head)?;
                Ok(&head == b"GGUF")
            })
            .unwrap_or(false);
        if !magic_ok {
            cleanup_part(&part, &meta_path);
            bail!("файл не начинается с магии GGUF — похоже, скачалась HTML-страница ошибки");
        }

        std::fs::rename(&part, &spec.dest).with_context(|| {
            format!(
                "не удалось переименовать {} в {}",
                part.display(),
                spec.dest.display()
            )
        })?;
        std::fs::remove_file(&meta_path).ok();

        Ok(())
    }
}

fn preserve_cancel(err: reqwest::Error, cancel: &CancelToken) -> anyhow::Error {
    if cancel.is_cancelled() {
        anyhow::Error::new(Cancelled)
    } else {
        anyhow::Error::new(err).context("сетевая ошибка")
    }
}

fn emit(
    on_progress: &mut impl FnMut(DownloadProgress),
    downloaded: u64,
    total: u64,
    resumed_from: u64,
    done: bool,
) {
    on_progress(DownloadProgress {
        downloaded,
        total,
        resumed_from,
        done,
    });
}

fn content_length(response: &reqwest::Response) -> Option<u64> {
    response
        .headers()
        .get(reqwest::header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse().ok())
}

/// "bytes 123-456/789" -> 789
fn content_range_total(response: &reqwest::Response) -> Option<u64> {
    response
        .headers()
        .get(reqwest::header::CONTENT_RANGE)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.rsplit('/').next()?.trim().parse().ok())
}

fn check_disk_space(dir: &Path, need: u64) -> Result<()> {
    let stats = fs4::statvfs(dir)
        .map_err(|e| anyhow!("не удалось определить свободное место на диске: {e}"))?;
    let available = stats.available_space();
    if available < need.saturating_add(DISK_MARGIN_BYTES) {
        bail!(
            "недостаточно места на диске: нужно ~{} МиБ (плюс запас), доступно {} МиБ",
            need / (1024 * 1024),
            available / (1024 * 1024)
        );
    }
    Ok(())
}

fn cleanup_part(part: &Path, meta: &Path) {
    std::fs::remove_file(part).ok();
    std::fs::remove_file(meta).ok();
}

/// Потоковый SHA-256 файла в hex (не грузит файл целиком в память).
pub fn sha256_hex_file(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("не удалось открыть {} для подсчёта sha256", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 512 * 1024];
    loop {
        let n = file
            .read(&mut buf)
            .context("ошибка чтения при подсчёте sha256")?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// F-007: сепараторы и percent-encoded traversal не должны уводить имя
    /// файла за пределы каталога загрузки.
    #[test]
    fn filename_from_url_strips_path_separators() {
        // Злонамеренный URL из находки F-007: %5c = '\', %2e%2e = '..'
        let name = filename_from_url("https://host/%2e%2e%5c%2e%2e%5cWindows\\Temp\\evil.gguf");
        assert!(!name.contains('\\'), "backslash должен заменяться: {name}");
        assert!(!name.contains('/'), "slash должен заменяться: {name}");
        // Имя безопасно: join(download_dir, name) не уходит за пределы каталога.
        assert_eq!(
            Path::new(&name).file_name(),
            Some(std::ffi::OsStr::new(name.as_str()))
        );

        // Обычный слэш-вариант traversal (после rsplit последнего сегмента
        // уже нет '/', но закодированный %2f приходит именно сюда).
        assert_eq!(filename_from_url("https://host/a%2fb.gguf"), "a_b.gguf");

        // Прямой backslash в сегменте (Windows-стиль URL-мусора).
        let name = filename_from_url("https://host/..%5C..%5Cevil.gguf");
        assert!(!name.contains('\\') && !name.contains('/'), "{name}");
        assert_eq!(
            Path::new(&name).file_name(),
            Some(std::ffi::OsStr::new(name.as_str()))
        );
    }

    #[test]
    fn filename_from_url_basics() {
        assert_eq!(
            filename_from_url("https://example.com/models/nanbeige.Q4_K_M.gguf?download=1"),
            "nanbeige.Q4_K_M.gguf"
        );
        assert_eq!(filename_from_url("https://example.com/"), "model.gguf");
        assert_eq!(filename_from_url("https://example.com/..gguf"), "gguf");
        // Windows-запрещённые символы — заменяются ('?' режется как query-разделитель).
        assert_eq!(filename_from_url("https://h/a<b>|c.gguf"), "a_b__c.gguf");
    }

    #[test]
    fn percent_decode_minimal_decodes_and_preserves() {
        assert_eq!(percent_decode_minimal("%2e%2e%5C"), "..\\");
        assert_eq!(percent_decode_minimal("a%20b"), "a b");
        // Незакрытый % остаётся как есть.
        assert_eq!(percent_decode_minimal("100%"), "100%");
        // Не-UTF8 байты не паникуют (lossy).
        assert_eq!(percent_decode_minimal("%FF"), "\u{FFFD}");
    }
}
