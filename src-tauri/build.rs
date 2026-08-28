//! Сборка Tauri-приложения + автоматическая загрузка бинарных зависимостей
//! бандла (Этап L5).
//!
//! Бинарники НЕ хранятся в git (best practices + история репозитория).
//! Вместо этого при сборке:
//!   1. Ищем файл в кэше target/deps_cache/ (не качаем при каждой сборке).
//!   2. Если файла нет — скачиваем с официального источника и проверяем
//!      SHA-256: для CUDA — против checksum'ов из официального
//!      redistrib_12.8.1.json NVIDIA; для ONNX Runtime и Qdrant — против
//!      зафиксированных ниже хэшей проверенных релизных zip-архивов. Для
//!      ONNX Runtime и Qdrant на GitHub существуют сторонние/релизные
//!      checksum'и, но единообразный pinned-SHA256 надёжнее для
//!      воспроизводимости: версии зафиксированы явно.
//!   3. Распаковываем и кладём в src-tauri/resources/ (DLL, данные БЗ)
//!      или src-tauri/binaries/ (qdrant sidecar, имя = <bin>-<TARGET>.exe
//!      по требованиям Tauri externalBin).
//!
//! Windows-only: проект официально поддерживает только Windows + NVIDIA.
//!
//! Источники (версии зафиксированы намеренно для воспроизводимости):
//!   - CUDA 12.8.1: https://developer.download.nvidia.com/compute/cuda/redist/
//!     (cuda_cudart 12.8.90, libcublas 12.8.4.1; SHA-256 из официального JSON)
//!   - ONNX Runtime 1.24.4: GitHub microsoft/onnxruntime releases
//!   - Qdrant 1.19.0: GitHub qdrant/qdrant releases
//!   - Данные базы знаний (qdrant-storage/, kb_chunks/): НЕ скачиваются из
//!     сети — копируются сборщиком из локального дерева проекта
//!     (tools_bin/qdrant_server/storage и ../kb_chunks); это данные
//!     мейнтейнера, в git их тоже нельзя.
use std::env;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

use sha2::{Digest, Sha256};

const CUDA_VERSION: &str = "12.8.1";
const CUDART_ZIP: &str = "cuda_cudart-windows-x86_64-12.8.90-archive.zip";
const CUBLAS_ZIP: &str = "libcublas-windows-x86_64-12.8.4.1-archive.zip";
const ONNX_ZIP: &str = "onnxruntime-win-x64-1.24.4.zip";
const QDRANT_ZIP: &str = "qdrant-x86_64-pc-windows-msvc.zip";

/// SHA-256 zip-архивов (pinned; для CUDA сверяются ещё и с redistrib JSON).
const ONNX_ZIP_SHA256: &str = "d2319fddfb6ea4db99ccc4b60c85c517bcd855721f5daa6a06d40d7cb2ee2357";
const QDRANT_ZIP_SHA256: &str = "980cb2e1ae771155cf211da8c0a8a9206b6482bd4effdc4db994d3adb707b087";

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let workspace = manifest_dir
        .parent()
        .expect("src-tauri живёт в workspace")
        .to_path_buf();
    let resources = manifest_dir.join("resources");
    let binaries = manifest_dir.join("binaries");
    let cache = workspace.join("target").join("deps_cache");
    fs::create_dir_all(&cache).expect("не удалось создать deps_cache");

    ensure_cuda_dll(
        &cache,
        &resources,
        "cudart64_12.dll",
        CUDART_ZIP,
        "cuda_cudart",
    );
    ensure_cuda_dll(
        &cache,
        &resources,
        "cublas64_12.dll",
        CUBLAS_ZIP,
        "libcublas",
    );
    ensure_cuda_dll(
        &cache,
        &resources,
        "cublasLt64_12.dll",
        CUBLAS_ZIP,
        "libcublas",
    );
    ensure_onnx_dlls(&cache, &resources);
    ensure_qdrant_sidecar(&cache, &binaries);
    ensure_kb_data(&workspace, &resources);

    println!("cargo:rerun-if-changed=build.rs");
    tauri_build::build();
}

fn sha256_file(path: &Path) -> String {
    let mut file = fs::File::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 512 * 1024];
    loop {
        let n = file.read(&mut buf).expect("read sha256");
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    hex::encode(hasher.finalize())
}

/// Скачивает url в cache/имя (если ещё нет) с отчётом о прогрессе в stdout.
fn download(cache: &Path, name: &str, url: &str) -> PathBuf {
    let dest = cache.join(name);
    if dest.exists() {
        return dest;
    }
    println!("cargo:warning=L5 build deps: скачиваю {name} ...");
    let partial = cache.join(format!("{name}.part"));
    get_url(url, &partial);
    fs::rename(&partial, &dest).expect("rename скачанного файла");
    dest
}

/// HTTP(S) GET без внешних crate'ов: Invoke-WebRequest (Windows-only проект).
/// Прогресс отключён ($ProgressPreference) — иначе качается в разы медленнее.
/// Параметры передаются через env: -Command не биндит $args при склейке
/// нескольких аргументов.
fn get_url(url: &str, dest: &Path) {
    let status = Command::new("powershell")
        .env("L5_URL", url)
        .env("L5_OUT", dest.to_string_lossy().into_owned())
        .args([
            "-NoProfile",
            "-Command",
            "$ProgressPreference='SilentlyContinue'; \
             [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; \
             Invoke-WebRequest -Uri $env:L5_URL -OutFile $env:L5_OUT",
        ])
        .status()
        .expect("не удалось запустить powershell для скачивания");
    if !status.success() {
        panic!("скачивание не удалось: {url}");
    }
}

/// Распаковка zip без внешних crate'ов: Expand-Archive (Windows-only).
fn unzip(zip: &Path, into: &Path) {
    let status = Command::new("powershell")
        .env("L5_ZIP", zip.to_string_lossy().into_owned())
        .env("L5_DST", into.to_string_lossy().into_owned())
        .args([
            "-NoProfile",
            "-Command",
            "$ProgressPreference='SilentlyContinue'; \
             Expand-Archive -Path $env:L5_ZIP -DestinationPath $env:L5_DST -Force",
        ])
        .status()
        .expect("не удалось запустить powershell для распаковки");
    if !status.success() {
        panic!("распаковка не удалась: {}", zip.display());
    }
}

fn verify_sha(path: &Path, expected: &str, what: &str) {
    let actual = sha256_file(path);
    if !actual.eq_ignore_ascii_case(expected) {
        panic!(
            "{what}: SHA-256 не совпал (ожидалось {expected}, фактически {actual}) — \
             файл удалён, повторите сборку",
        );
    }
}

/// CUDA-компонент: качаем официальный redist-архив, сверяем SHA-256 с
/// официальным redistrib_{CUDA_VERSION}.json NVIDIA, достаём DLL из bin/.
fn ensure_cuda_dll(cache: &Path, resources: &Path, dll: &str, zip_name: &str, component: &str) {
    let dest = resources.join(dll);
    if dest.exists() {
        return;
    }
    // 1. Публикуемый NVIDIA checksum из официального манифеста версии.
    let manifest = cache.join(format!("redistrib_{CUDA_VERSION}.json"));
    if !manifest.exists() {
        get_url(
            &format!(
                "https://developer.download.nvidia.com/compute/cuda/redist/redistrib_{CUDA_VERSION}.json"
            ),
            &manifest,
        );
    }
    let json: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&manifest).expect("чтение redistrib json"))
            .expect("парсинг redistrib json");
    let entry = &json[component]["windows-x86_64"];
    let relative = entry["relative_path"]
        .as_str()
        .unwrap_or_else(|| panic!("в redistrib json нет {component}/windows-x86_64"));
    let published_sha = entry["sha256"]
        .as_str()
        .unwrap_or_else(|| panic!("в redistrib json нет sha256 для {component}"));

    // 2. Скачивание + проверка по официальному checksum'у.
    let zip = download(
        cache,
        zip_name,
        &format!("https://developer.download.nvidia.com/compute/cuda/redist/{relative}"),
    );
    verify_sha(&zip, published_sha, &format!("CUDA redist {component}"));

    // 3. Достаём нужный DLL из bin/ архива.
    let extracted = cache.join("cuda_extract");
    unzip(&zip, &extracted);
    let stem = zip_name.strip_suffix(".zip").expect("имя zip");
    let src = extracted.join(stem).join("bin").join(dll);
    if !src.exists() {
        panic!("в архиве {zip_name} нет bin/{dll}");
    }
    fs::copy(&src, &dest).unwrap_or_else(|e| panic!("copy {dll}: {e}"));
}

fn ensure_onnx_dlls(cache: &Path, resources: &Path) {
    let ort = resources.join("onnxruntime.dll");
    if ort.exists() {
        return;
    }
    let url =
        format!("https://github.com/microsoft/onnxruntime/releases/download/v1.24.4/{ONNX_ZIP}");
    let zip = download(cache, ONNX_ZIP, &url);
    verify_sha(&zip, ONNX_ZIP_SHA256, "ONNX Runtime zip");
    let extracted = cache.join("ort_extract");
    unzip(&zip, &extracted);
    let stem = ONNX_ZIP.strip_suffix(".zip").expect("имя zip");
    let lib = extracted.join(stem).join("lib");
    for dll in ["onnxruntime.dll", "onnxruntime_providers_shared.dll"] {
        fs::copy(lib.join(dll), resources.join(dll)).unwrap_or_else(|e| panic!("copy {dll}: {e}"));
    }
}

/// Qdrant sidecar: Tauri externalBin требует имя <bin>-<TARGET_TRIPLE>.exe
/// в src-tauri/binaries/.
fn ensure_qdrant_sidecar(cache: &Path, binaries: &Path) {
    let target = env::var("TARGET").expect("TARGET");
    let dest = binaries.join(format!("qdrant-{target}.exe"));
    if dest.exists() {
        return;
    }
    let url = format!("https://github.com/qdrant/qdrant/releases/download/v1.19.0/{QDRANT_ZIP}");
    let zip = download(cache, QDRANT_ZIP, &url);
    verify_sha(&zip, QDRANT_ZIP_SHA256, "Qdrant zip");
    let extracted = cache.join("qdrant_extract");
    unzip(&zip, &extracted);
    fs::create_dir_all(binaries).expect("создание binaries dir");
    fs::copy(extracted.join("qdrant.exe"), &dest)
        .unwrap_or_else(|e| panic!("copy qdrant.exe: {e}"));
}

/// Данные базы знаний: векторный стор Qdrant (коллекция mentor_kb) и тексты
/// чанков. Это данные мейнтейнера — из сети не качаются, копируются из
/// локального дерева проекта. Переопределяется env QDRANT_STORAGE_SRC /
/// KB_CHUNKS_SRC.
fn ensure_kb_data(workspace: &Path, resources: &Path) {
    let storage_src = env::var("QDRANT_STORAGE_SRC").map_or_else(
        |_| workspace.join("tools_bin/qdrant_server/storage"),
        PathBuf::from,
    );
    copy_tree_if_missing(&storage_src, &resources.join("qdrant-storage"));

    // ../kb_chunks относительно корня workspace (см. config.toml).
    let chunks_src = env::var("KB_CHUNKS_SRC").map_or_else(
        |_| {
            workspace
                .parent()
                .expect("workspace имеет родителя")
                .join("kb_chunks")
        },
        PathBuf::from,
    );
    copy_tree_if_missing(&chunks_src, &resources.join("kb_chunks"));

    // Шаблон config.toml для первого запуска (пути внутри будут переписаны
    // на AppData при инициализации приложения). Всегда перезаписывается из
    // корневого config.toml репозитория — единый источник правды.
    let defaults = resources.join("defaults");
    fs::create_dir_all(&defaults).expect("создание defaults dir");
    let cfg_src = workspace.join("config.toml");
    let cfg_dest = defaults.join("config.toml");
    if cfg_src.exists() {
        fs::copy(&cfg_src, &cfg_dest).expect("копирование шаблона config.toml");
    }
}

/// Копирует дерево src -> dest, только если dest ещё не существует
/// (данные «первого запуска», пересборка их не затирает).
fn copy_tree_if_missing(src: &Path, dest: &Path) {
    if !src.exists() {
        panic!("исходник данных базы знаний не найден: {}", src.display());
    }
    if dest.exists() {
        return;
    }
    copy_tree(src, dest);
}

fn copy_tree(src: &Path, dest: &Path) {
    fs::create_dir_all(dest).unwrap_or_else(|e| panic!("mkdir {}: {e}", dest.display()));
    for entry in fs::read_dir(src).unwrap_or_else(|e| panic!("readdir {}: {e}", src.display())) {
        let entry = entry.expect("dirent");
        let from = entry.path();
        let to = dest.join(entry.file_name());
        if from.is_dir() {
            copy_tree(&from, &to);
        } else {
            fs::copy(&from, &to).unwrap_or_else(|e| panic!("copy {}: {e}", from.display()));
        }
    }
}
