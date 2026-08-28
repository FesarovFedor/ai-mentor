//! Tauri-РєРѕРјР°РЅРґС‹: РјРѕСЃС‚ РјРµР¶РґСѓ С„СЂРѕРЅС‚РµРЅРґРѕРј (frontend/) Рё СЏРґСЂРѕРј mentor-core.
//!
//! Р­С‚Р°Рї L5: self-contained СѓСЃС‚Р°РЅРѕРІРєР° вЂ” Qdrant РїРѕРґРЅРёРјР°РµС‚СЃСЏ РєР°Рє sidecar
//! (РјРѕРґСѓР»СЊ qdrant), РґР°РЅРЅС‹Рµ Р¶РёРІСѓС‚ РІ AppData, РїРѕСЂС‚ РІС‹Р±РёСЂР°РµС‚СЃСЏ РґРёРЅР°РјРёС‡РµСЃРєРё.
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
// AsyncMutex вЂ” С‚РѕР»СЊРєРѕ РґР»СЏ РіРІР°СЂРґРѕРІ, СѓРґРµСЂР¶РёРІР°РµРјС‹С… С‡РµСЂРµР· .await (rag-РїРѕРёСЃРє,
// СЃРµСЂРёР°Р»РёР·Р°С†РёСЏ Р·Р°РіСЂСѓР·РєРё РјРѕРґРµР»Рё). РљРѕСЂРѕС‚РєРёРµ СЃРёРЅС…СЂРѕРЅРЅС‹Рµ СЃРµРєС†РёРё вЂ” parking_lot.
use tokio::sync::Mutex as AsyncMutex;

use mentor_core::config::{save_generation_fields, save_string_field, AppConfig};
use mentor_core::downloader::{filename_from_url, CancelToken, DownloadSpec, Downloader};
use mentor_core::generator::{generate_response_streaming, StreamKind};
use mentor_core::inference::{gguf_display_name, Inference, LlamaBackend};
use mentor_core::rag::{format_context, Rag, SearchHit};

/// Р Р°Р·РґРµР»СЏРµРјРѕРµ СЃРѕСЃС‚РѕСЏРЅРёРµ РїСЂРёР»РѕР¶РµРЅРёСЏ: RAG-СЏРґСЂРѕ + РєРѕРЅС„РёРі + Р·Р°РіСЂСѓР·С‡РёРє РјРѕРґРµР»Рё.
///
/// Р’Р»Р°РґРµР»РµС† РµРґРёРЅСЃС‚РІРµРЅРЅРѕРіРѕ LlamaBackend: Arc РґСѓР±Р»РёСЂСѓРµС‚СЃСЏ РІ РєР°Р¶РґСѓСЋ Inference,
/// РїРѕСЌС‚РѕРјСѓ Р±СЌРєРµРЅРґ Р¶РёРІС‘С‚, РїРѕРєР° Р¶РёРІР° С…РѕС‚СЊ РѕРґРЅР° РјРѕРґРµР»СЊ, Рё РіР°СЂР°РЅС‚РёСЂРѕРІР°РЅРЅРѕ
/// РґСЂРѕРїР°РµС‚СЃСЏ РІРјРµСЃС‚Рµ СЃ AppState РїСЂРё Р·Р°РІРµСЂС€РµРЅРёРё РїСЂРёР»РѕР¶РµРЅРёСЏ (RunEvent::Exit) вЂ”
/// CUDA-РєРѕРЅС‚РµРєСЃС‚ РѕСЃРІРѕР±РѕР¶РґР°РµС‚ VRAM (С„РёРєСЃ P0 СѓС‚РµС‡РєРё РёР· Р°СѓРґРёС‚Р°, СЌС‚Р°Рї L4).
pub struct AppState {
    pub backend: Arc<LlamaBackend>,
    /// RAG РґРµСЂР¶РёС‚СЃСЏ С‡РµСЂРµР· .await (РїРѕРёСЃРє РїРѕ Qdrant) вЂ” AsyncMutex.
    pub rag: AsyncMutex<Rag>,
    /// РљРѕРЅС„РёРі РїРѕРґ РјСЊСЋС‚РµРєСЃРѕРј: РѕР±РЅРѕРІР»СЏРµС‚СЃСЏ РїРѕСЃР»Рµ РІС‹Р±РѕСЂР°/СЃРєР°С‡РёРІР°РЅРёСЏ РјРѕРґРµР»Рё.
    /// РљРѕСЂРѕС‚РєРёРµ СЃРµРєС†РёРё Р±РµР· .await вЂ” parking_lot.
    pub cfg: Mutex<AppConfig>,
    /// РџСѓС‚СЊ Рє config.toml РІ AppData (Р·Р°РїРёСЃСЊ model_path РїРѕСЃР»Рµ РІС‹Р±РѕСЂР°/СЃРєР°С‡РёРІР°РЅРёСЏ).
    pub cfg_path: PathBuf,
    /// Р—Р°РіСЂСѓР¶РµРЅРЅР°СЏ LLM Рё РїСѓС‚СЊ, РїРѕ РєРѕС‚РѕСЂРѕРјСѓ РѕРЅР° Р·Р°РіСЂСѓР¶РµРЅР° (РґР»СЏ РїРµСЂРµР·Р°РіСЂСѓР·РєРё
    /// РїСЂРё СЃРјРµРЅРµ model_path). РњРѕРґРµР»СЊ Р»РµРЅРёРІРѕ РіСЂСѓР·РёС‚СЃСЏ РЅР° РїРµСЂРІРѕРј РІРѕРїСЂРѕСЃРµ.
    pub llm: Mutex<Option<(PathBuf, Arc<Inference>)>>,
    /// РЎРµСЂРёР°Р»РёР·СѓРµС‚ Р·Р°РіСЂСѓР·РєСѓ/РїРµСЂРµР·Р°РіСЂСѓР·РєСѓ РјРѕРґРµР»Рё: Р±РµР· РЅРµРіРѕ РґРІР° РїР°СЂР°Р»Р»РµР»СЊРЅС‹С…
    /// send_message РїСЂРё СЃРјРµРЅРµ РјРѕРґРµР»Рё Р·Р°РіСЂСѓР·РёР»Рё Р±С‹ РµС‘ РґРІР°Р¶РґС‹. Р“РІР°СЂРґ РґРµСЂР¶РёС‚СЃСЏ
    /// С‡РµСЂРµР· .await (С‚Р°Рј РіСЂСѓР·РёС‚СЃСЏ РјРѕРґРµР»СЊ) вЂ” AsyncMutex.
    pub llm_load_lock: AsyncMutex<()>,
    /// РћС‚РјРµРЅР° Р°РєС‚РёРІРЅРѕР№ Р·Р°РіСЂСѓР·РєРё.
    pub download_cancel: Mutex<CancelToken>,
    /// РРґС‘С‚ Р»Рё СЃРµР№С‡Р°СЃ Р·Р°РіСЂСѓР·РєР° (Р·Р°С‰РёС‚Р° РѕС‚ РїРѕРІС‚РѕСЂРЅРѕРіРѕ Р·Р°РїСѓСЃРєР°).
    pub download_active: AtomicBool,
    /// РџРѕСЃР»РµРґРЅРёР№ СЃРЅРёРјРѕРє РїСЂРѕРіСЂРµСЃСЃР° Р·Р°РіСЂСѓР·РєРё РґР»СЏ РѕРїСЂРѕСЃР° С„СЂРѕРЅС‚РѕРј (СЃС‚СЂР°С…РѕРІРєР° РЅР°
    /// СЃР»СѓС‡Р°Р№, РµСЃР»Рё С€РёРЅР° СЃРѕР±С‹С‚РёР№ РЅРµРґРѕСЃС‚СѓРїРЅР° РІРѕ РІРµР±РІСЊСЋ). РљРѕСЂРѕС‚РєРёРµ СЃРµРєС†РёРё,
    /// РЅСѓР¶РµРЅ Рё РёР· СЃРёРЅС…СЂРѕРЅРЅРѕРіРѕ РєРѕР»Р±СЌРєР° вЂ” parking_lot::Mutex.
    pub download_progress: Mutex<Option<DownloadEvent>>,
    /// РќР°РєРѕРїР»РµРЅРЅС‹Р№ РїРѕС‚РѕРє РіРµРЅРµСЂР°С†РёРё (thinking/answer) РґР»СЏ РѕРїСЂРѕСЃР° С„СЂРѕРЅС‚РѕРј вЂ”
    /// С‚РѕС‚ Р¶Рµ Р·Р°РїР°СЃРЅРѕР№ РєР°РЅР°Р», С‡С‚Рѕ Сѓ download_progress. РђР±СЃРѕР»СЋС‚РЅС‹Рµ Р·РЅР°С‡РµРЅРёСЏ:
    /// С„СЂРѕРЅС‚ СЃРІРµСЂСЏРµС‚ РґР»РёРЅС‹ Рё РґРѕР·Р°Р±РёСЂР°РµС‚ РЅРµРґРѕСЃС‚Р°СЋС‰РµРµ.
    pub gen_stream: Mutex<GenStreamSnapshot>,
    /// Р”РѕС‡РµСЂРЅРёР№ РїСЂРѕС†РµСЃСЃ Qdrant sidecar: РѕСЃС‚Р°РЅР°РІР»РёРІР°РµС‚СЃСЏ РІ RunEvent::Exit.
    pub qdrant: qdrant::QdrantProc,
    /// Pre-flight (Р­С‚Р°Рї L5, С€Р°Рі 6): РµСЃС‚СЊ Р»Рё NVIDIA-РґСЂР°Р№РІРµСЂ РІ СЃРёСЃС‚РµРјРµ.
    /// false -> РёРЅС„РµСЂРµРЅСЃ Р±Р»РѕРєРёСЂСѓРµС‚СЃСЏ СЃ РґСЂСѓР¶РµР»СЋР±РЅРѕР№ РѕС€РёР±РєРѕР№ (РґРµРіСЂР°РґР°С†РёСЏ
    /// РІС‹Р±СЂР°РЅР° РєР°Рє "Р±Р»РѕРєРёСЂРѕРІРєР°", РЅРµ CPU: РїРѕР»РЅС‹Р№ РѕС„С„Р»РѕР°Рґ 3B-РјРѕРґРµР»Рё РЅР° CPU
    /// Р·Р°РЅСЏР» Р±С‹ РјРёРЅСѓС‚С‹ РЅР° С‚РѕРєРµРЅ Рё РІС‹РіР»СЏРґРµР» Р±С‹ РєР°Рє Р·Р°РІРёСЃР°РЅРёРµ).
    pub gpu_ready: AtomicBool,
}

#[derive(Serialize)]
pub struct ChatReply {
    pub answer: String,
    /// РҐРѕРґ СЂР°СЃСЃСѓР¶РґРµРЅРёР№ РјРѕРґРµР»Рё (<think>вЂ¦</think>); РїСѓСЃС‚ Сѓ РЅРѕРЅ-reasoning
    /// РјРѕРґРµР»РµР№. Р¤СЂРѕРЅС‚РµРЅРґ СЂРµРЅРґРµСЂРёС‚ РµРіРѕ РѕС‚РґРµР»СЊРЅС‹Рј СЃРІРѕСЂР°С‡РёРІР°РµРјС‹Рј Р±Р»РѕРєРѕРј.
    pub thinking: String,
    pub sources: Vec<SearchHit>,
    /// РџРѕР»РЅС‹Р№ РїСЂРѕРјРїС‚, РєРѕС‚РѕСЂС‹Р№ РїРѕР»СѓС‡РёР»(Р° Р±С‹) LLM.
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
    pub generator_stub: bool,
}

/// РЎРѕСЃС‚РѕСЏРЅРёРµ РјРѕРґРµР»Рё РґР»СЏ СЃС‚Р°СЂС‚РѕРІРѕРіРѕ СЌРєСЂР°РЅР° (С‡РёС‚Р°РµС‚ РєРѕРЅС„РёРі СЃ РґРёСЃРєР° вЂ” РІСЃРµРіРґР° СЃРІРµР¶РёР№).
#[derive(Serialize, Clone)]
pub struct ModelStatus {
    /// РњРѕРґРµР»СЊ РіРѕС‚РѕРІР°: РїСѓС‚СЊ Р·Р°РґР°РЅ Рё С„Р°Р№Р» СЃСѓС‰РµСЃС‚РІСѓРµС‚.
    pub found: bool,
    pub path: String,
    /// РџР»РµР№СЃС…РѕР»РґРµСЂ URL РёР· config.toml (РїСѓСЃС‚ -> СЃРєР°С‡РёРІР°РЅРёРµ РЅРµРґРѕСЃС‚СѓРїРЅРѕ).
    pub download_url: String,
    /// Р—Р°РґР°РЅР° Р»Рё РєРѕРЅС‚СЂРѕР»СЊРЅР°СЏ СЃСѓРјРјР° РґР»СЏ РїСЂРѕРІРµСЂРєРё РїРѕСЃР»Рµ СЃРєР°С‡РёРІР°РЅРёСЏ.
    pub sha256_set: bool,
}

/// РЎРѕР±С‹С‚РёРµ РїСЂРѕРіСЂРµСЃСЃР° Р·Р°РіСЂСѓР·РєРё РІ РІРµР±РІСЊСЋ (РєР°РЅР°Р» "download-progress").
#[derive(Serialize, Clone)]
pub struct DownloadEvent {
    pub downloaded: u64,
    /// 0 = СЂР°Р·РјРµСЂ РЅРµРёР·РІРµСЃС‚РµРЅ, РїСЂРѕС†РµРЅС‚ РЅРµ РїРѕСЃС‡РёС‚Р°С‚СЊ.
    pub total: u64,
    pub resumed_from: u64,
    pub done: bool,
    pub error: Option<String>,
}

/// РЎРѕР±С‹С‚РёРµ РїРѕС‚РѕРєРѕРІРѕР№ РіРµРЅРµСЂР°С†РёРё РІ РІРµР±РІСЊСЋ (РєР°РЅР°Р» "gen-token"): РѕРґРёРЅ РєСѓСЃРѕРє
/// С‚РµРєСЃС‚Р°, СЂР°Р·РјРµС‡РµРЅРЅС‹Р№ РїРѕ Р±Р»РѕРєСѓ (thinking/answer) РЅР° Р±СЌРєРµРЅРґРµ.
#[derive(Serialize, Clone)]
pub struct GenTokenEvent {
    /// "think" | "answer"
    pub kind: String,
    pub text: String,
}

/// РЎРЅРёРјРѕРє РЅР°РєРѕРїР»РµРЅРЅРѕРіРѕ СЃС‚СЂРёРјР° РіРµРЅРµСЂР°С†РёРё (Р·Р°РїР°СЃРЅРѕР№ РєР°РЅР°Р» РґР»СЏ РѕРїСЂРѕСЃР°).
#[derive(Serialize, Clone, Default)]
pub struct GenStreamSnapshot {
    pub thinking: String,
    pub answer: String,
    /// Р“РµРЅРµСЂР°С†РёСЏ Р·Р°РІРµСЂС€РµРЅР° (С„СЂРѕРЅС‚ РјРѕР¶РµС‚ РїСЂРµРєСЂР°С‚РёС‚СЊ РѕРїСЂРѕСЃ).
    pub done: bool,
}

fn read_model_status(cfg_path: &std::path::Path) -> Result<ModelStatus, String> {
    let cfg = AppConfig::load(cfg_path).map_err(|e| format!("config.toml: {e:#}"))?;
    Ok(ModelStatus {
        found: cfg.model_ready(),
        path: cfg.model_file_path().to_string_lossy().into_owned(),
        download_url: cfg.model_download_url.clone(),
        sha256_set: !cfg.model_sha256.trim().is_empty(),
    })
}

/// РљР°С‚Р°Р»РѕРі РїРѕР»СЊР·РѕРІР°С‚РµР»СЊСЃРєРёС… РґР°РЅРЅС‹С…: %APPDATA%\ai-mentor (Р­С‚Р°Рї L5, С€Р°Рі 7).
/// Р’СЃРµ РїРѕР»СЊР·РѕРІР°С‚РµР»СЊСЃРєРёРµ РґР°РЅРЅС‹Рµ (config.toml, РјРѕРґРµР»Рё, РєСЌС€ СЌРјР±РµРґРґРёРЅРіРѕРІ,
/// storage РІРµРєС‚РѕСЂРЅРѕР№ Р‘Р”) Р¶РёРІСѓС‚ Р·РґРµСЃСЊ вЂ” РєР°С‚Р°Р»РѕРі СѓСЃС‚Р°РЅРѕРІРєРё РјРѕР¶РµС‚ Р±С‹С‚СЊ
/// read-only (Program Files).
fn app_data_dir() -> PathBuf {
    let base = std::env::var("APPDATA")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir());
    base.join("ai-mentor")
}

/// РџРµСЂРІС‹Р№ Р·Р°РїСѓСЃРє: РµСЃР»Рё config.toml РµС‰С‘ РЅРµС‚ РІ AppData вЂ” СЂР°Р·РІРѕСЂР°С‡РёРІР°РµРј РµРіРѕ РёР·
/// Р±Р°РЅРґР»Р° (РїСѓС‚Рё kb_chunks РїРµСЂРµРїРёСЃС‹РІР°СЋС‚СЃСЏ СЃ ../kb_chunks РЅР° AppData) Рё
/// РєРѕРїРёСЂСѓРµРј РґР°РЅРЅС‹Рµ Р±Р°Р·С‹ Р·РЅР°РЅРёР№ (С‚РµРєСЃС‚С‹ С‡Р°РЅРєРѕРІ + РІРµРєС‚РѕСЂРЅС‹Р№ СЃС‚РѕСЂ СЃ
/// РєРѕР»Р»РµРєС†РёРµР№ mentor_kb). РџРѕРІС‚РѕСЂРЅС‹Рµ Р·Р°РїСѓСЃРєРё РЅРёС‡РµРіРѕ РЅРµ РїРµСЂРµР·Р°С‚РёСЂР°СЋС‚.
fn provision_first_run(resource_dir: &Path) -> Result<PathBuf, anyhow::Error> {
    let dir = app_data_dir();
    fs::create_dir_all(&dir).with_context(|| format!("mkdir {}", dir.display()))?;
    let cfg_path = dir.join("config.toml");
    if !cfg_path.exists() {
        let template_path = resource_dir.join("defaults").join("config.toml");
        let template = fs::read_to_string(&template_path).with_context(|| {
            format!(
                "РІ Р±Р°РЅРґР»Рµ РЅРµС‚ С€Р°Р±Р»РѕРЅР° РєРѕРЅС„РёРіР° {}",
                template_path.display()
            )
        })?;
        // kb_chunks РІ Р±Р°РЅРґР»Рµ Р»РµР¶Р°С‚ СЂСЏРґРѕРј СЃ config.toml РІ AppData.
        let rewritten = template.replace("../kb_chunks", "kb_chunks");
        fs::write(&cfg_path, rewritten)
            .with_context(|| format!("Р·Р°РїРёСЃСЊ {}", cfg_path.display()))?;
    }
    copy_missing(
        &resource_dir.join("kb_chunks"),
        &dir.join("kb_chunks"),
        "С‚РµРєСЃС‚С‹ С‡Р°РЅРєРѕРІ",
    )?;
    copy_missing(
        &resource_dir.join("qdrant-storage"),
        &dir.join("qdrant").join("storage"),
        "РІРµРєС‚РѕСЂРЅС‹Р№ СЃС‚РѕСЂ Qdrant",
    )?;
    Ok(cfg_path)
}

fn copy_missing(src: &Path, dst: &Path, what: &str) -> Result<(), anyhow::Error> {
    if !src.exists() {
        anyhow::bail!("{what}: РІ Р±Р°РЅРґР»Рµ РЅРµС‚ {}", src.display());
    }
    if dst.exists() {
        return Ok(());
    }
    copy_tree(src, dst)
        .with_context(|| format!("РєРѕРїРёСЂРѕРІР°РЅРёРµ {what} РІ {}", dst.display()))
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

/// РџРµСЂРµС‡РёС‚С‹РІР°РµС‚ config.toml СЃ РґРёСЃРєР° РІ AppState, РЎРћРҐР РђРќРЇРЇ РґРёРЅР°РјРёС‡РµСЃРєРёР№
/// Р°РґСЂРµСЃ Qdrant: РїРѕСЂС‚ РІС‹Р±РёСЂР°РµС‚СЃСЏ РІ СЂР°РЅС‚Р°Р№РјРµ Рё РЅР° РґРёСЃРє РЅРµ РїРёС€РµС‚СЃСЏ (L5,
/// С€Р°Рі 5), РїРѕСЌС‚РѕРјСѓ РїРѕСЃР»Рµ РєР°Р¶РґРѕРіРѕ reload РµРіРѕ РЅСѓР¶РЅРѕ РЅР°РєР»Р°РґС‹РІР°С‚СЊ Р·Р°РЅРѕРІРѕ.
fn reload_cfg_preserving_port(app: &AppState) -> Result<(), String> {
    let mut fresh = AppConfig::load(&app.cfg_path).map_err(|e| format!("config.toml: {e:#}"))?;
    fresh.qdrant.url = app.cfg.lock().qdrant.url.clone();
    *app.cfg.lock() = fresh;
    Ok(())
}

/// Pre-flight (С€Р°Рі 6): РїСЂРѕРІРµСЂСЏРµРј РЅР°Р»РёС‡РёРµ NVIDIA-РґСЂР°Р№РІРµСЂР° (nvcuda.dll вЂ”
/// РєР»РёРµРЅС‚СЃРєР°СЏ Р±РёР±Р»РёРѕС‚РµРєР° CUDA СЃС‚Р°РІРёС‚СЃСЏ РўРћР›Р¬РљРћ РІРјРµСЃС‚Рµ СЃ РґСЂР°Р№РІРµСЂРѕРј NVIDIA).
pub fn gpu_driver_available() -> bool {
    let system32 = std::env::var("SystemRoot").map_or_else(
        |_| PathBuf::from(r"C:\Windows"),
        |root| PathBuf::from(root).join("System32"),
    );
    system32.join("nvcuda.dll").is_file()
}

/// РљР°С‚Р°Р»РѕРі СЂРµСЃСѓСЂСЃРѕРІ Р±Р°РЅРґР»Р°: Tauri РєР»Р°РґС‘С‚ СЂРµСЃСѓСЂСЃС‹ (DLL, qdrant-storage,
/// kb_chunks, defaults/) СЂСЏРґРѕРј СЃ РѕСЃРЅРѕРІРЅС‹Рј Р±РёРЅР°СЂРµРј вЂ” Рё РІ MSI, Рё РїСЂРё dev-Р·Р°РїСѓСЃРєРµ.
fn resource_dir() -> Result<PathBuf, anyhow::Error> {
    Ok(std::env::current_exe()
        .context("РЅРµ СѓРґР°Р»РѕСЃСЊ РѕРїСЂРµРґРµР»РёС‚СЊ РєР°С‚Р°Р»РѕРі РїСЂРёР»РѕР¶РµРЅРёСЏ")?
        .parent()
        .expect("exe Р»РµР¶РёС‚ РІ РєР°С‚Р°Р»РѕРіРµ")
        .to_path_buf())
}

/// РРЅРёС†РёР°Р»РёР·Р°С†РёСЏ СЃРѕСЃС‚РѕСЏРЅРёСЏ: РїРµСЂРІС‹Р№ Р·Р°РїСѓСЃРє -> Qdrant sidecar -> RAG -> РјРѕРґРµР»СЊ.
/// Р’С‹Р·С‹РІР°РµС‚СЃСЏ РёР· setup-С…СѓРєР° Р”Рћ СЃРѕР·РґР°РЅРёСЏ РѕРєРЅР°: РїРѕР»СЊР·РѕРІР°С‚РµР»СЊ РЅРµ РІРёРґРёС‚
/// "Р·Р°РІРёСЃС€РµРіРѕ" РѕРєРЅР°, Р° Р»СЋР±С‹Рµ РѕС€РёР±РєРё РїРѕРєР°Р·С‹РІР°СЋС‚СЃСЏ РґРёР°Р»РѕРіРѕРј (РЅРµ РјРѕР»С‡Р°).
async fn init_state() -> Result<AppState, String> {
    let resource_dir =
        resource_dir().map_err(|e| format!("РєР°С‚Р°Р»РѕРі СЂРµСЃСѓСЂСЃРѕРІ: {e:#}"))?;
    // РўСЏР¶С‘Р»РѕРµ РєРѕРїРёСЂРѕРІР°РЅРёРµ РґР°РЅРЅС‹С… РїРµСЂРІРѕРіРѕ Р·Р°РїСѓСЃРєР° (РґРѕ ~600 РњР‘ СЃС‚РѕСЂ Qdrant) вЂ”
    // РІ Р±Р»РѕРєРёСЂСѓСЋС‰РµРј РїСѓР»Рµ, main-РїРѕС‚РѕРє РЅРµ Р·Р°РЅСЏС‚ С„Р°Р№Р»РѕРІС‹Рј I/O.
    let res_for_provision = resource_dir.clone();
    let cfg_path =
        tauri::async_runtime::spawn_blocking(move || provision_first_run(&res_for_provision))
            .await
            .map_err(|e| format!("РїРѕС‚РѕРє РїРµСЂРІРѕРіРѕ Р·Р°РїСѓСЃРєР°: {e}"))?
            .map_err(|e| format!("РїРµСЂРІС‹Р№ Р·Р°РїСѓСЃРє: {e:#}"))?;

    let mut cfg = AppConfig::load(&cfg_path).map_err(|e| format!("config.toml: {e:#}"))?;

    // Pre-flight РґСЂР°Р№РІРµСЂР°: РјРѕРґРµР»СЊ РіСЂСѓР·РёРј С‚РѕР»СЊРєРѕ РµСЃР»Рё РµСЃС‚СЊ NVIDIA.
    let gpu_ready = gpu_driver_available();
    let backend = Arc::new(
        mentor_core::inference::init_backend()
            .map_err(|e| format!("РёРЅРёС†РёР°Р»РёР·Р°С†РёСЏ llama.cpp backend: {e:#}"))?,
    );

    // Qdrant sidecar: РґРёРЅР°РјРёС‡РµСЃРєРёР№ РїРѕСЂС‚ + AppData-СЃС‚РѕСЂ; Р¶РґС‘Рј readiness.
    let qproc = qdrant::QdrantProc::new();
    let storage = app_data_dir().join("qdrant").join("storage");
    let sidecar_exe = qdrant::sidecar_exe_path().map_err(|e| format!("Qdrant: {e:#}"))?;
    let (http_port, grpc_port) =
        qdrant::start_qdrant(&qproc, &sidecar_exe, &storage).map_err(|e| {
            qdrant::stop_qdrant(&qproc);
            format!("Р·Р°РїСѓСЃРє Qdrant: {e:#}")
        })?;
    qdrant::wait_for_ready(http_port, Duration::from_secs(30))
        .await
        .map_err(|e| {
            qdrant::stop_qdrant(&qproc);
            format!("Qdrant: {e:#}")
        })?;
    // РџРѕСЂС‚ вЂ” РІРµР»РёС‡РёРЅР° РІСЂРµРјРµРЅРё РёСЃРїРѕР»РЅРµРЅРёСЏ: РїРµСЂРµРѕРїСЂРµРґРµР»СЏРµРј Р°РґСЂРµСЃ РўРћР›Р¬РљРћ РІ
    // РїР°РјСЏС‚Рё (config.toml РЅР° РґРёСЃРєРµ РѕСЃС‚Р°С‘С‚СЃСЏ СЃ РґРµС„РѕР»С‚РЅС‹Рј 6334).
    cfg.qdrant.url = format!("http://127.0.0.1:{grpc_port}");

    let rag = Rag::new(cfg.clone())
        .await
        .map_err(|e| format!("РёРЅРёС†РёР°Р»РёР·Р°С†РёСЏ RAG: {e:#}"))?;

    let llm = if gpu_ready && cfg.model_ready() {
        let path_str = cfg.model_file_path().to_string_lossy().into_owned();
        let backend = backend.clone();
        match tauri::async_runtime::spawn_blocking(move || {
            mentor_core::inference::load_model(&path_str, &backend)
        })
        .await
        .map_err(|e| format!("РїРѕС‚РѕРє РїСЂРµРґР·Р°РіСЂСѓР·РєРё РјРѕРґРµР»Рё: {e}"))?
        {
            Ok(inf) => Some((cfg.model_file_path(), Arc::new(inf))),
            Err(e) => {
                eprintln!("РїСЂРµРґР·Р°РіСЂСѓР·РєР° РјРѕРґРµР»Рё РЅРµ СѓРґР°Р»Р°СЃСЊ: {e:#}");
                None // РїСЂРёС‡РёРЅР° СѓРІРёРґРёС‚ РїРѕР»СЊР·РѕРІР°С‚РµР»СЊ РїСЂРё РїРµСЂРІРѕР№ РѕС‚РїСЂР°РІРєРµ РІРѕРїСЂРѕСЃР°
            }
        }
    } else {
        None
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
        qdrant: qproc,
        gpu_ready: AtomicBool::new(gpu_ready),
    })
}

/// Р’РѕР·РІСЂР°С‰Р°РµС‚ Р·Р°РіСЂСѓР¶РµРЅРЅСѓСЋ РјРѕРґРµР»СЊ, РїСЂРё РЅРµРѕР±С…РѕРґРёРјРѕСЃС‚Рё РіСЂСѓР·РёС‚ РµС‘ РІ С„РѕРЅРµ
/// (Р»РµРЅРёРІРѕ РЅР° РїРµСЂРІРѕРј РІРѕРїСЂРѕСЃРµ) РёР»Рё РїРµСЂРµР·Р°РіСЂСѓР¶Р°РµС‚ РїРѕСЃР»Рµ СЃРјРµРЅС‹ model_path.
/// Р’СЃСЏ РїРѕСЃР»РµРґРѕРІР°С‚РµР»СЊРЅРѕСЃС‚СЊ "РїСЂРѕРІРµСЂРєР° -> РІС‹РіСЂСѓР·РєР° -> Р·Р°РіСЂСѓР·РєР° -> Р·Р°РїРёСЃСЊ"
/// РґРµСЂР¶РёС‚ llm_load_lock: РїР°СЂР°Р»Р»РµР»СЊРЅС‹Рµ Р·Р°РїСЂРѕСЃС‹ РЅРµ РјРѕРіСѓС‚ Р·Р°РіСЂСѓР·РёС‚СЊ РјРѕРґРµР»СЊ
/// РґРІР°Р¶РґС‹ РёР»Рё СѓРІРёРґРµС‚СЊ РїСЂРѕРјРµР¶СѓС‚РѕС‡РЅРѕРµ None.
async fn ensure_llm(app: &Arc<AppState>, model_path: PathBuf) -> Result<Arc<Inference>, String> {
    // Pre-flight (L5, С€Р°Рі 6): Р±РµР· NVIDIA-РґСЂР°Р№РІРµСЂР° РЅРµ РїС‹С‚Р°РµРјСЃСЏ РіСЂСѓР·РёС‚СЊ
    // РјРѕРґРµР»СЊ СЃ РїРѕР»РЅС‹Рј GPU-РѕС„С„Р»РѕР°РґРѕРј вЂ” С‡РµСЃС‚РЅР°СЏ РѕС€РёР±РєР° СЃ СЃСЃС‹Р»РєРѕР№ РЅР° РґСЂР°Р№РІРµСЂ.
    if !app.gpu_ready.load(Ordering::SeqCst) {
        return Err(
            "NVIDIA-РґСЂР°Р№РІРµСЂ РЅРµ РЅР°Р№РґРµРЅ (nvcuda.dll РѕС‚СЃСѓС‚СЃС‚РІСѓРµС‚). РРЅС„РµСЂРµРЅСЃ С‚СЂРµР±СѓРµС‚ \
             GPU: СѓСЃС‚Р°РЅРѕРІРёС‚Рµ РґСЂР°Р№РІРµСЂ СЃ https://www.nvidia.com/drivers Рё \
             РїРµСЂРµР·Р°РїСѓСЃС‚РёС‚Рµ РїСЂРёР»РѕР¶РµРЅРёРµ."
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
    // РЎРјРµРЅР° РјРѕРґРµР»Рё: СЃР±СЂР°СЃС‹РІР°РµРј РїСЂРµР¶РЅРёР№ РґРІРёР¶РѕРє Рё РіСЂСѓР·РёРј РЅРѕРІС‹Р№. In-flight
    // Р·Р°РїСЂРѕСЃС‹ РґРµСЂР¶Р°С‚ СЃРѕР±СЃС‚РІРµРЅРЅС‹Р№ Arc СЃС‚Р°СЂРѕР№ РјРѕРґРµР»Рё Рё С‡РµСЃС‚РЅРѕ РґРѕСЃС‡РёС‚С‹РІР°СЋС‚ РЅР°
    // РЅРµР№; РЅРѕРІС‹Рµ Р·Р°РїСЂРѕСЃС‹ РїРѕР»СѓС‡Р°С‚ СѓР¶Рµ РЅРѕРІСѓСЋ.
    *app.llm.lock() = None;
    let path_str = model_path.to_string_lossy().into_owned();
    let backend = app.backend.clone();
    let inf = tauri::async_runtime::spawn_blocking(move || {
        mentor_core::inference::load_model(&path_str, &backend)
    })
    .await
    .map_err(|e| format!("РїРѕС‚РѕРє Р·Р°РіСЂСѓР·РєРё РјРѕРґРµР»Рё: {e}"))?
    .map_err(|e| format!("Р·Р°РіСЂСѓР·РєР° РјРѕРґРµР»Рё: {e:#}"))?;
    let arc = Arc::new(inf);
    *app.llm.lock() = Some((model_path, arc.clone()));
    Ok(arc)
}

/// Р’РѕРїСЂРѕСЃ РїРѕР»СЊР·РѕРІР°С‚РµР»СЏ -> РєРѕРЅС‚РµРєСЃС‚ РёР· Р±Р°Р·С‹ -> РѕС‚РІРµС‚ СЂРµР°Р»СЊРЅРѕР№ LLM.
/// Р“РµРЅРµСЂР°С†РёСЏ СЃС‚СЂРёРјРёС‚СЃСЏ РІРѕ С„СЂРѕРЅС‚РµРЅРґ СЃРѕР±С‹С‚РёСЏРјРё "gen-token" (РїРѕ РєСѓСЃРєСѓ РЅР° С‚РѕРєРµРЅ,
/// СЂР°Р·РјРµС‡РµРЅРЅС‹Р№ thinking/answer); РїР°СЂР°Р»Р»РµР»СЊРЅРѕ РєРѕРїРёС‚СЃСЏ СЃРЅРёРјРѕРє gen_stream РґР»СЏ
/// РѕРїСЂРѕСЃР° РєР°Рє Р·Р°РїР°СЃРЅРѕРіРѕ РєР°РЅР°Р»Р° вЂ” С‚РѕС‚ Р¶Рµ РїР°С‚С‚РµСЂРЅ, С‡С‚Рѕ Сѓ download-progress.
#[tauri::command]
async fn send_message(
    app_handle: AppHandle,
    state: State<'_, Arc<AppState>>,
    question: String,
) -> Result<ChatReply, String> {
    if question.trim().is_empty() {
        return Err("РїСѓСЃС‚РѕР№ РІРѕРїСЂРѕСЃ".into());
    }
    let app = state.inner().clone();
    let cfg = app.cfg.lock().clone();
    if !cfg.model_ready() {
        return Err(
            "РјРѕРґРµР»СЊ РЅРµ РїРѕРґРєР»СЋС‡РµРЅР°: СѓРєР°Р¶Рё .gguf РІСЂСѓС‡РЅСѓСЋ РёР»Рё СЃРєР°С‡Р°Р№ РЅР° СЃС‚Р°СЂС‚РѕРІРѕРј СЌРєСЂР°РЅРµ".into(),
        );
    }
    let llm = ensure_llm(&app, cfg.model_file_path()).await?;

    // 1. СЂРµС‚СЂРёРІ
    let mut rag = app.rag.lock().await;
    let k = cfg.qdrant.top_k as usize;
    let hits = rag
        .search(&question, k)
        .await
        .map_err(|e| format!("РїРѕРёСЃРє РїРѕ Р±Р°Р·Рµ Р·РЅР°РЅРёР№: {e:#}"))?;
    // Р“РµРЅРµСЂР°С†РёСЏ РјРѕР¶РµС‚ Р·Р°РЅРёРјР°С‚СЊ СЃРµРєСѓРЅРґС‹ вЂ” РЅРµ РґРµСЂР¶РёРј RAG-РјСЊСЋС‚РµРєСЃ.
    drop(rag);

    // 2. РєРѕРЅС‚РµРєСЃС‚ РґР»СЏ РїСЂРѕРјРїС‚Р° (РµРґРёРЅС‹Р№ С„РѕСЂРјР°С‚С‚РµСЂ вЂ” СЌС‚Р°Рї K)
    let context = format_context(&hits);

    // 3. СЂРµР°Р»СЊРЅС‹Р№ РёРЅС„РµСЂРµРЅСЃ РІРЅРµ async-РїРѕС‚РѕРєРѕРІ (Р±Р»РѕРєРёСЂСѓСЋС‰Р°СЏ CPU-СЂР°Р±РѕС‚Р°);
    //    РєР°Р¶РґС‹Р№ С‚РѕРєРµРЅ СѓС…РѕРґРёС‚ РІРѕ РІРµР±РІСЊСЋ СЃРѕР±С‹С‚РёРµРј Рё РІ СЃРЅРёРјРѕРє РґР»СЏ РѕРїСЂРѕСЃР°.
    {
        let mut snap = app.gen_stream.lock();
        snap.thinking.clear();
        snap.answer.clear();
        snap.done = false;
    }
    let generation = cfg.generation.clone();
    let q = question.trim().to_string();
    let emitter = app_handle.clone();
    let snap_state = app.clone();
    let generated = tauri::async_runtime::spawn_blocking(move || {
        generate_response_streaming(&llm, &q, &context, &generation, |piece| {
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
        })
    })
    .await
    .map_err(|e| format!("РїРѕС‚РѕРє РёРЅС„РµСЂРµРЅСЃР°: {e}"))?
    .map_err(|e| format!("РёРЅС„РµСЂРµРЅСЃ: {e:#}"))?;
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

/// РЎР»СѓР¶РµР±РЅР°СЏ РёРЅС„РѕСЂРјР°С†РёСЏ РґР»СЏ СЃС‚Р°С‚СѓСЃ-Р±Р°СЂР° С„СЂРѕРЅС‚РµРЅРґР°.
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
    Ok(StatusInfo {
        qdrant_url: cfg.qdrant.url.clone(),
        collection: cfg.qdrant.collection.clone(),
        points,
        top_k: cfg.qdrant.top_k,
        embedding_model: cfg.embedding.model.clone(),
        model_path_set: !cfg.model_path.trim().is_empty(),
        generator_stub: !llm_loaded,
    })
}

/// РџСЂРѕРІРµСЂРєР° РјРѕРґРµР»Рё РґР»СЏ СЂРµС€РµРЅРёСЏ "С‡Р°С‚ РёР»Рё СЌРєСЂР°РЅ РЅР°СЃС‚СЂРѕР№РєРё".
#[tauri::command]
async fn get_model_status(state: State<'_, Arc<AppState>>) -> Result<ModelStatus, String> {
    read_model_status(&state.inner().cfg_path)
}

/// РџРѕСЃР»РµРґРЅРёР№ СЃРЅРёРјРѕРє РїСЂРѕРіСЂРµСЃСЃР° Р·Р°РіСЂСѓР·РєРё (С„СЂРѕРЅС‚РµРЅРґ РѕРїСЂР°С€РёРІР°РµС‚ РєР°Рє Р·Р°РїР°СЃРЅРѕР№
/// РєР°РЅР°Р», РµСЃР»Рё СЃРѕР±С‹С‚РёСЏ РЅРµ РґРѕС…РѕРґСЏС‚).
#[tauri::command]
async fn get_download_progress(
    state: State<'_, Arc<AppState>>,
) -> Result<Option<DownloadEvent>, String> {
    Ok(state.inner().download_progress.lock().clone())
}

/// РЎРЅРёРјРѕРє РЅР°РєРѕРїР»РµРЅРЅРѕРіРѕ СЃС‚СЂРёРјР° РіРµРЅРµСЂР°С†РёРё (Р·Р°РїР°СЃРЅРѕР№ РєР°РЅР°Р» Рє СЃРѕР±С‹С‚РёСЏРј
/// "gen-token", РµСЃР»Рё РѕРЅРё РЅРµ РґРѕС…РѕРґСЏС‚ РґРѕ РІРµР±РІСЊСЋ). РђР±СЃРѕР»СЋС‚РЅС‹Рµ Р·РЅР°С‡РµРЅРёСЏ вЂ”
/// С„СЂРѕРЅС‚ СЃРІРµСЂСЏРµС‚ РґР»РёРЅС‹ Рё РґРѕР·Р°Р±РёСЂР°РµС‚ РЅРµРґРѕСЃС‚Р°СЋС‰РµРµ.
#[tauri::command]
async fn get_gen_progress(state: State<'_, Arc<AppState>>) -> Result<GenStreamSnapshot, String> {
    Ok(state.inner().gen_stream.lock().clone())
}

/// Р”РёР°Р»РѕРі РІС‹Р±РѕСЂР° РіРѕС‚РѕРІРѕРіРѕ .gguf; РїСѓС‚СЊ СЃРѕС…СЂР°РЅСЏРµС‚СЃСЏ РІ config.toml.
/// Р’РѕР·РІСЂР°С‰Р°РµС‚ None, РµСЃР»Рё РїРѕР»СЊР·РѕРІР°С‚РµР»СЊ Р·Р°РєСЂС‹Р» РґРёР°Р»РѕРі Р±РµР· РІС‹Р±РѕСЂР°.
#[tauri::command]
async fn pick_model_file(state: State<'_, Arc<AppState>>) -> Result<Option<ModelStatus>, String> {
    let app = state.inner().clone();
    let picked = tauri::async_runtime::spawn_blocking(|| {
        rfd::FileDialog::new()
            .set_title("Р’С‹Р±РµСЂРё GGUF-С„Р°Р№Р» РјРѕРґРµР»Рё")
            .add_filter("GGUF-РјРѕРґРµР»Рё", &["gguf"])
            .add_filter("Р’СЃРµ С„Р°Р№Р»С‹", &["*"])
            .pick_file()
    })
    .await
    .map_err(|e| format!("РїРѕС‚РѕРє РґРёР°Р»РѕРіР°: {e}"))?;

    let Some(path) = picked else {
        return Ok(None); // РґРёР°Р»РѕРі Р·Р°РєСЂС‹С‚ вЂ” РЅРµ РѕС€РёР±РєР°
    };
    let path_str = path.to_string_lossy().into_owned();
    save_string_field(&app.cfg_path, "model_path", &path_str)
        .map_err(|e| format!("Р·Р°РїРёСЃСЊ config.toml: {e:#}"))?;
    reload_cfg_preserving_port(&app)?;
    Ok(Some(read_model_status(&app.cfg_path)?))
}

/// Р¤РѕРЅРѕРІР°СЏ Р·Р°РґР°С‡Р° СЃРєР°С‡РёРІР°РЅРёСЏ: СЃРѕР±С‹С‚РёСЏ РїСЂРѕРіСЂРµСЃСЃР° РІ РєР°РЅР°Р» "download-progress",
/// РїРѕ СѓСЃРїРµС…Рµ РїСѓС‚СЊ СЃРѕС…СЂР°РЅСЏРµС‚СЃСЏ РІ config.toml. РЎРёРіРЅР°Р» done С€Р»С‘С‚СЃСЏ С‚РѕР»СЊРєРѕ
/// РїРѕСЃР»Рµ Р·Р°РїРёСЃРё РєРѕРЅС„РёРіР°, С‡С‚РѕР±С‹ С„СЂРѕРЅС‚РµРЅРґ РЅРµ СѓРІРёРґРµР» РїСЂРѕРјРµР¶СѓС‚РѕС‡РЅРѕРµ СЃРѕСЃС‚РѕСЏРЅРёРµ.
async fn run_download(
    app_handle: AppHandle,
    app: Arc<AppState>,
    url: String,
    cancel: CancelToken,
) -> Result<(), String> {
    let cfg = AppConfig::load(&app.cfg_path).map_err(|e| format!("config.toml: {e:#}"))?;
    let dest = cfg.download_dir().join(filename_from_url(&url));
    let spec = DownloadSpec {
        url: url.clone(),
        expected_sha256: {
            let sha = cfg.model_sha256.trim();
            (!sha.is_empty()).then(|| sha.to_string())
        },
        dest,
    };

    let downloader = Downloader::new().map_err(|e| format!("HTTP-РєР»РёРµРЅС‚: {e:#}"))?;
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
            Err(String::from("Р·Р°РіСЂСѓР·РєР° РЅРµ Р·Р°РІРµСЂС€РµРЅР°"))
        }
    }
}

/// РЎС‚Р°СЂС‚ СЃРєР°С‡РёРІР°РЅРёСЏ РјРѕРґРµР»Рё РїРѕ URL РёР· config.toml (model_download_url).
#[tauri::command]
async fn start_model_download(
    app_handle: AppHandle,
    state: State<'_, Arc<AppState>>,
) -> Result<(), String> {
    let app = state.inner().clone();
    if app.download_active.swap(true, Ordering::SeqCst) {
        return Err("Р·Р°РіСЂСѓР·РєР° СѓР¶Рµ РёРґС‘С‚".into());
    }
    // РЎРІРµР¶РёР№ РєРѕРЅС„РёРі СЃ РґРёСЃРєР°: URL РјРѕРіР»Рё С‚РѕР»СЊРєРѕ С‡С‚Рѕ РѕС‚СЂРµРґР°РєС‚РёСЂРѕРІР°С‚СЊ.
    let cfg = AppConfig::load(&app.cfg_path).map_err(|e| {
        app.download_active.store(false, Ordering::SeqCst);
        format!("config.toml: {e:#}")
    })?;
    let url = cfg.model_download_url.trim().to_string();
    if url.is_empty() {
        app.download_active.store(false, Ordering::SeqCst);
        return Err(
            "model_download_url РІ config.toml РїСѓСЃС‚ вЂ” Р·Р°РїРѕР»РЅРё РµРіРѕ СЂРµР°Р»СЊРЅС‹Рј Р°РґСЂРµСЃРѕРј .gguf".into(),
        );
    }
    // РќРѕРІС‹Р№ С‚РѕРєРµРЅ РѕС‚РјРµРЅС‹ РЅР° РєР°Р¶РґСѓСЋ Р·Р°РіСЂСѓР·РєСѓ (РїСЂРѕС€Р»Р°СЏ РѕС‚РјРµРЅР° РЅРµ В«РІРёСЃРёС‚В»).
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

/// РћС‚РјРµРЅР° Р°РєС‚РёРІРЅРѕР№ Р·Р°РіСЂСѓР·РєРё (.part СЃРѕС…СЂР°РЅСЏРµС‚СЃСЏ вЂ” СЃР»РµРґСѓСЋС‰РёР№ Р·Р°РїСѓСЃРє РґРѕРєР°С‡Р°РµС‚).
#[tauri::command]
async fn cancel_model_download(state: State<'_, Arc<AppState>>) -> Result<(), String> {
    state.inner().download_cancel.lock().cancel();
    Ok(())
}

/// РЎРЅРёРјРѕРє РЅР°СЃС‚СЂРѕРµРє РіРµРЅРµСЂР°С†РёРё РґР»СЏ РѕРєРЅР° РЅР°СЃС‚СЂРѕРµРє С„СЂРѕРЅС‚РµРЅРґР°.
#[derive(Serialize)]
pub struct SettingsInfo {
    /// РџСѓС‚СЊ Рє .gguf РёР· config.toml (РјРѕР¶РµС‚ Р±С‹С‚СЊ РїСѓСЃС‚).
    pub model_path: String,
    /// РћС‚РѕР±СЂР°Р¶Р°РµРјРѕРµ РёРјСЏ РёР· GGUF-РјРµС‚Р°РґР°РЅРЅС‹С… (general.name); РµСЃР»Рё РјРµС‚Р°РґР°РЅРЅС‹С…
    /// РЅРµС‚ вЂ” РёРјСЏ С„Р°Р№Р»Р°; РµСЃР»Рё С„Р°Р№Р»Р° РЅРµС‚ вЂ” РїСѓСЃС‚Р°СЏ СЃС‚СЂРѕРєР°.
    pub model_name: String,
    pub temperature: f32,
    pub max_tokens: u32,
    pub n_ctx: u32,
}

fn read_settings(cfg_path: &std::path::Path) -> Result<SettingsInfo, String> {
    let cfg = AppConfig::load(cfg_path).map_err(|e| format!("config.toml: {e:#}"))?;
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
    Ok(SettingsInfo {
        model_path: cfg.model_path.clone(),
        model_name,
        temperature: cfg.generation.temperature,
        max_tokens: cfg.generation.max_tokens,
        n_ctx: cfg.generation.n_ctx,
    })
}

/// РўРµРєСѓС‰РёРµ РЅР°СЃС‚СЂРѕР№РєРё: РїСѓС‚СЊ/РёРјСЏ РјРѕРґРµР»Рё + РїР°СЂР°РјРµС‚СЂС‹ [generation].
#[tauri::command]
async fn get_settings(state: State<'_, Arc<AppState>>) -> Result<SettingsInfo, String> {
    read_settings(&state.inner().cfg_path)
}

/// РЎРѕС…СЂР°РЅСЏРµС‚ temperature/max_tokens РІ config.toml (СЃ СЃРѕС…СЂР°РЅРµРЅРёРµРј РєРѕРјРјРµРЅС‚Р°СЂРёРµРІ)
/// Рё РѕР±РЅРѕРІР»СЏРµС‚ РєРѕРЅС„РёРі РІ СЃРѕСЃС‚РѕСЏРЅРёРё. Р’РѕР·РІСЂР°С‰Р°РµС‚ СЃРІРµР¶РёР№ СЃРЅРёРјРѕРє РЅР°СЃС‚СЂРѕРµРє.
#[tauri::command]
async fn set_settings(
    state: State<'_, Arc<AppState>>,
    temperature: f64,
    max_tokens: u32,
) -> Result<SettingsInfo, String> {
    if !(0.0..=2.0).contains(&temperature) {
        return Err("temperature РґРѕР»Р¶РµРЅ Р±С‹С‚СЊ РІ РґРёР°РїР°Р·РѕРЅРµ 0.0вЂ“2.0".into());
    }
    if !(128..=32768).contains(&max_tokens) {
        return Err("max_tokens РґРѕР»Р¶РµРЅ Р±С‹С‚СЊ РІ РґРёР°РїР°Р·РѕРЅРµ 128вЂ“32768".into());
    }
    let app = state.inner().clone();
    let cfg_path = app.cfg_path.clone();
    tauri::async_runtime::spawn_blocking(move || {
        save_generation_fields(&cfg_path, temperature, max_tokens)
    })
    .await
    .map_err(|e| format!("РїРѕС‚РѕРє Р·Р°РїРёСЃРё РєРѕРЅС„РёРіР°: {e}"))?
    .map_err(|e| format!("Р·Р°РїРёСЃСЊ config.toml: {e:#}"))?;
    reload_cfg_preserving_port(&app)?;
    read_settings(&app.cfg_path)
}

pub fn run() {
    // РРЅРёС†РёР°Р»РёР·Р°С†РёСЏ (РїРµСЂРІС‹Р№ Р·Р°РїСѓСЃРє, Qdrant sidecar, RAG, РјРѕРґРµР»СЊ) РґРѕ СЃС‚Р°СЂС‚Р°
    // event loop: РѕРєРЅР° РµС‰С‘ РЅРµС‚, Р·Р°РјРµСЂР·Р°С‚СЊ РЅРµС‡РµРјСѓ; РѕС€РёР±РєРё РїРѕРєР°Р·С‹РІР°СЋС‚СЃСЏ
    // РїРѕРЅСЏС‚РЅС‹Рј РґРёР°Р»РѕРіРѕРј вЂ” РЅРµ РјРѕР»С‡Р° (L5, С€Р°Рі 4).
    let state = match tauri::async_runtime::block_on(init_state()) {
        Ok(s) => s,
        Err(e) => {
            // Р”РёР°РіРЅРѕСЃС‚РёРєР°: С‚РµРєСЃС‚ РѕС€РёР±РєРё РѕСЃС‚Р°С‘С‚СЃСЏ РІ С„Р°Р№Р»Рµ (РІ
            // windows_subsystem=windows eprintln РЅРµРєСѓРґР° РїРёСЃР°С‚СЊ).
            let _ = std::fs::write(
                std::env::temp_dir().join("ai-mentor-init-error.log"),
                format!("{e}\n"),
            );
            let description = format!(
                "РќРµ СѓРґР°Р»РѕСЃСЊ РёРЅРёС†РёР°Р»РёР·РёСЂРѕРІР°С‚СЊ РїСЂРёР»РѕР¶РµРЅРёРµ:\n\n{e}\n\n\
                 РџСЂРёР»РѕР¶РµРЅРёРµ Р±СѓРґРµС‚ Р·Р°РєСЂС‹С‚Рѕ."
            );
            rfd::MessageDialog::new()
                .set_title("AI Mentor вЂ” РѕС€РёР±РєР° Р·Р°РїСѓСЃРєР°")
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
            // РћРєРЅРѕ СЃРѕР·РґР°С‘Рј С‚РѕР»СЊРєРѕ РєРѕРіРґР° СЃРµСЂРІРёСЃС‹ РіРѕС‚РѕРІС‹.
            use tauri::webview::WebviewWindowBuilder;
            use tauri::WebviewUrl;
            WebviewWindowBuilder::new(app.handle(), "main", WebviewUrl::default())
                .title("AI Mentor вЂ” Р»РѕРєР°Р»СЊРЅС‹Р№ РЅР°СЃС‚Р°РІРЅРёРє")
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
            set_settings
        ])
        .build(tauri::generate_context!())
        .expect("РѕС€РёР±РєР° СЃР±РѕСЂРєРё Tauri-РїСЂРёР»РѕР¶РµРЅРёСЏ")
        .run(|app_handle, event| {
            // РћР‘РЇР—РђРўР•Р›Р¬РќРћ (L5, С€Р°Рі 4): РїСЂРё РІС‹С…РѕРґРµ РѕСЃС‚Р°РЅР°РІР»РёРІР°РµРј sidecar
            // РїСЂРѕС†РµСЃСЃС‹, РёРЅР°С‡Рµ qdrant.exe РѕСЃС‚Р°РЅРµС‚СЃСЏ РІРёСЃРµС‚СЊ РїРѕСЃР»Рµ Р·Р°РєСЂС‹С‚РёСЏ РѕРєРЅР°.
            if let tauri::RunEvent::Exit = event {
                let state: State<Arc<AppState>> = app_handle.state();
                qdrant::stop_qdrant(&state.qdrant);
            }
        });
}
