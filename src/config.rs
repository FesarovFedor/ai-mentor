//! Загрузка и разрешение путей конфигурации приложения (config.toml).
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Deserialize;

/// Префикс запроса, обязательный для моделей семейства e5
/// (совпадает с Python-версией: kb_common.E5_QUERY_PREFIX).
pub const E5_QUERY_PREFIX: &str = "query: ";

#[derive(Debug, Clone, Deserialize)]
pub struct AppConfig {
    /// Путь к GGUF-модели; пуст или файл отсутствует -> экран "модель не найдена".
    #[serde(default)]
    pub model_path: String,
    /// Плейсхолдер URL для скачивания GGUF; реальный адрес подставляет человек позже.
    #[serde(default)]
    pub model_download_url: String,
    /// Опциональная контрольная сумма SHA-256 файла модели (проверяется после скачивания).
    #[serde(default)]
    pub model_sha256: String,
    /// Параметры генерации LLM (этап D).
    #[serde(default)]
    pub generation: GenerationConfig,
    /// Куда класть скачанные файлы моделей.
    #[serde(default)]
    pub download: DownloadConfig,
    pub embedding: EmbeddingConfig,
    pub qdrant: QdrantConfig,
    pub kb_chunks: KbChunksConfig,
    /// Каталог, относительно которого лежал config.toml (заполняется при загрузке).
    #[serde(skip)]
    pub base_dir: PathBuf,
}

/// Параметры инференса с разумными дефолтами (переопределяются в config.toml).
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct GenerationConfig {
    pub temperature: f32,
    pub max_tokens: u32,
    /// Размер окна контекста llama.cpp.
    pub n_ctx: u32,
    /// Формат промпта для модели: "chatml" (instruct-модели Qwen и т.п.) или "raw".
    pub chat_template: String,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        // Синхронизировано с production-конфигом (F-014 аудита): у reasoning-
        // модели <think> до ~3105 токенов (замеры этапа G), поэтому меньший
        // бюджет тихо обрезал бы ответ. Подробности — decisions.md D5/G1.
        Self {
            temperature: 0.7,
            max_tokens: 5500,
            n_ctx: 12288,
            chat_template: String::from("chatml"),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct DownloadConfig {
    pub dir: String,
}

impl Default for DownloadConfig {
    fn default() -> Self {
        Self {
            dir: String::from(".models/downloaded"),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingConfig {
    pub model: String,
    pub cache_dir: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QdrantConfig {
    pub url: String,
    pub collection: String,
    #[serde(default = "default_top_k")]
    pub top_k: u32,
}

fn default_top_k() -> u32 {
    5
}

#[derive(Debug, Clone, Deserialize)]
pub struct KbChunksConfig {
    pub files: Vec<String>,
}

impl AppConfig {
    /// Читает config.toml. Относительные пути внутри файла разрешаются
    /// от каталога самого конфига.
    pub fn load(path: &Path) -> Result<Self> {
        let raw = fs::read_to_string(path)
            .with_context(|| format!("не удалось прочитать конфиг {}", path.display()))?;
        let mut cfg: AppConfig =
            toml::from_str(&raw).with_context(|| format!("ошибка парсинга {}", path.display()))?;
        cfg.base_dir = path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        Ok(cfg)
    }

    /// Относительный путь -> абсолютный (от корня проекта с config.toml).
    pub fn resolve(&self, p: &str) -> PathBuf {
        let path = Path::new(p);
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.base_dir.join(path)
        }
    }

    pub fn embedding_cache_dir(&self) -> PathBuf {
        self.resolve(&self.embedding.cache_dir)
    }

    /// Каталог для скачанных файлов моделей.
    pub fn download_dir(&self) -> PathBuf {
        self.resolve(&self.download.dir)
    }

    /// Абсолютный путь к файлу модели (model_path разрешается как остальные пути).
    pub fn model_file_path(&self) -> PathBuf {
        self.resolve(&self.model_path)
    }

    /// Модель готова к инференсу: путь задан и файл существует.
    pub fn model_ready(&self) -> bool {
        !self.model_path.trim().is_empty() && self.model_file_path().is_file()
    }

    pub fn chunk_files(&self) -> Vec<PathBuf> {
        self.kb_chunks
            .files
            .iter()
            .map(|p| self.resolve(p))
            .collect()
    }
}

/// Атомарная запись файла (F-012 аудита, аналог Q2 audit-v4): сначала во
/// временный файл в том же каталоге, затем rename поверх целевого. Внезапное
/// завершение процесса посреди записи больше не даёт битый config.toml:
/// либо старая версия, либо полностью записанная новая.
fn atomic_write(path: &Path, contents: &str) -> Result<()> {
    let tmp = path.with_extension("toml.tmp");
    fs::write(&tmp, contents).with_context(|| format!("не удалось записать {}", tmp.display()))?;
    // fs::rename на Windows использует MoveFileEx с заменой существующего.
    fs::rename(&tmp, path).with_context(|| format!("не удалось заменить {}", path.display()))?;
    Ok(())
}

/// Записывает строковое поле верхнего уровня config.toml (например model_path),
/// сохраняя форматирование и комментарии остальных полей.
pub fn save_string_field(config_path: &Path, field: &str, value: &str) -> Result<()> {
    let raw = fs::read_to_string(config_path)
        .with_context(|| format!("не удалось прочитать конфиг {}", config_path.display()))?;
    let mut doc = raw
        .parse::<toml_edit::DocumentMut>()
        .with_context(|| format!("ошибка парсинга {}", config_path.display()))?;
    doc[field] = toml_edit::value(value);
    atomic_write(config_path, &doc.to_string())
}

/// Записывает параметры [generation] (temperature/max_tokens) в config.toml,
/// сохраняя форматирование и комментарии остальных полей. Вызывается из окна
/// настроек фронтенда.
pub fn save_generation_fields(config_path: &Path, temperature: f64, max_tokens: u32) -> Result<()> {
    let raw = fs::read_to_string(config_path)
        .with_context(|| format!("не удалось прочитать конфиг {}", config_path.display()))?;
    let mut doc = raw
        .parse::<toml_edit::DocumentMut>()
        .with_context(|| format!("ошибка парсинга {}", config_path.display()))?;
    if !doc.contains_key("generation") {
        doc["generation"] = toml_edit::Item::Table(toml_edit::Table::new());
    }
    // Присваивается только значение: ключи и комментарии вокруг них
    // сохраняются toml_edit как есть. f64 приходит от фронта как есть —
    // без артефактов округления f32.
    doc["generation"]["temperature"] = toml_edit::value(temperature);
    doc["generation"]["max_tokens"] = toml_edit::value(i64::from(max_tokens));
    atomic_write(config_path, &doc.to_string())
}
