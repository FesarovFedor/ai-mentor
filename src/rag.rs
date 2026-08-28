//! RAG-модуль Rust-версии: текстовый запрос -> top-k релевантных чанков
//! из готовой коллекции `mentor_kb` локального Qdrant.
//!
//! Отличия от Python-прототипа (app/mentor/rag.py + tools/query_kb.py):
//! - доступ к Qdrant через официальный Rust-клиент (gRPC) к локальному серверу;
//! - эмбеддер тот же (intfloat/multilingual-e5-small), но инференс на ONNX
//!   средствами fastembed-rs; префикс "query: " и нормализация сохранены;
//! - тексты чанков по-прежнему не хранятся в payload и читаются из
//!   ../kb_chunks/*.jsonl (только чтение).
use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::path::Path;

use anyhow::{bail, Context, Result};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use qdrant_client::qdrant::value::Kind;
use qdrant_client::qdrant::{QueryPointsBuilder, ScoredPoint, Value};
use qdrant_client::Qdrant;
use serde::Serialize;

use crate::config::{AppConfig, E5_QUERY_PREFIX};

/// Один результат поиска: метаданные точки + score + исходный текст чанка.
#[derive(Debug, Clone, Serialize)]
pub struct SearchHit {
    pub score: f32,
    pub chunk_id: String,
    pub parent_doc_id: Option<String>,
    pub topic: Option<String>,
    pub chunk_index: Option<i64>,
    pub text: String,
}

/// Готовый к работе RAG-контекст: клиент Qdrant, эмбеддер и индекс текстов.
pub struct Rag {
    cfg: AppConfig,
    client: Qdrant,
    embedder: TextEmbedding,
    /// chunk_id -> исходный текст чанка
    chunks: HashMap<String, String>,
}

impl Rag {
    /// Подключается к Qdrant, инициализирует эмбеддер и загружает тексты чанков.
    ///
    /// Асинхронная: тяжёлые синхронные операции (ONNX-инициализация
    /// fastembed и чтение всех *.jsonl базы знаний, которое на больших БЗ
    /// занимает секунды) выполняются в блокирующем пуле tokio
    /// (spawn_blocking), поэтому вызывающий async-контекст (в приложении —
    /// init_state перед стартом UI) и main-поток Tauri не замирают на
    /// файловом I/O (P1 из аудита).
    pub async fn new(cfg: AppConfig) -> Result<Self> {
        let client = Qdrant::from_url(&cfg.qdrant.url)
            .build()
            .with_context(|| format!("не удалось подключиться к Qdrant ({})", cfg.qdrant.url))?;

        let model = match cfg.embedding.model.as_str() {
            "multilingual-e5-small" | "intfloat/multilingual-e5-small" => {
                EmbeddingModel::MultilingualE5Small
            }
            other => bail!("неизвестная embedding-модель в config.toml: {other}"),
        };
        let cache_dir = cfg.embedding_cache_dir();
        // ONNX-инициализация: диск + вычисления, синхронная и тяжёлая.
        let embedder = tokio::task::spawn_blocking(move || {
            TextEmbedding::try_new(InitOptions::new(model).with_cache_dir(cache_dir))
                .context("не удалось инициализировать fastembed (ONNX)")
        })
        .await
        .context("поток инициализации fastembed прерван")??;

        let cfg_for_chunks = cfg.clone();
        let chunks = tokio::task::spawn_blocking(move || {
            load_chunk_index(&cfg_for_chunks).context("не удалось загрузить индекс чанков")
        })
        .await
        .context("поток загрузки индекса чанков прерван")??;

        Ok(Self {
            cfg,
            client,
            embedder,
            chunks,
        })
    }

    pub fn config(&self) -> &AppConfig {
        &self.cfg
    }

    /// Проверка коллекции через API: существует и непуста. Возвращает число точек.
    pub async fn verify_collection(&self) -> Result<u64> {
        let name = &self.cfg.qdrant.collection;
        if !self
            .client
            .collection_exists(name.clone())
            .await
            .with_context(|| format!("ошибка запроса существования коллекции {name}"))?
        {
            bail!(
                "коллекция '{name}' отсутствует на {}. Запусти tools_migrate/export_points.py",
                self.cfg.qdrant.url
            );
        }
        let info = self
            .client
            .collection_info(name.clone())
            .await
            .with_context(|| format!("collection_info({name}) недоступен"))?;
        let points = info
            .result
            .map(|r| r.points_count.unwrap_or(0))
            .unwrap_or(0);
        if points == 0 {
            bail!("коллекция '{name}' пуста");
        }
        Ok(points)
    }

    /// Вектор запроса: e5-префикс + L2-нормализация внутри fastembed.
    /// Синхронный CPU-инференс; вызывать до await-ов (см. split-API ниже).
    pub fn embed_query(&mut self, query: &str) -> Result<Vec<f32>> {
        let text = format!("{E5_QUERY_PREFIX}{}", query.trim());
        let vectors = self
            .embedder
            .embed(vec![text], None)
            .context("ошибка эмбеддинга запроса")?;
        vectors.into_iter().next().ok_or_else(|| {
            anyhow::anyhow!("эмбеддер вернул пустой результат для запроса: {query:?}")
        })
    }

    /// Собственно поиск по готовому вектору (async, gRPC).
    pub async fn search_by_vector(&self, vector: Vec<f32>, k: usize) -> Result<Vec<SearchHit>> {
        let name = self.cfg.qdrant.collection.clone();
        let response = self
            .client
            .query(
                QueryPointsBuilder::new(name)
                    .query(vector)
                    .limit(k as u64)
                    .with_payload(true),
            )
            .await
            .context("ошибка query_points к Qdrant")?;

        Ok(response
            .result
            .into_iter()
            .map(|point| Self::hit_from_point(point, &self.chunks))
            .collect())
    }

    /// Полный путь "текст -> топ-k" одним вызовом.
    pub async fn search(&mut self, query: &str, k: usize) -> Result<Vec<SearchHit>> {
        let vector = self.embed_query(query)?;
        self.search_by_vector(vector, k).await
    }

    fn hit_from_point(point: ScoredPoint, chunks: &HashMap<String, String>) -> SearchHit {
        let payload = point.payload;
        let get_str = |key: &str| -> Option<String> {
            payload
                .get(key)
                .and_then(|v: &Value| match v.kind.clone()? {
                    Kind::StringValue(s) => Some(s),
                    _ => None,
                })
        };
        let get_int = |key: &str| -> Option<i64> {
            payload
                .get(key)
                .and_then(|v: &Value| match v.kind.clone()? {
                    Kind::IntegerValue(i) => Some(i),
                    _ => None,
                })
        };
        let chunk_id = get_str("chunk_id").unwrap_or_default();
        let text = chunks
            .get(&chunk_id)
            .cloned()
            .unwrap_or_else(|| "<текст не найден в kb_chunks>".to_string());
        SearchHit {
            score: point.score,
            parent_doc_id: get_str("parent_doc_id"),
            topic: get_str("topic"),
            chunk_index: get_int("chunk_index"),
            chunk_id,
            text,
        }
    }
}

/// Единое форматирование RAG-контекста для промпта: "[Фрагмент N | тема: T]
/// текст". Используется приложением и CLI-бинарями (раньше дублировалось
/// в трёх местах — этап K).
pub fn format_context(hits: &[SearchHit]) -> String {
    hits.iter()
        .enumerate()
        .map(|(i, h)| {
            format!(
                "[Фрагмент {} | тема: {}]\n{}",
                i + 1,
                h.topic.as_deref().unwrap_or("-"),
                h.text
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// chunk_id -> текст чанка из jsonl-файлов (формат как в kb_common.read_chunks).
fn load_chunk_index(cfg: &AppConfig) -> Result<HashMap<String, String>> {
    let mut index = HashMap::new();
    for path in cfg.chunk_files() {
        if !path.exists() {
            bail!("файл чанков не найден: {}", path.display());
        }
        let file = std::fs::File::open(&path)
            .with_context(|| format!("не удалось открыть {}", path.display()))?;
        for (line_no, line) in BufReader::new(file).lines().enumerate() {
            let line =
                line.with_context(|| format!("ошибка чтения {}:{}", path.display(), line_no))?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let rec: serde_json::Value = serde_json::from_str(line)
                .with_context(|| format!("битый jsonl: {}:{}", path.display(), line_no))?;
            let id = rec
                .get("chunk_id")
                .and_then(|v| v.as_str())
                .map(str::to_string);
            let text = rec.get("text").and_then(|v| v.as_str()).map(str::to_string);
            if let (Some(id), Some(text)) = (id, text) {
                index.entry(id).or_insert(text); // дубликаты между файлами пропускаем
            }
        }
    }
    if index.is_empty() {
        bail!("ни один файл чанков не дал записей");
    }
    Ok(index)
}

/// Утилита для тестов/CLI: абсолютный путь до конфига проекта по умолчанию.
pub fn default_config_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("config.toml")
}
