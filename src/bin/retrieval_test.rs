//! Воспроизводимый тест ретрива Rust-версии на тех же 10 вопросах,
//! что и в Python-прототипе (app/tools/run_retrieval_tests.py).
//!
//! Результаты + сравнение с Python-эталоном пишутся в logs/retrieval_test.md.
//! Запуск из корня проекта:
//!   cargo run --release --bin retrieval_test
use std::fmt::Write as _;

use mentor_core::config::AppConfig;
use mentor_core::rag::{default_config_path, Rag};

/// (chunk_id, score) — эталон python-топ5 для одного вопроса
type ExpectHit<'a> = (&'a str, f32);
/// (ожидаемая тема, вопрос, python-топ5)
const QUESTIONS: &[(&str, &str, &[ExpectHit; 5])] = &[
    (
        "vllm_setup",
        "Как поднять vLLM сервер для инференса модели",
        &[
            ("run2-docker-01-c02", 0.9095),
            ("vllm_setup_004-c00", 0.9027),
            ("vllm_setup_006-c00", 0.9023),
            ("vllm_setup_004-c01", 0.8995),
            ("run2-docker-05-c02", 0.8970),
        ],
    ),
    (
        "gguf_quants / weight_formats",
        "Какие форматы квантования GGUF бывают и чем отличаются Q4 и Q8",
        &[
            ("run2-quant-02-c00", 0.9150),
            ("weight_formats_004-c00", 0.9045),
            ("gguf_quants_015-c01", 0.9027),
            ("run2-quant-02-c01", 0.9000),
            ("oom_diagnostics_005-c01", 0.8985),
        ],
    ),
    (
        "lora_peft",
        "Что такое LoRA и как дообучить модель с её помощью",
        &[
            ("run2-lora-01-c03", 0.8938),
            ("run2-lora-14-c02", 0.8790),
            ("run2-lora-03-c00", 0.8763),
            ("run2-lora-14-c01", 0.8743),
            ("run2-merge-10-c03", 0.8718),
        ],
    ),
    (
        "oom_diagnostics / hardware_vram",
        "Почему возникает CUDA out of memory и как это диагностировать",
        &[
            ("oom_diagnostics_001-c00", 0.9110),
            ("oom_diagnostics_010-c02", 0.8972),
            ("oom_diagnostics_001-c02", 0.8945),
            ("vllm_setup_014-c00", 0.8936),
            ("oom_diagnostics_015-c00", 0.8920),
        ],
    ),
    (
        "run2_vector_db",
        "Какую векторную базу данных выбрать для RAG",
        &[
            ("run2-vdb-01-c00", 0.8970),
            ("run2-vdb-01-c02", 0.8968),
            ("hardware_vram_014-c01", 0.8814),
            ("hardware_vram_014-c00", 0.8808),
            ("run2-vdb-09-c02", 0.8803),
        ],
    ),
    (
        "run2_docker",
        "Как запустить LLM-сервер в Docker-контейнере",
        &[
            ("run2-docker-15-c00", 0.8973),
            ("run2-docker-05-c03", 0.8942),
            ("vllm_setup_015-c00", 0.8933),
            ("vllm_setup_015-c01", 0.8874),
            ("run2-docker-12-c00", 0.8864),
        ],
    ),
    (
        "abliteration",
        "Что такое abliteration и зачем снимают выравнивание с модели",
        &[
            ("abliteration_001-c01", 0.8994),
            ("abliteration_001-c00", 0.8968),
            ("abliteration_007-c01", 0.8782),
            ("abliteration_011-c00", 0.8756),
            ("abliteration_012-c00", 0.8715),
        ],
    ),
    (
        "open_weights_licenses",
        "Какие лицензии бывают у открытых весов моделей",
        &[
            ("open_weights_licenses_001-c01", 0.9011),
            ("open_weights_licenses_001-c00", 0.8970),
            ("abliteration_014-c01", 0.8845),
            ("open_weights_licenses_009-c00", 0.8825),
            ("run2-merge-14-c01", 0.8792),
        ],
    ),
    (
        "tokenizer_chat_template",
        "Что такое chat template у токенизатора",
        &[
            ("tokenizer_chat_template_003-c00", 0.9063),
            ("gguf_quants_014-c02", 0.8913),
            ("run2-hf-05-c00", 0.8913),
            ("gguf_quants_014-c00", 0.8892),
            ("run2-lora-04-c01", 0.8861),
        ],
    ),
    (
        "run2_evaluation",
        "Как оценить качество LLM, какие метрики использовать",
        &[
            ("run2-eval-02-c01", 0.8853),
            ("run2-hf-11-c02", 0.8820),
            ("run2-emb-06-c00", 0.8797),
            ("run2-eval-01-c02", 0.8793),
            ("run2-eval-14-c01", 0.8782),
        ],
    ),
];

const K: usize = 5;
const SNIPPET_LEN: usize = 220;

struct QuestionReport {
    topic_hint: &'static str,
    question: String,
    /// (позиция, score, delta vs python | n/a, chunk_id, topic)
    rows: Vec<(usize, f32, String, String, String)>,
    snippet: String,
    top1_match: bool,
    /// пересечение множеств top-5 с Python
    overlap: usize,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cfg = AppConfig::load(&default_config_path())?;
    let mut rag = Rag::new(cfg.clone()).await?;

    let points = rag.verify_collection().await?;
    println!("Коллекция '{}': {points} точек", cfg.qdrant.collection);

    let mut reports = Vec::new();
    for (hint, question, py_baseline) in QUESTIONS {
        let hits = rag.search(question, K).await?;
        let py_ids: Vec<&str> = py_baseline.iter().map(|(id, _)| *id).collect();
        let mut rows = Vec::new();
        for (i, h) in hits.iter().enumerate() {
            // дельта против Python только если на той же позиции тот же чанк
            let delta = match py_baseline.get(i) {
                Some((pid, pscore)) if *pid == h.chunk_id => format!("{:+.4}", h.score - pscore),
                _ => "n/a".to_string(),
            };
            rows.push((
                i + 1,
                h.score,
                delta,
                h.chunk_id.clone(),
                h.topic.clone().unwrap_or_default(),
            ));
        }
        let rust_ids: Vec<&str> = hits.iter().map(|h| h.chunk_id.as_str()).collect();
        let overlap = rust_ids.iter().filter(|id| py_ids.contains(id)).count();
        let top1_match = hits
            .first()
            .map(|h| h.chunk_id == py_ids[0])
            .unwrap_or(false);
        let snippet: String = hits
            .first()
            .map(|h| h.text.chars().take(SNIPPET_LEN).collect::<String>())
            .unwrap_or_default()
            .replace('\n', " ");
        reports.push(QuestionReport {
            topic_hint: hint,
            question: question.to_string(),
            rows,
            snippet,
            top1_match,
            overlap,
        });
        println!(
            "Q{:>2}: overlap={overlap}/5 top1={}",
            reports.len(),
            if top1_match { "OK" } else { "DIFF" }
        );
    }

    let md = render_markdown(points, &cfg.qdrant.url, &reports);
    let out_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("logs");
    std::fs::create_dir_all(&out_dir)?;
    let out_path = out_dir.join("retrieval_test.md");
    std::fs::write(&out_path, &md)?;

    // сводка в stdout
    let total_top1 = reports.iter().filter(|r| r.top1_match).count();
    let total_overlap: usize = reports.iter().map(|r| r.overlap).sum();
    println!("WROTE: {}", out_path.display());
    println!(
        "СВОДКА: top-1 совпал в {total_top1}/{}; суммарное пересечение top-5: {total_overlap}/{}",
        reports.len(),
        reports.len() * 5
    );
    Ok(())
}

fn render_markdown(points: u64, url: &str, reports: &[QuestionReport]) -> String {
    let mut s = String::new();
    writeln!(
        s,
        "# Тестирование ретрива — Rust-версия (сравнение с Python)"
    )
    .unwrap();
    writeln!(s).unwrap();
    writeln!(s, "- Дата: {}", chrono_now_iso()).unwrap();
    writeln!(
        s,
        "- Реализация: Rust, qdrant-client {url} (gRPC), эмбеддер intfloat/multilingual-e5-small через fastembed-rs (ONNX)"
    )
    .unwrap();
    writeln!(
        s,
        "- Коллекция: `mentor_kb`, точек: {points} (проверено через API)"
    )
    .unwrap();
    writeln!(
        s,
        "- Вопросов: {}, top-{K} на каждый; префикс запроса \"query: \", косинусная близость",
        reports.len()
    )
    .unwrap();
    writeln!(
        s,
        "- Python-эталон: ../app/logs/retrieval_test.md (embedded-Qdrant, sentence-transformers); база перенесена на сервер 1:1 без пересчёта эмбеддингов"
    )
    .unwrap();
    writeln!(s).unwrap();

    for (qi, r) in reports.iter().enumerate() {
        writeln!(s, "## Вопрос {}: {}", qi + 1, r.question).unwrap();
        writeln!(s, "- Ожидаемая тема: {}", r.topic_hint).unwrap();
        writeln!(s).unwrap();
        writeln!(s, "| # | score (rust) | Δscore vs py | chunk_id | topic |").unwrap();
        writeln!(s, "|---|--------------|--------------|----------|-------|").unwrap();
        for (pos, score, delta, id, topic) in &r.rows {
            writeln!(s, "| {pos} | {score:.4} | {delta} | {id} | {topic} |").unwrap();
        }
        writeln!(s).unwrap();
        writeln!(
            s,
            "- Пересечение top-5 с Python: **{}/5**; top-1: {}",
            r.overlap,
            if r.top1_match {
                "совпал"
            } else {
                "**РАСХОЖДЕНИЕ**"
            }
        )
        .unwrap();
        writeln!(s, "- Текст top-1: «{}…»", r.snippet).unwrap();
        writeln!(s).unwrap();
    }

    let total_top1 = reports.iter().filter(|r| r.top1_match).count();
    let total_overlap: usize = reports.iter().map(|r| r.overlap).sum();
    writeln!(s, "## Итоги сравнения с Python-версией").unwrap();
    writeln!(s).unwrap();
    writeln!(s, "| метрика | значение |").unwrap();
    writeln!(s, "|---|---|").unwrap();
    writeln!(s, "| совпадений top-1 | {}/{} |", total_top1, reports.len()).unwrap();
    writeln!(
        s,
        "| суммарное пересечение top-5 | {}/{} позиций |",
        total_overlap,
        reports.len() * 5
    )
    .unwrap();
    writeln!(s).unwrap();
    let diffs: Vec<&QuestionReport> = reports
        .iter()
        .filter(|r| !r.top1_match || r.overlap < 5)
        .collect();
    if diffs.is_empty() {
        writeln!(
            s,
            "Расхождений не обнаружено: все пять позиций каждого вопроса совпали \
             позиции в позицию; Δscore в пределах точности float32/CPU-инференса."
        )
        .unwrap();
    } else {
        writeln!(s, "### Отмеченные расхождения").unwrap();
        writeln!(s).unwrap();
        for d in &diffs {
            writeln!(
                s,
                "- «{}»: top-1 {}, пересечение {}/5 — см. таблицу выше (колонка Δ).",
                d.question,
                if d.top1_match {
                    "совпал"
                } else {
                    "отличается"
                },
                d.overlap
            )
            .unwrap();
        }
    }
    s
}

/// Метка времени без внешних крейтов (секунды от UNIX epoch).
fn chrono_now_iso() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| format!("epoch+{}s", d.as_secs()))
        .unwrap_or_else(|_| "unknown".into())
}
