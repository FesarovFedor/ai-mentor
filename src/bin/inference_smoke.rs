//! CLI-проверка этапа D: load_model -> RAG-контекст -> реальная генерация.
//! Не является частью приложения. Запуск:
//!   inference_smoke [вопрос]
use std::process::ExitCode;
use std::time::Instant;

use mentor_core::config::AppConfig;
use mentor_core::generator::generate_response;
use mentor_core::inference::load_model;
use mentor_core::rag::{default_config_path, format_context, Rag};

fn default_question() -> String {
    String::from("Какие форматы квантования GGUF бывают и чем отличаются Q4 и Q8?")
}

#[tokio::main]
async fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let question = args.get(1).cloned().unwrap_or_else(default_question);

    let cfg = match AppConfig::load(&default_config_path()) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("config.toml: {e:#}");
            return ExitCode::FAILURE;
        }
    };
    if !cfg.model_ready() {
        eprintln!(
            "model_path не готов: {:?} (файл должен существовать)",
            cfg.model_path
        );
        return ExitCode::FAILURE;
    }
    let path = cfg.model_file_path();
    println!("MODEL: {}", path.display());
    println!(
        "SIZE: {} байт ({:.2} GiB)",
        std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0),
        std::fs::metadata(&path)
            .map(|m| m.len() as f64 / (1024.0 * 1024.0 * 1024.0))
            .unwrap_or(0.0)
    );

    // 1. загрузка модели
    let t0 = Instant::now();
    let backend = match mentor_core::inference::init_backend() {
        Ok(b) => std::sync::Arc::new(b),
        Err(e) => {
            eprintln!("BACKEND FAIL: {e:#}");
            return ExitCode::FAILURE;
        }
    };
    let inf = match load_model(&path.to_string_lossy(), &backend) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("LOAD FAIL: {e:#}");
            return ExitCode::FAILURE;
        }
    };
    println!(
        "LOADED за {:.1}s; n_ctx_train={}",
        t0.elapsed().as_secs_f64(),
        inf.n_ctx_train()
    );
    match inf.embedded_chat_template() {
        Some(t) => {
            let one_line = t.replace('\n', "\\n");
            let cut = one_line.chars().take(240).collect::<String>();
            println!("CHAT_TEMPLATE: {cut}...");
        }
        None => println!("CHAT_TEMPLATE: (в GGUF не указан)"),
    }
    println!(
        "GEN PARAMS: temperature={} max_tokens={} n_ctx={} chat_template={}",
        cfg.generation.temperature,
        cfg.generation.max_tokens,
        cfg.generation.n_ctx,
        cfg.generation.chat_template
    );

    // 2. ретрив
    let mut rag = match Rag::new(cfg.clone()).await {
        Ok(r) => r,
        Err(e) => {
            eprintln!("RAG INIT FAIL: {e:#}");
            return ExitCode::FAILURE;
        }
    };
    let points = match rag.verify_collection().await {
        Ok(p) => p,
        Err(e) => {
            eprintln!("QDRANT FAIL: {e:#}");
            return ExitCode::FAILURE;
        }
    };
    let hits = match rag.search(&question, cfg.qdrant.top_k as usize).await {
        Ok(h) => h,
        Err(e) => {
            eprintln!("SEARCH FAIL: {e:#}");
            return ExitCode::FAILURE;
        }
    };
    println!("QUESTION: {question}");
    println!("COLLECTION: {} точек, top-{}:", points, hits.len());
    for h in &hits {
        println!(
            "  {} · {} · {:.4}",
            h.chunk_id,
            h.topic.as_deref().unwrap_or("-"),
            h.score
        );
    }

    // 3. генерация
    let context = format_context(&hits);
    let t1 = Instant::now();
    let answer = match generate_response(&inf, &question, &context, &[], &cfg.generation) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("GENERATE FAIL: {e:#}");
            return ExitCode::FAILURE;
        }
    };
    let secs = t1.elapsed().as_secs_f64();
    let prompt_head: String = answer.prompt.chars().take(200).collect();
    println!("--- PROMPT HEAD ({}):", cfg.generation.chat_template);
    println!("{}", prompt_head.replace('\n', "\\n"));
    println!("--- ANSWER (за {secs:.1}s):");
    if !answer.thinking.is_empty() {
        println!("[thinking]\n{}", answer.thinking);
        println!("[answer]");
    }
    println!("{}", answer.answer);
    ExitCode::SUCCESS
}
