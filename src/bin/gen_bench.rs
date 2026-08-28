//! CLI-бенчмарк генерации (этапы E/F/G): скорость токенов/с, счётчики
//! prompt/thinking/answer-токенов. Не является частью приложения.
//!
//! Запуск:
//!   gen_bench [--max-tokens N] [--file вопросы.txt] [--label ИМЯ] [вопрос]
//!
//! По одному вопросу на строку файла; строки `#` пропускаются.

use std::process::ExitCode;
use std::time::Instant;

use mentor_core::config::AppConfig;
use mentor_core::generator::{build_prompt_parts, format_prompt};
use mentor_core::inference::{GenerateParams, Inference, DEFAULT_SEED};
use mentor_core::rag::{default_config_path, format_context, Rag};
use std::sync::Arc;

struct Args {
    max_tokens: Option<u32>,
    file: Option<String>,
    label: String,
    question: Option<String>,
}

fn parse_args() -> Args {
    let mut a = Args {
        max_tokens: None,
        file: None,
        label: String::from("bench"),
        question: None,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut it = argv.into_iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--max-tokens" => {
                a.max_tokens = it.next().and_then(|v| v.parse().ok());
            }
            "--file" => a.file = it.next(),
            "--label" => a.label = it.next().unwrap_or_else(|| "bench".into()),
            s => a.question = Some(String::from(s)),
        }
    }
    a
}

#[tokio::main]
async fn main() -> ExitCode {
    let args = parse_args();
    let cfg = match AppConfig::load(&default_config_path()) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("config.toml: {e:#}");
            return ExitCode::FAILURE;
        }
    };
    if !cfg.model_ready() {
        eprintln!("model_path не готов: {:?}", cfg.model_file_path());
        return ExitCode::FAILURE;
    }

    // Список вопросов: из --file либо единственный позиционный аргумент.
    let mut questions: Vec<String> = Vec::new();
    if let Some(path) = &args.file {
        let raw = std::fs::read_to_string(path).unwrap_or_default();
        for line in raw.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            questions.push(line.to_string());
        }
    } else if let Some(q) = args.question {
        questions.push(q);
    }
    if questions.is_empty() {
        eprintln!("нет вопросов: передай вопрос или --file");
        return ExitCode::FAILURE;
    }

    println!(
        "BENCH {} | model={} | temperature={} max_tokens={} n_ctx={}",
        args.label,
        cfg.model_file_path().display(),
        cfg.generation.temperature,
        cfg.generation.max_tokens,
        cfg.generation.n_ctx
    );

    let t0 = Instant::now();
    let backend = match mentor_core::inference::init_backend() {
        Ok(b) => Arc::new(b),
        Err(e) => {
            eprintln!("BACKEND FAIL: {e:#}");
            return ExitCode::FAILURE;
        }
    };
    let inf = match Inference::load(&cfg.model_file_path(), &backend) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("LOAD FAIL: {e:#}");
            return ExitCode::FAILURE;
        }
    };
    println!("LOADED {:.1}s", t0.elapsed().as_secs_f64());

    let mut rag = match Rag::new(cfg.clone()).await {
        Ok(r) => r,
        Err(e) => {
            eprintln!("RAG INIT FAIL: {e:#}");
            return ExitCode::FAILURE;
        }
    };

    for (i, question) in questions.iter().enumerate() {
        let hits = match rag.search(question, cfg.qdrant.top_k as usize).await {
            Ok(h) => h,
            Err(e) => {
                eprintln!("SEARCH FAIL: {e:#}");
                return ExitCode::FAILURE;
            }
        };
        let context = format_context(&hits);

        let (system, user_body) = build_prompt_parts(question, &context);
        let prompt = format_prompt(&system, &user_body, &cfg.generation.chat_template);
        let params = GenerateParams {
            temperature: cfg.generation.temperature,
            max_tokens: args.max_tokens.unwrap_or(cfg.generation.max_tokens),
            n_ctx: cfg.generation.n_ctx,
            seed: DEFAULT_SEED,
        };

        let t1 = Instant::now();
        let out = match inf.generate(&prompt, params) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("GENERATE FAIL: {e:#}");
                return ExitCode::FAILURE;
            }
        };
        let secs = t1.elapsed().as_secs_f64();
        let tps = out.n_gen_tokens as f64 / secs;

        // Точный разрез thinking/answer по позиции закрывающего тега:
        // токены, начавшиеся до конца "</think>", относятся к thinking.
        let think_end = out.text.find("</think>").map(|p| p + "</think>".len());
        let (think_tok, ans_tok, closed) = match think_end {
            Some(mark) => (
                out.token_offsets.iter().filter(|&&o| o < mark).count(),
                out.token_offsets.iter().filter(|&&o| o >= mark).count(),
                true,
            ),
            None => (out.n_gen_tokens, 0, false),
        };

        println!(
            "Q#{} | prompt_tok={} gen_tok={} think_tok={} answer_tok={} time={:.1}s tok_s={:.1} truncated={} think_closed={}",
            i + 1,
            out.n_prompt_tokens,
            out.n_gen_tokens,
            think_tok,
            ans_tok,
            secs,
            tps,
            out.truncated,
            closed
        );
        println!("QUESTION: {question}");
        let raw_full = out.text.trim().replace('\n', "\\n");
        println!("RAW_FULL: {raw_full}");
    }
    ExitCode::SUCCESS
}
