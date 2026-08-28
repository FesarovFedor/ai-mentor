//! Сборка промпта и точка инференса. Раньше здесь была ЗАГЛУШКА
//! generate_response(); с этапа D функция вызывает реальную модель через
//! llama-cpp-2 (см. inference.rs), контракт "вопрос + контекст -> ответ"
//! сохранён.
//!
//! Промпт собирается из system + RAG-контекста + вопроса (формат 1:1 с
//! build_rag_answer_inputs Python-версии). Для instruct-моделей текст
//! оборачивается в ChatML (Qwen/Nanbeige и т.п.); вариант "raw" оставляет
//! текст как есть — для base-моделей и нестандартных шаблонов.
use crate::config::GenerationConfig;
use crate::inference::{GenerateParams, Inference, DEFAULT_SEED};
use anyhow::Result;

/// Системный промпт наставника (перенесён из Python-версии без изменений).
pub const SYSTEM_PROMPT: &str =
    "Ты — дружелюбный ИИ-наставник для новичков в локальном LLM-инженерстве. \
Отвечай по-русски, простым языком, без перегруженных терминов. \
Опирайся на приведённые фрагменты базы знаний. Если во фрагментах нет \
ответа — честно скажи об этом и предложи, в какой теме искать.";

/// Разделяет полный промпт на системную часть и тело с контекстом+вопросом.
pub fn build_prompt_parts(question: &str, context: &str) -> (String, String) {
    let user_body = if context.trim().is_empty() {
        format!("## Вопрос пользователя\n{question}")
    } else {
        format!("## Фрагменты базы знаний\n{context}\n\n## Вопрос пользователя\n{question}")
    };
    (SYSTEM_PROMPT.to_string(), user_body)
}

/// Оборачивает промпт в выбранный шаблон модели.
pub fn format_prompt(system: &str, user_body: &str, chat_template: &str) -> String {
    match chat_template {
        // ChatML: Qwen, Nanbeige и другие instruct-модели этого формата.
        "chatml" => format!(
            "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user_body}<|im_end|>\n<|im_start|>assistant\n"
        ),
        // Без шаблона: как есть (base/нестандартные модели).
        _ => format!("{system}\n\n{user_body}"),
    }
}

/// Результат генерации: ОБА блока — ход рассуждений и финальный ответ —
/// плюс финальный промпт, реально ушедший в модель (для прозрачности в UI).
pub struct GeneratedAnswer {
    /// Ход рассуждений из <think>…</think>; пуст у нон-reasoning моделей.
    pub thinking: String,
    /// Финальный ответ; при обрыве по лимиту токенов содержит честную пометку.
    pub answer: String,
    pub prompt: String,
}

/// Разделяет полный вывод модели на (thinking, answer) БЕЗ потери
/// рассуждений: оба блока возвращаются вызывающей стороне (фронтенд
/// показывает их раздельно). Правила:
///   - есть "<think>…</think>" → содержимое тега = thinking, остаток = answer;
///   - "<think>" без закрывающего (обрыв по лимиту внутри рассуждений) →
///     всё содержимое = thinking, answer = "" (пометка об обрыве добавится);
///   - тега нет вовсе (нон-reasoning модель) → thinking = "", весь текст = answer.
fn split_think(text: String, truncated: bool) -> (String, String) {
    let (thinking, mut answer) = match text.find("<think>") {
        Some(open) => {
            let rest = &text[open + "<think>".len()..];
            match rest.find("</think>") {
                Some(close) => (
                    rest[..close].trim().to_string(),
                    rest[close + "</think>".len()..].trim().to_string(),
                ),
                None => (rest.trim().to_string(), String::new()),
            }
        }
        None => (String::new(), text.trim().to_string()),
    };
    if truncated {
        // Пометка обязательна: пользователь должен видеть неполноту ответа.
        let note = "[ответ обрезан лимитом [generation] max_tokens — увеличь его \
                    в config.toml или настройках приложения для полного ответа]";
        if answer.is_empty() {
            answer.push_str(note);
        } else {
            answer = format!("{answer}\n\n{note}");
        }
    }
    (thinking, answer)
}

/// Реальная генерация ответа загруженной моделью (llama-cpp-2).
///
/// question/context -> собранный по шаблону промпт -> generate() ->
/// ответ + промпт. Блокирующая CPU-операция: вызывать из spawn_blocking.
pub fn generate_response(
    model: &Inference,
    question: &str,
    context: &str,
    generation: &GenerationConfig,
) -> Result<GeneratedAnswer> {
    generate_response_streaming(model, question, context, generation, |_| {})
}

/// Куда относится кусок потокового вывода.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamKind {
    Thinking,
    Answer,
}

/// Кусок потокового вывода, уже разложенный по блокам.
#[derive(Debug, Clone)]
pub struct StreamPiece {
    pub kind: StreamKind,
    pub text: String,
}

/// Инкрементальный роутер потока токенов по границам "<think>"/"</think>".
///
/// Тег может быть разрезан между токенами ("<th" + "ink>"), поэтому хвост
/// буфера длиной "максимальный тег - 1" держится неразобранным до следующего
/// куска; границы срезов выравниваются по границам символов UTF-8.
/// Финальная каноническая нарезка всё равно выполняется split_think по
/// полному тексту — роутер влияет только на живой стрим во фронтенд.
#[derive(Default)]
pub struct ThinkRouter {
    buffer: String,
    in_think: bool,
}

impl ThinkRouter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Добавляет кусок текста и выдаёт готовые (уже не разрезанные тегом)
    /// фрагменты.
    pub fn feed(&mut self, piece: &str, out: &mut Vec<StreamPiece>) {
        self.buffer.push_str(piece);
        loop {
            if self.in_think {
                match self.buffer.find("</think>") {
                    Some(i) => {
                        self.push(StreamKind::Thinking, i, out);
                        self.buffer.drain(.."</think>".len());
                        self.in_think = false;
                    }
                    None => {
                        // до 7 последних байт могут быть началом "</think>"
                        self.push_tail(StreamKind::Thinking, 7, out);
                        break;
                    }
                }
            } else {
                match self.buffer.find("<think>") {
                    Some(i) => {
                        self.push(StreamKind::Answer, i, out);
                        self.buffer.drain(.."<think>".len());
                        self.in_think = true;
                    }
                    None => {
                        // до 6 последних байт могут быть началом "<think>"
                        self.push_tail(StreamKind::Answer, 6, out);
                        break;
                    }
                }
            }
        }
    }

    /// Конец генерации: отдаёт остаток буфера в текущий блок.
    pub fn finish(&mut self, out: &mut Vec<StreamPiece>) {
        if !self.buffer.is_empty() {
            let kind = if self.in_think {
                StreamKind::Thinking
            } else {
                StreamKind::Answer
            };
            let text = std::mem::take(&mut self.buffer);
            out.push(StreamPiece { kind, text });
        }
    }

    fn push(&mut self, kind: StreamKind, up_to: usize, out: &mut Vec<StreamPiece>) {
        if up_to > 0 {
            let text = self.buffer.drain(..up_to).collect::<String>();
            out.push(StreamPiece { kind, text });
        }
    }

    /// Отдаёт префикс буфера минус `hold` байт (хвост может оказаться началом
    /// тега), выравнивая срез по границе символа UTF-8.
    fn push_tail(&mut self, kind: StreamKind, hold: usize, out: &mut Vec<StreamPiece>) {
        let len = self.buffer.len();
        if len > hold {
            let mut cut = len - hold;
            while !self.buffer.is_char_boundary(cut) {
                cut += 1;
            }
            self.push(kind, cut, out);
        }
    }
}

/// Потоковая версия generate_response: каждый токен после инкрементальной
/// разметки (thinking/answer) отдаётся в on_piece ДО завершения генерации.
/// Итоговый GeneratedAnswer тот же, что вернул бы generate_response
/// (канонический разрез делает split_think по полному тексту).
pub fn generate_response_streaming(
    model: &Inference,
    question: &str,
    context: &str,
    generation: &GenerationConfig,
    mut on_piece: impl FnMut(StreamPiece),
) -> Result<GeneratedAnswer> {
    let (system, user_body) = build_prompt_parts(question, context);
    let prompt = format_prompt(&system, &user_body, &generation.chat_template);
    let mut router = ThinkRouter::new();
    let out = model.generate_with_callback(
        &prompt,
        GenerateParams {
            temperature: generation.temperature,
            max_tokens: generation.max_tokens,
            n_ctx: generation.n_ctx,
            seed: DEFAULT_SEED,
        },
        |piece| {
            let mut pieces = Vec::new();
            router.feed(piece, &mut pieces);
            for p in pieces {
                on_piece(p);
            }
        },
    )?;
    let mut tail = Vec::new();
    router.finish(&mut tail);
    for p in tail {
        on_piece(p);
    }
    let (thinking, answer) = split_think(out.text, out.truncated);
    Ok(GeneratedAnswer {
        thinking,
        answer,
        prompt,
    })
}
