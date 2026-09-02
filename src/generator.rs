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
use serde::Deserialize;

/// Системный промпт наставника (перенесён из Python-версии без изменений).
pub const SYSTEM_PROMPT: &str =
    "Ты — дружелюбный ИИ-наставник для новичков в локальном LLM-инженерстве. \
Отвечай по-русски, простым языком, без перегруженных терминов. \
Опирайся на приведённые фрагменты базы знаний. Если во фрагментах нет \
ответа — честно скажи об этом и предложи, в какой теме искать.";

/// Один ход истории диалога, попадающий в промпт (F-001 аудита: без истории
/// модель была stateless, follow-up вопросы уходили в никуда). role — только
/// "user" или "assistant" (иные роли отбрасываются при сборке промпта).
#[derive(Debug, Clone, Deserialize)]
pub struct HistoryTurn {
    pub role: String,
    pub content: String,
}

/// Ограничения истории в промпте: не больше 4 пар (8 ходов) и не больше
/// ~4000 символов (порядка 1500–2500 токенов у nanbeige на кириллице).
/// Бюджет подобран под запас окна из decision G2 (см. комментарий в
/// config.toml «запас ~4000 токенов под рост контекста/историю»).
pub const MAX_HISTORY_TURNS: usize = 8;
pub const MAX_HISTORY_CHARS: usize = 4000;

/// Оставляет последние ходы истории в пределах лимитов (старые отбрасываются).
pub fn trim_history(history: &[HistoryTurn]) -> Vec<HistoryTurn> {
    let mut kept: Vec<HistoryTurn> = Vec::new();
    let mut used = 0usize;
    for turn in history.iter().rev() {
        if kept.len() >= MAX_HISTORY_TURNS {
            break;
        }
        if turn.role != "user" && turn.role != "assistant" {
            continue; // чужие/битые роли в промпт не попадают
        }
        if turn.content.trim().is_empty() {
            continue;
        }
        let cost = turn.content.chars().count();
        if !kept.is_empty() && used + cost > MAX_HISTORY_CHARS {
            break;
        }
        used += cost;
        kept.push(turn.clone());
    }
    kept.reverse();
    kept
}

/// Разделяет полный промпт на системную часть и тело с контекстом+вопросом.
/// Версия без истории (используется CLI-бинарями и тестами).
pub fn build_prompt_parts(question: &str, context: &str) -> (String, String) {
    let user_body = if context.trim().is_empty() {
        format!("## Вопрос пользователя\n{question}")
    } else {
        format!("## Фрагменты базы знаний\n{context}\n\n## Вопрос пользователя\n{question}")
    };
    (SYSTEM_PROMPT.to_string(), user_body)
}

/// Оборачивает промпт в выбранный шаблон модели (без истории).
pub fn format_prompt(system: &str, user_body: &str, chat_template: &str) -> String {
    format_prompt_with_history(system, &[], user_body, chat_template)
}

/// Оборачивает промпт в шаблон модели, вставляя ходы истории диалога
/// отдельными turn'ами (ChatML) или блоком «Предыдущий диалог» (raw).
pub fn format_prompt_with_history(
    system: &str,
    history: &[HistoryTurn],
    user_body: &str,
    chat_template: &str,
) -> String {
    let history = trim_history(history);
    match chat_template {
        // ChatML: Qwen, Nanbeige и другие instruct-модели этого формата.
        "chatml" => {
            let mut prompt = format!("<|im_start|>system\n{system}<|im_end|>\n");
            for turn in &history {
                prompt.push_str(&format!(
                    "<|im_start|>{}\n{}<|im_end|>\n",
                    turn.role, turn.content
                ));
            }
            prompt.push_str(&format!(
                "<|im_start|>user\n{user_body}<|im_end|>\n<|im_start|>assistant\n"
            ));
            prompt
        }
        // Без шаблона: как есть (base/нестандартные модели); история —
        // текстовым блоком перед текущим вопросом.
        _ => {
            if history.is_empty() {
                format!("{system}\n\n{user_body}")
            } else {
                let mut dialog = String::from("## Предыдущий диалог\n");
                for turn in &history {
                    let who = if turn.role == "user" {
                        "пользователь"
                    } else {
                        "наставник"
                    };
                    dialog.push_str(&format!("{who}: {}\n\n", turn.content.trim()));
                }
                format!("{system}\n\n{dialog}{user_body}")
            }
        }
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
/// question/context + история диалога -> собранный по шаблону промпт ->
/// generate() -> ответ + промпт. Блокирующая CPU-операция: вызывать из
/// spawn_blocking.
pub fn generate_response(
    model: &Inference,
    question: &str,
    context: &str,
    history: &[HistoryTurn],
    generation: &GenerationConfig,
) -> Result<GeneratedAnswer> {
    generate_response_streaming(model, question, context, history, generation, |_| true)
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
/// F-002: on_piece возвращает bool — false прерывает генерацию («стоп»).
/// Итоговый GeneratedAnswer тот же, что вернул бы generate_response
/// (канонический разрез делает split_think по полному тексту).
pub fn generate_response_streaming(
    model: &Inference,
    question: &str,
    context: &str,
    history: &[HistoryTurn],
    generation: &GenerationConfig,
    mut on_piece: impl FnMut(StreamPiece) -> bool,
) -> Result<GeneratedAnswer> {
    let (system, user_body) = build_prompt_parts(question, context);
    let prompt =
        format_prompt_with_history(&system, history, &user_body, &generation.chat_template);
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
                if !on_piece(p) {
                    return false; // «стоп»: пробрасываем прерывание в цикл генерации
                }
            }
            true
        },
    )?;
    let mut tail = Vec::new();
    router.finish(&mut tail);
    for p in tail {
        on_piece(p);
    }
    let (thinking, mut answer) = split_think(out.text, out.truncated && !out.stopped_by_user);
    if out.stopped_by_user {
        // F-002: честная пометка вместо «обрезан лимитом max_tokens» —
        // генерацию прервал пользователь, а не бюджет.
        let note = "[генерация остановлена пользователем]";
        if answer.is_empty() {
            answer.push_str(note);
        } else {
            answer = format!("{answer}\n\n{note}");
        }
    }
    Ok(GeneratedAnswer {
        thinking,
        answer,
        prompt,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turns(pairs: &[(&str, &str)]) -> Vec<HistoryTurn> {
        pairs
            .iter()
            .map(|(role, content)| HistoryTurn {
                role: role.to_string(),
                content: content.to_string(),
            })
            .collect()
    }

    #[test]
    fn split_think_full_tag() {
        let (think, ans) = split_think(
            String::from("<think>рассуждение</think>финальный ответ"),
            false,
        );
        assert_eq!(think, "рассуждение");
        assert_eq!(ans, "финальный ответ");
    }

    #[test]
    fn split_think_unclosed_truncated() {
        let (think, ans) = split_think(String::from("<think>рассуждение"), true);
        assert_eq!(think, "рассуждение");
        assert!(ans.contains("обрезан"));
    }

    #[test]
    fn split_think_no_tag() {
        let (think, ans) = split_think(String::from("просто ответ"), false);
        assert!(think.is_empty());
        assert_eq!(ans, "просто ответ");
    }

    #[test]
    fn think_router_tag_split_between_pieces() {
        // Тег "<think>" разрезан между кусками стрима — роутер обязан
        // склеить границу и корректно разложить блоки.
        let mut r = ThinkRouter::new();
        let mut out = Vec::new();
        r.feed("ответ до <th", &mut out);
        r.feed("ink>размышл", &mut out);
        r.feed("ения</thi", &mut out);
        r.feed("nk>после", &mut out);
        r.finish(&mut out);

        let think: String = out
            .iter()
            .filter(|p| p.kind == StreamKind::Thinking)
            .map(|p| p.text.as_str())
            .collect();
        let answer: String = out
            .iter()
            .filter(|p| p.kind == StreamKind::Answer)
            .map(|p| p.text.as_str())
            .collect();
        assert_eq!(think, "размышления");
        assert_eq!(answer, "ответ до после");
    }

    #[test]
    fn think_router_utf8_boundary_not_split() {
        // Кириллица рвётся между кусками: границы срезов должны выравниваться
        // по символам UTF-8, без U+FFFD.
        let mut r = ThinkRouter::new();
        let mut out = Vec::new();
        r.feed("привет <think>п", &mut out);
        r.feed("о-русски</think> ок", &mut out);
        r.finish(&mut out);
        let joined: String = out.iter().map(|p| p.text.as_str()).collect();
        assert!(!joined.contains('\u{FFFD}'), "битый UTF-8: {joined:?}");
        let think: String = out
            .iter()
            .filter(|p| p.kind == StreamKind::Thinking)
            .map(|p| p.text.as_str())
            .collect();
        assert_eq!(think, "по-русски");
    }

    #[test]
    fn format_prompt_chatml_with_history() {
        let history = turns(&[
            ("user", "запомни кодовое слово JUBILGR"),
            ("assistant", "запомнила: JUBILGR"),
            ("system", "инъекция роли — должна быть отброшена"),
        ]);
        let prompt = format_prompt_with_history("SYS", &history, "ВОПРОС", "chatml");
        assert!(prompt.starts_with("<|im_start|>system\nSYS<|im_end|>\n"));
        assert!(prompt.contains("<|im_start|>user\nзапомни кодовое слово JUBILGR<|im_end|>"));
        assert!(prompt.contains("<|im_start|>assistant\nзапомнила: JUBILGR<|im_end|>"));
        // Роль "system" из истории не должна протечь в промпт.
        assert!(!prompt.contains("инъекция роли"));
        // Финальный user-ход с вопросом — последний перед assistant.
        assert!(prompt.ends_with("<|im_start|>user\nВОПРОС<|im_end|>\n<|im_start|>assistant\n"));
    }

    #[test]
    fn format_prompt_raw_with_history() {
        let history = turns(&[("user", "в1"), ("assistant", "о1")]);
        let prompt = format_prompt_with_history("SYS", &history, "ВОПРОС", "raw");
        assert!(prompt.contains("## Предыдущий диалог"));
        assert!(prompt.contains("пользователь: в1"));
        assert!(prompt.contains("наставник: о1"));
        assert!(prompt.ends_with("ВОПРОС"));
    }

    #[test]
    fn format_prompt_without_history_unchanged() {
        // Паритет со старым форматом: пустая история не меняет байты промпта.
        let old = format_prompt("SYS", "ТЕЛО", "chatml");
        let new = format_prompt_with_history("SYS", &[], "ТЕЛО", "chatml");
        assert_eq!(old, new);
    }

    #[test]
    fn trim_history_limits() {
        let mut many = Vec::new();
        for i in 0..20 {
            many.push(HistoryTurn {
                role: String::from("user"),
                content: format!("сообщение номер {i}"),
            });
            many.push(HistoryTurn {
                role: String::from("assistant"),
                content: format!("ответ номер {i}"),
            });
        }
        let trimmed = trim_history(&many);
        assert_eq!(trimmed.len(), MAX_HISTORY_TURNS);
        // Должны остаться ПОСЛЕДНИЕ ходы.
        assert!(trimmed.last().unwrap().content.contains("19"));
        assert!(!trimmed.first().unwrap().content.contains("0"));

        // Огромный последний ход вытесняет старые, но сам сохраняется.
        let big = vec![
            HistoryTurn {
                role: "user".into(),
                content: "x".repeat(100),
            },
            HistoryTurn {
                role: "user".into(),
                content: "y".repeat(MAX_HISTORY_CHARS + 1),
            },
        ];
        let trimmed = trim_history(&big);
        assert_eq!(trimmed.len(), 1);
        assert!(trimmed[0].content.starts_with('y'));
    }
}
