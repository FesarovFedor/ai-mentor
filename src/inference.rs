//! Инференс GGUF через llama-cpp-2 (нативный llama.cpp).
//!
//! Архитектура владения бэкендом: llama_backend_init/free не реентерабельны
//! (см. llama-cpp-2), поэтому LlamaBackend создаётся РОВНО один раз — в
//! AppState Tauri-приложения (или в CLI-бинарях) и раздаётся владельцем
//! Arc<LlamaBackend>. Глобальный static OnceLock сознательно не используется:
//! статические переменные дропаются слишком поздно/недетерминированно, из-за
//! чего CUDA-контекст не освобождается при закрытии приложения (утечка VRAM,
//! P0 из аудита). AppState гарантирует вызов Drop и освобождение VRAM при
//! штатном и аварийном завершении. LlamaModel (Send+Sync) хранится в
//! Inference; контекст генерации создаётся на каждый запрос — вызывать
//! generate из spawn_blocking/отдельного потока.
use std::num::NonZeroU32;
use std::path::Path;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use llama_cpp_2::context::params::{KvCacheType, LlamaContextParams};
/// Реэкспорт: AppState Tauri-оболочки держит Arc<LlamaBackend>, не завися
/// напрямую от llama-cpp-2.
pub use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;

/// Детерминированный сид по умолчанию: воспроизводимость ответов важнее
/// случайности для тестирования и логов.
pub const DEFAULT_SEED: u32 = 1_234;

/// Единственная точка инициализации llama.cpp на процесс. Вызывается
/// владельцем (AppState / CLI-main) и дальше раздаётся как Arc<LlamaBackend>.
pub fn init_backend() -> Result<LlamaBackend> {
    LlamaBackend::init().context("не удалось инициализировать llama.cpp backend")
}

/// Параметры генерации одного запроса.
#[derive(Debug, Clone, Copy)]
pub struct GenerateParams {
    pub temperature: f32,
    pub max_tokens: u32,
    /// Окно контекста (промпт + генерация); обрезается до n_ctx_train.
    pub n_ctx: u32,
    pub seed: u32,
}

/// Загруженная GGUF-модель. Держит Arc на общий LlamaBackend: пока жива
/// хотя бы одна Inference, бэкенд не дропнется.
pub struct Inference {
    model: LlamaModel,
    /// Поле не читается напрямую — держит Arc живым на время жизни модели
    /// (см. комментарий к struct).
    #[allow(dead_code)]
    backend: Arc<LlamaBackend>,
}

/// Требуемая задача точка входа: загрузить GGUF по строковому пути.
pub fn load_model(path: &str, backend: &Arc<LlamaBackend>) -> Result<Inference> {
    Inference::load(Path::new(path), backend)
}

/// Отображаемое имя модели из GGUF-метаданных (general.name), без загрузки
/// весов — читается только заголовок файла. None, если метаданных нет.
pub fn gguf_display_name(path: &Path) -> Option<String> {
    let ctx = llama_cpp_2::gguf::GgufContext::from_file(path)?;
    let idx = ctx.find_key("general.name");
    if idx < 0 {
        return None;
    }
    ctx.val_str(idx).map(String::from)
}

/// Результат полной генерации: текст плюс фактические счётчики токенов.
pub struct GenerationOutput {
    /// Сгенерированный текст (обрезан по краям от пробелов).
    pub text: String,
    /// true — лимит max_tokens исчерпан до EOS модели.
    pub truncated: bool,
    /// true — генерация прервана по требованию пользователя (F-002):
    /// колбэк токена вернул false. Частичный текст сохранён в text.
    pub stopped_by_user: bool,
    /// Фактическая длина промпта в токенах (после токенизации).
    pub n_prompt_tokens: usize,
    /// Число реально сгенерированных токенов (без EOG).
    pub n_gen_tokens: usize,
    /// Байтовое смещение начала каждого сгенерированного токена в text
    /// (для точного разреза thinking/answer по позиции "</think>").
    /// Всегда len() == n_gen_tokens и offsets[i] — начало i-го куска.
    pub token_offsets: Vec<usize>,
}

impl Inference {
    /// Реально загружает GGUF-файл средствами llama.cpp (валидация формата,
    /// маппинг весов в память). Ошибки: несуществующий путь, не-GGUF формат,
    /// нехватка памяти.
    ///
    /// Все слои offload-ятся на GPU (`n_gpu_layers = i32::MAX`): 3B-модель
    /// (~2.65 GiB весов) с запасом помещается в 8 GB VRAM. Без собранного
    /// GPU-бэкенда llama.cpp игнорирует оффлоад и считает на CPU.
    ///
    /// Примечание: «все слои» в llama.cpp документально кодируется как -1,
    /// но обёртка принимает только u32 (отрицательное недостижимо), поэтому
    /// используется u32::MAX -> i32::MAX: значение больше числа слоёв трактуется
    /// как min(n_gpu_layers, n_layer+1) — тот же полный оффлоад (проверено:
    /// "offloaded 45/45 layers to GPU" в логах этапа E).
    pub fn load(path: &Path, backend: &Arc<LlamaBackend>) -> Result<Self> {
        if !path.is_file() {
            bail!("файл модели не найден: {}", path.display());
        }
        let params = LlamaModelParams::default().with_n_gpu_layers(u32::MAX);
        let model = LlamaModel::load_from_file(backend, path, &params).with_context(|| {
            format!(
                "не удалось загрузить модель {}: битый/неподдерживаемый GGUF или нехватка памяти",
                path.display()
            )
        })?;
        Ok(Self {
            model,
            backend: backend.clone(),
        })
    }

    /// Размер окна, на котором модель обучена (для клампа n_ctx из конфига).
    pub fn n_ctx_train(&self) -> u32 {
        self.model.n_ctx_train()
    }

    /// Встроенный в GGUF chat-шаблон (если автор модели его указал).
    pub fn embedded_chat_template(&self) -> Option<String> {
        self.model.meta_val_str("tokenizer.chat_template").ok()
    }

    /// Полная генерация по готовому промпту. Блокирующая CPU/GPU-операция:
    /// вызывать вне async-потока UI. Возвращает GenerationOutput с текстом,
    /// флагом обрыва по лимиту и фактическими счётчиками токенов.
    pub fn generate(&self, prompt: &str, params: GenerateParams) -> Result<GenerationOutput> {
        self.generate_with_callback(prompt, params, |_| true)
    }

    /// То же, что generate(), но каждый сгенерированный токен (уже
    /// детокенизированный кусок UTF-8) отдаётся в on_token ДО декодирования
    /// следующего — основа потокового вывода во фронтенд (этап J).
    /// F-002: колбэк возвращает false для прерывания генерации («стоп» в UI);
    /// частичный текст сохраняется, stopped_by_user = true.
    pub fn generate_with_callback(
        &self,
        prompt: &str,
        params: GenerateParams,
        mut on_token: impl FnMut(&str) -> bool,
    ) -> Result<GenerationOutput> {
        let n_ctx_train = self.model.n_ctx_train();
        let n_ctx = params.n_ctx.clamp(512, n_ctx_train.max(512));
        // ChatML-шаблоны начинаются с <|im_start|>, который у многих моделей
        // и есть BOS: не добавляем второй. Для прочих промптов BOS ставим.
        // special=true — чтобы BOS-токен отдался своим литеральным текстом.
        let add_bos = {
            let mut bos_decoder = encoding_rs::UTF_8.new_decoder();
            match self
                .model
                .token_to_piece(self.model.token_bos(), &mut bos_decoder, true, None)
            {
                Ok(bos) if !bos.is_empty() && prompt.starts_with(bos.as_str()) => AddBos::Never,
                _ => AddBos::Always,
            }
        };
        let prompt_tokens = self
            .model
            .str_to_token(prompt, add_bos)
            .map_err(|e| anyhow::anyhow!("ошибка токенизации промпта: {e}"))?;
        if prompt_tokens.is_empty() {
            bail!("пустой промпт после токенизации");
        }
        if prompt_tokens.len() + 1 >= n_ctx as usize {
            bail!(
                "промпт ({}) токенов не помещается в окно {} вместе с ответом; \
                 уменьши top_k или увеличь [generation] n_ctx",
                prompt_tokens.len(),
                n_ctx
            );
        }
        // Генерация не должна вылезти за окно поверх промпта: клампим бюджет
        // до свободного места (иначе переполняется KV-кэш).
        let headroom = n_ctx as usize - prompt_tokens.len() - 1;
        let max_new = params.max_tokens.min(headroom as u32);
        if max_new < 32 {
            bail!(
                "в окне {} осталось только {} токенов на ответ после промпта из {}; \
                 увеличь [generation] n_ctx",
                n_ctx,
                headroom,
                prompt_tokens.len()
            );
        }

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx))
            // Батч обязан вмещать весь промпт целиком (иначе GGML_ASSERT
            // внутри llama.cpp): RAG-контекст легко превышает дефолтные 2048.
            .with_n_batch(prompt_tokens.len().min(n_ctx as usize) as u32)
            .with_n_threads(0)
            .with_n_threads_batch(0)
            // Flash Attention обязателен для квантованного V-кэша (иначе
            // llama.cpp игнорирует type_v). FA включён явно, не AUTO.
            .with_flash_attention_policy(llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_ENABLED)
            // 4-битное квантование KV-кэша: экономия ~4x на кэше против F16
            // (1408 -> ~352 MiB при n_ctx 8192), см. этап F build_log.
            .with_type_k(KvCacheType::Q4_0)
            .with_type_v(KvCacheType::Q4_0);
        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .context("не удалось создать контекст инференса (нехватка памяти?)")?;

        let mut batch = LlamaBatch::new(prompt_tokens.len(), 1);
        for (pos, token) in prompt_tokens.iter().enumerate() {
            batch
                .add(*token, pos as i32, &[0], pos + 1 == prompt_tokens.len())
                .context("ошибка наполнения батча промпта")?;
        }
        ctx.decode(&mut batch)
            .map_err(|e| anyhow::anyhow!("ошибка декодирования промпта (OOM?): {e}"))?;

        // Сэмплер: temperature>0 -> temp+dist(seed), иначе жадный выбор.
        let mut sampler = if params.temperature > 0.01 {
            LlamaSampler::chain_simple([
                LlamaSampler::temp(params.temperature),
                LlamaSampler::dist(params.seed),
            ])
        } else {
            LlamaSampler::chain_simple([LlamaSampler::greedy()])
        };

        // Один декодер на всю генерацию: кириллица часто разрезается между
        // токенами, per-token декодер давал бы U+FFFD.
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut out = String::new();
        let mut token_offsets: Vec<usize> = Vec::new();
        let mut truncated = true;
        let mut stopped_by_user = false;
        let first_gen_pos = prompt_tokens.len() as i32;
        for pos in first_gen_pos..first_gen_pos + max_new as i32 {
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            if self.model.is_eog_token(token) {
                truncated = false;
                break;
            }
            token_offsets.push(out.len());
            let piece = self
                .model
                .token_to_piece(token, &mut decoder, true, None)
                .map_err(|e| anyhow::anyhow!("ошибка детокенизации: {e}"))?;
            out.push_str(&piece);
            if !on_token(&piece) {
                // «Стоп» от пользователя (F-002): цикл прерывается, частичный
                // текст остаётся в out — вызывающая сторона решает, что с ним.
                stopped_by_user = true;
                break;
            }

            batch.clear();
            batch
                .add(token, pos, &[0], true)
                .context("ошибка расширения батча при генерации")?;
            ctx.decode(&mut batch)
                .map_err(|e| anyhow::anyhow!("ошибка декодирования шага генерации: {e}"))?;
        }

        let n_gen_tokens = token_offsets.len();
        // Текст возвращается КАК СГЕНЕРИРОВАН (без обрезки краёв), поэтому
        // token_offsets всегда 1:1 с text: offsets[i] — байтовое начало
        // i-го токена. Обрезку краёв делают потребители при необходимости.
        Ok(GenerationOutput {
            text: out,
            truncated,
            stopped_by_user,
            n_prompt_tokens: prompt_tokens.len(),
            n_gen_tokens,
            token_offsets,
        })
    }
}
