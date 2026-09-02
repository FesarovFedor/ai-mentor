# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.1] — 2026-09-02

### Security
- Qdrant sidecar now binds to `127.0.0.1` instead of `0.0.0.0` — the
  knowledge base was previously reachable from the local network.
- Fixed path traversal in model download filenames: percent-encoded
  separators (`%5c`, `%2f`) can no longer escape the download directory
  (defense in depth: separator sanitizing + canonicalize check).
- Hardened CSP: `object-src 'none'; base-uri 'none'; frame-ancestors 'none'`.
- GGUF magic-bytes check is now enforced regardless of the file extension.
- `.cargo/config.toml` (machine-specific libclang path) is no longer tracked;
  a `.cargo/config.toml.example` template is provided.

### Added
- Chat history in prompts: the last turns of the active thread are sent to
  the model (ChatML turns / raw-text block), trimmed to 8 turns / 4000 chars;
  foreign roles are dropped.
- Stop button: a running generation can be interrupted, the partial answer
  is kept and labeled "[generation stopped by user]".
- Embedding model cache (fastembed, ~465 MB) ships inside the MSI — first
  run is now fully offline, as promised ("Zero cloud").
- 14 unit tests covering prompt assembly, ThinkRouter UTF-8 boundaries,
  context formatting and download-filename traversal.

### Fixed
- Corrupted Cyrillic in native window title and backend error messages
  (encoding mishap in a maintenance toolchain).
- UI: switching threads during streaming lost the live message and leaked
  the final answer into another dialog (sidebar is locked while generating).
- Embedding inference no longer blocks the tokio worker thread
  (`spawn_blocking`), so concurrent commands stay responsive.
- Config writes are atomic (write-to-temp + rename) — an abrupt exit can no
  longer produce a truncated `config.toml`.
- Model preload failures are now visible in the status bar with the reason
  (previously only in an invisible release-stderr).
- Without an NVIDIA driver the llama.cpp backend is not initialized at all —
  the friendly driver message shows before any CUDA failure.
- Free-port search reports a proper error instead of panicking;
  broken `.part.meta.json` no longer silently resets a resumable download;
  `gen_bench` fails loudly on an unreadable questions file.

### Changed
- `generator_stub` renamed to `llm_loaded` (the old name was semantically
  inverted since the stub era).
- Default generation parameters synced with the production config
  (`max_tokens=5500`, `n_ctx=12288`) so a config without a `[generation]`
  section no longer truncates reasoning models.
- MSI size grew to ~1.1 GB (embedding model cache is bundled).

## [1.1.0] — 2026-08-28

### Added
- 1-click MSI installer with all binary dependencies inside (CUDA runtime,
  cuBLAS, ONNX Runtime, Qdrant, knowledge base) — nothing to install manually.
- Qdrant as a Tauri sidecar: automatic start, dynamic port selection,
  readiness polling, guaranteed shutdown on app exit (no zombie processes).
- NVIDIA driver pre-flight check with a user-friendly error and download link.
- First-run experience: user data provisioned to `%APPDATA%\ai-mentor`,
  model download URL prefilled — one button to get the model.
- Unified Markdown renderer: streaming output is pixel-identical to the
  final answer (single `buildAiText()` factory, think-details + .md-body).
- Auto-resume downloads with progress events and a polling fallback channel.

### Changed
- Binary dependencies are auto-downloaded by `build.rs` with SHA-256
  verification (CUDA: official NVIDIA redistrib manifest; ONNX Runtime and
  Qdrant: pinned hashes) and cached in `target/deps_cache/` — no binaries
  stored in git.
- `LlamaBackend` is owned by `AppState` (`Arc<LlamaBackend>`) instead of a
  global `static OnceLock` — deterministic VRAM release on exit.
- Synchronous `std::sync::Mutex` replaced with `parking_lot::Mutex`
  (no poisoning) for short critical sections; `tokio::sync::Mutex` kept
  only for locks held across `.await`.
- RAG initialization is async: ONNX embedding model init and knowledge-base
  loading run in `tokio::task::spawn_blocking` (no UI freeze on start).
- Dynamic Qdrant port applied in memory only; `config.toml` on disk keeps
  the default.

### Fixed
- VRAM leak on application exit (CUDA context left behind by the global
  llama.cpp backend).
- Eternal hang of the model downloader when a server accepts a connection
  but stops sending data (`read_timeout` 60 s + total timeout 2 h).
- Poisoned-mutex masking of panics in neighboring threads
  (`unwrap_or_else(|poisoned| ...)` removed everywhere).
- UI freeze during knowledge-base loading (multi-hundred-MB JSONL files
  were read on the main thread).

## [1.0.0] — 2026-08-27

### Added
- Initial release: fully offline AI mentor (Rust/Tauri), RAG over local
  Qdrant, GGUF inference via llama.cpp (CUDA 12), streaming thinking/answer,
  model downloader with SHA-256 and resume.
