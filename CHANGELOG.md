# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
