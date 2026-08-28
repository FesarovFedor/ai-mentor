# Description

Please include a summary of the change and which issue is fixed
(if applicable). Fixes #<issue-number>

## Type of change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Checklist

- [ ] `cargo fmt --all` applied
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` passes
- [ ] `cargo tauri build` succeeds
- [ ] I have read [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) — key
      invariants respected (prompt parity, seed determinism, UTF-8 decoder,
      BOS, XSS policy)
- [ ] Tests / manual verification described below

## How has this been tested?

Describe the verification steps.
