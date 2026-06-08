# Changelog

All notable changes to the cuFile programming skill.

## [1.0.0] — 2026-06-08

### Added

- Initial release of cuFile programming Agent Skill
- `SKILL.md` — Main skill file with cuFile/GDS API lifecycle, performance tuning, and quick reference
- `CLAUDE.md` — Agent guidance with skill structure, design decisions, and cross-skill decision tree
- `README.md` — Human-readable project overview

### Reference Files (14 files)

- `references/api-reference.md` — Complete cuFile API: function signatures, structs, enums, constants
- `references/driver-lifecycle.md` — Driver open/close, version negotiation, multi-GPU setup
- `references/buffer-management.md` — GPU buffer registration, alignment guarantees, pinning strategies
- `references/file-handle-management.md` — File handle registration, O_DIRECT, filesystem support
- `references/sync-io.md` — Synchronous read/write with partial I/O handling
- `references/async-io.md` — CUDA stream-based async I/O with compute overlap
- `references/batch-io.md` — Batch I/O API for high-throughput small I/O
- `references/performance-tuning.md` — End-to-end tuning workflow with benchmarking methodology
- `references/configuration.md` — cufile.json complete reference with templates
- `references/error-handling.md` — Error classification, retry patterns, diagnostic utilities
- `references/integration-patterns.md` — Production patterns: pipeline, checkpoint, striping, streaming
- `references/hardware-requirements.md` — GPU/NVMe/PCIe compatibility matrix and verification checklist
- `references/comparison-spdk.md` — cuFile vs SPDK vs POSIX I/O performance comparison
- `references/quick-reference.md` — One-page cheat sheet: types, API layering, errors, requirements

### Code Examples (6 examples + 2 shared utilities)

- `examples/01-driver-init.cu` — Driver lifecycle + properties query
- `examples/02-sync-read-write.cu` — Sync read/write with GPU buffer verification
- `examples/03-async-read-write.cu` — Async I/O via CUDA streams with compute overlap
- `examples/04-batch-io.cu` — Batch I/O with sync vs batch benchmark comparison
- `examples/05-end-to-end-pipeline.cu` — Double-buffered prefetch pipeline (production pattern)
- `examples/06-alignment-check.cu` — GDS alignment and readiness diagnostic
- `examples/common/cufile_utils.h` — Shared helper function declarations
- `examples/common/cufile_utils.cu` — Shared helper function implementations
- `examples/CMakeLists.txt` — Build system with auto-detection and dependency checks

### Scripts

- `scripts/check_gds.sh` — Comprehensive GDS readiness check covering GPU, NVMe, PCIe topology, ACS, filesystem

### Cross-References

- Links to `nvme-programming` skill for NVMe protocol fundamentals
- Links to `cuda-knowledge` skill for CUDA stream and memory management
- Links to `cuda-samples` skill for GPU kernel code patterns

### Design

- Progressive disclosure: SKILL.md stays lightweight (512 lines), detailed content in 14 reference files
- Compatibility mode awareness throughout (the #1 cuFile performance pitfall)
- Bilingual support: English + Chinese trigger keywords and section headings
- Follows `nvme-programming` skill structure conventions
