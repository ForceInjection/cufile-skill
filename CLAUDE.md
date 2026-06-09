# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

An Agent Skill for NVIDIA cuFile / GPUDirect Storage (GDS) high-performance programming. Covers direct DMA data transfer between NVMe SSDs and GPU memory — bypassing CPU memory entirely. Includes the full cuFile API lifecycle, buffer/file handle management, synchronous/async/batch I/O patterns, cufile.json configuration, performance tuning, and GDS diagnostic workflows.

This is a **knowledge-base skill**, not a library or application. The primary deliverable is `SKILL.md` (loaded by Claude Code when the skill is invoked) supported by detailed reference files, compilable CUDA examples, and a diagnostic shell script.

## Skill Structure

```text
cufile-skill/
├── SKILL.md                         # Main skill file — loaded by Claude Code at invocation
├── CLAUDE.md                        # This file — guidance for working on the skill itself
├── README.md                        # Human-readable overview
├── CHANGELOG.md                      # Version history (keep updated)
├── .gitignore
├── .vscode/                         # VS Code: cmake.sourceDirectory = examples/ (CMake Tools integration)
├── references/                      # 14 detailed topic references (loaded on demand)
│   ├── api-reference.md             # Full cuFile API: functions, structs, enums, constants
│   ├── driver-lifecycle.md          # Driver open/close, version negotiation, multi-GPU
│   ├── buffer-management.md         # GPU buffer registration, alignment, pinning strategies
│   ├── file-handle-management.md    # File handle registration, O_DIRECT, filesystem support
│   ├── sync-io.md                   # Synchronous read/write deep dive
│   ├── async-io.md                  # CUDA stream-based async I/O, stream ordering
│   ├── batch-io.md                  # Batch I/O API: setup, submit, status, cancel
│   ├── performance-tuning.md        # Complete tuning workflow and methodology
│   ├── configuration.md             # cufile.json full reference with tuning guidance
│   ├── error-handling.md            # Error classification, recovery, diagnostics
│   ├── integration-patterns.md      # Production patterns: pipelining, striping, checkpoint
│   ├── hardware-requirements.md     # GPU/NVMe/PCIe compatibility matrix
│   ├── comparison-spdk.md           # cuFile vs SPDK vs standard I/O comparison
│   └── quick-reference.md           # One-page cheat sheet: types, APIs, errors, requirements
├── assets/                          # Static resources (currently empty)
├── examples/                        # 6 compilable CUDA C examples + shared utilities
│   ├── CMakeLists.txt               # Build all examples; auto-detects GPU arch from nvidia-smi
│   ├── 01-driver-init.cu            # Driver lifecycle + properties query
│   ├── 02-sync-read-write.cu        # Sync read/write with GPU buffers
│   ├── 03-async-read-write.cu       # Async I/O via CUDA streams
│   ├── 04-batch-io.cu               # Batch I/O for high throughput
│   ├── 05-end-to-end-pipeline.cu    # Double-buffered prefetch pipeline
│   ├── 06-alignment-check.cu        # Validate buffer/file alignment for GDS
│   └── common/
│       ├── cufile_utils.h           # Shared helpers: check_gds_available, check_alignment, etc.
│       └── cufile_utils.cu
└── scripts/
    └── check_gds.sh                 # Comprehensive GDS readiness check (needs root for ACS)
```

## Key Design Decisions

- **Language**: CUDA C/C++ for examples. cuFile is a C API, but GPU buffer management requires CUDA (`cudaMalloc`, etc.). `.cu` extension throughout.
- **Scope**: Application-level (user-space) cuFile API programming. Not NVIDIA kernel driver (`nvidia-fs.ko`) development. Not NVMe protocol-level programming (see `nvme-programming` skill for that).
- **Pattern**: Follows the `nvme-programming` skill structure: SKILL.md + references/ + examples/. SKILL.md as progressive disclosure (lightweight, stays in context), detailed references load on demand.
- **Compatibility mode is central**: Every section emphasizes GDS vs compat mode detection because silent fallback is THE #1 cuFile performance pitfall.
- **Trigger Keywords**: Both English (cuFile, GPUDirect Storage, GDS, gdscheck, etc.) and Chinese (GPU直通存储, GPU数据流水线, etc.) to support bilingual users.
- **Conventional Commits**: Commit messages follow the `type(scope): description` format. Use the `update-submitter` skill to auto-generate well-formed commits from staged changes.

## Building & Running Examples

All examples require: CUDA Toolkit (12.0+), cuFile library, and nvidia-fs kernel module.

### Build all examples

```bash
cd examples
cmake -B build -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90"
cmake --build build -j$(nproc)
```

CMakeLists.txt auto-detects GPU architecture from `nvidia-smi` if `CMAKE_CUDA_ARCHITECTURES` is not set, falling back to `sm_80` (Ampere). It also checks for nvidia-fs kernel module, `/etc/cufile.json`, and libcufile — all non-fatal for build, but reported as warnings.

### Build a single example

```bash
cd examples
cmake -B build -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90"
cmake --build build --target 01-driver-init -j$(nproc)
```

### Debug build (with line info for profiling)

```bash
cd examples
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90"
cmake --build build -j$(nproc)
```

### Run examples

Examples accessing NVMe devices typically need `sudo`:

```bash
sudo build/01-driver-init
sudo build/02-sync-read-write /mnt/nvme/testfile
sudo build/03-async-read-write /mnt/nvme/testfile
sudo build/04-batch-io /mnt/nvme/testfile
sudo build/05-end-to-end-pipeline /mnt/nvme/testfile
sudo build/06-alignment-check /mnt/nvme
```

## Running Diagnostics Script

```bash
# Platform-level checks only (GPU, CUDA, nvidia-fs, NVMe, PCIe topology, ACS, GDS tools, cufile.json)
bash scripts/check_gds.sh

# Full check including filesystem verification at a mount point
bash scripts/check_gds.sh /mnt/nvme

# ACS check requires root — run with sudo for complete results
sudo bash scripts/check_gds.sh /mnt/nvme
```

The script covers 7 sections: GPU, CUDA Toolkit, nvidia-fs kernel module, NVMe devices, PCIe topology & ACS, GDS tools, filesystem (optional), and cufile.json configuration. ACS (Access Control Services) check is skipped without root — the script warns and suggests re-running with `sudo`.

### Standalone GDS diagnostic tools

In addition to `check_gds.sh`, NVIDIA provides these CLI tools (installed with `nvidia-gds` package):

```bash
gdscheck -p          # Verify GDS path: GPU ↔ NVMe P2P capability
gdsio -f /mnt/nvme   # Benchmark GDS throughput (read/write)
gdscopyme -f /mnt/nvme -s 1G  # Copy test: verify end-to-end GDS data integrity
```

## Extending This Skill

To add coverage for additional cuFile/GDS topics:

1. Add a new `.md` file in `references/` following the existing pattern:
   - Concept overview (1-2 paragraphs)
   - Key API/struct/parameter tables
   - Code snippets in CUDA C
   - Common errors and pitfalls
   - Cross-reference to related reference files

2. Add corresponding code in `examples/` if applicable:
   - Self-contained `.cu` file with clear compile/run instructions in header comment
   - Use `common/cufile_utils.h` for shared helpers
   - Add to `examples/CMakeLists.txt` as a new `add_executable` + `target_link_libraries` block

3. Update the reference file table and example table in `SKILL.md`

4. Add relevant trigger keywords to the `triggers` list in `SKILL.md` frontmatter

5. Update `CHANGELOG.md` with the addition

### Reviewing and committing changes

Before committing, validate against Agent Skill best practices:

```
/agent-skill-reviewer
```

To generate a Conventional Commits message from staged changes and create a PR:

```
/update-submitter
```

## Prerequisites Reference

| Component     | Requirement                         | Check Command                             |
| ------------- | ----------------------------------- | ----------------------------------------- |
| GPU           | Pascal (SM 6.0+) or newer           | `nvidia-smi --query-gpu=compute_cap`      |
| NVIDIA driver | 470.57.02+ (R470), 525+ recommended | `nvidia-smi`                              |
| CUDA Toolkit  | 11.4+ (cuFile bundled), 12.0+ rec.  | `nvcc --version`                          |
| libcufile     | Bundled with CUDA Toolkit           | `find /usr/local/cuda -name "libcufile*"` |
| nvidia-fs.ko  | Loaded kernel module                | `lsmod \| grep nvidia_fs`                 |
| NVMe SSD      | PCIe Gen3 x4+, with CMB or P2P      | `lspci -vv \| grep "Non-Volatile"`        |
| Filesystem    | ext4 / xfs / GPFS (NOT NFS/CIFS)    | `df -T /mnt/nvme`                         |
| GDS Tools     | gdscheck, gdsio                     | `which gdscheck gdsio`                    |

## Related Skills

- `nvme-programming` — NVMe protocol fundamentals (queue model, PRP/SGL, doorbells). Use when you need to understand what cuFile does under the hood or debug at the NVMe protocol level.
- `cuda-knowledge` — CUDA API reference (streams, memory types, `cudaMalloc`). Use when writing GPU kernels that consume/produce data for cuFile.
- `cuda-samples` — Working CUDA code patterns. Use when building end-to-end GPU compute + cuFile I/O pipelines.
- `agent-skill-reviewer` — Review Agent Skill directories and SKILL.md files against best practices. Use before committing changes to this skill.
- `update-submitter` — Generate standardized Conventional Commits from staged changes. Use when committing or creating a PR.
