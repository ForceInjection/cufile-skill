# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

An Agent Skill for NVIDIA cuFile / GPUDirect Storage (GDS) high-performance programming. Covers direct DMA data transfer between NVMe SSDs and GPU memory — bypassing CPU memory entirely. Includes the full cuFile API lifecycle, buffer/file handle management, synchronous/async/batch I/O patterns, cufile.json configuration, performance tuning, and GDS diagnostic workflows.

## Skill Structure

```text
cufile-skill/
├── SKILL.md                         # Main skill file — loaded by Claude Code
├── CLAUDE.md                        # This file — project guidance
├── README.md                        # Human-readable overview
├── CHANGELOG.md                      # Version history
├── .gitignore
├── references/                      # Detailed topic references (grep-friendly)
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
├── examples/                        # Compilable CUDA C examples
│   ├── CMakeLists.txt               # Build all examples with CMake + CUDAToolkit
│   ├── 01-driver-init.cu            # Driver lifecycle + properties query
│   ├── 02-sync-read-write.cu        # Sync read/write with GPU buffers
│   ├── 03-async-read-write.cu       # Async I/O via CUDA streams
│   ├── 04-batch-io.cu               # Batch I/O for high throughput
│   ├── 05-end-to-end-pipeline.cu    # Double-buffered prefetch pipeline
│   ├── 06-alignment-check.cu        # Validate buffer/file alignment for GDS
│   └── common/
│       ├── cufile_utils.h           # Shared helper functions
│       └── cufile_utils.cu
└── scripts/
    └── check_gds.sh                 # Verify GDS readiness on a system
```

## Key Design Decisions

- **Language**: CUDA C/C++ for examples. cuFile is a C API, but GPU buffer management requires CUDA (`cudaMalloc`, etc.). `.cu` extension throughout.
- **Scope**: Application-level (user-space) cuFile API programming. Not NVIDIA kernel driver (`nvidia-fs.ko`) development. Not NVMe protocol-level programming (see `nvme-programming` skill for that).
- **Pattern**: Follows the `nvme-programming` skill structure: SKILL.md + references/ + examples/. SKILL.md as progressive disclosure (lightweight, stays in context), detailed references load on demand.
- **Compatibility mode is central**: Every section emphasizes GDS vs compat mode detection because silent fallback is THE #1 cuFile performance pitfall.
- **Integration with existing skills**: Cross-references `nvme-programming` (NVMe protocol fundamentals when you need to go deeper) and CUDA skills (kernel/stream management).
- **Trigger Keywords**: Both English (cuFile, GPUDirect Storage, GDS, gdscheck, etc.) and Chinese (GPU直通存储, GPU数据流水线, etc.) to support bilingual users.

## How to Use This Skill

### When developing cuFile applications

- Load `SKILL.md` for quick reference: API lifecycle, performance checklist, common traps, quick reference card
- Dive into `references/` files for detailed field layouts, function signatures, and tuning methodology
- Use `examples/` as starting templates for common patterns (driver init, sync I/O, async I/O, batch, pipeline)
- Run `scripts/check_gds.sh` before starting development to verify GDS readiness

### Search patterns (when reference files are loaded)

```bash
# Find API function details
grep -r "cuFileRead\|cuFileWrite\|cuFileBufRegister\|cuFileHandleRegister" references/

# Find error codes
grep -r "CU_FILE_\|CUfileOpError" references/

# Find configuration parameters
grep -r "max_direct_io_size\|max_device_cache_size\|cufile.json" references/

# Find performance tuning guidance
grep -r "alignment\|batch size\|IO size\|throughput\|compat" references/

# Find hardware requirements
grep -r "Pascal\|Ampere\|Hopper\|PCIe Gen\|ACS\|P2P" references/
```

## When to Use This Skill vs Other Skills

### ✅ Use cufile-skill when

- You need to move data directly between NVMe storage and GPU memory
- You're writing/debugging applications that use the cuFile API (`cuFileRead`, `cuFileWrite`, etc.)
- You're tuning cuFile performance (alignment, IO sizing, batch aggregation)
- You need to configure GDS (`cufile.json`, driver properties)
- You're building GPU data pipelines, checkpoint/restore, or streaming ingest
- You're diagnosing whether GDS is actually engaged

### ❌ Do NOT use cufile-skill when

- **You need NVMe protocol-level programming** (PRP/SGL construction, doorbell registers, SQ/CQ management) — use the **`nvme-programming`** skill instead. cuFile abstracts NVMe details away; if you're writing an NVMe driver or using SPDK, this skill is the wrong layer.
- **You're optimizing CUDA kernels or GPU compute** (thread block config, shared memory, Tensor Cores) — use the **`cuda-knowledge`** or **`cuda-optimizer`** skills. cuFile is about data movement, not kernel computation.
- **You need CPU-only NVMe I/O** (no GPU involvement) — cuFile's purpose is GPU-direct I/O. For CPU-only NVMe access, use the **`nvme-programming`** skill or standard POSIX I/O.
- **You're working with non-NVIDIA GPUs** (AMD ROCm, Intel oneAPI) — cuFile is NVIDIA-specific. For AMD GPUs, look at ROCm's HIP equivalents; for cross-platform, use O_DIRECT + `cudaMemcpy`/`hipMemcpy`.
- **You need network-attached storage** (NFS, CIFS, NVMe-oF across network) — cuFile/GDS requires local NVMe storage on the same PCIe root complex. NVMe-oF is covered by the **`nvme-programming`** skill.
- **The user is asking about GPU-to-GPU transfers** (P2P, NCCL) — use **`cuda-knowledge`** (NCCL section). cuFile handles NVMe↔GPU, not GPU↔GPU.

### Decision Quick Reference

| User's Need                      | Correct Skill                       |
| -------------------------------- | ----------------------------------- |
| NVMe → GPU data transfer         | **cufile-skill** (this)             |
| NVMe protocol (commands, queues) | `nvme-programming`                  |
| CUDA kernel optimization         | `cuda-knowledge` / `cuda-optimizer` |
| CPU-side NVMe I/O                | `nvme-programming`                  |
| GPU P2P / NCCL collectives       | `cuda-knowledge`                    |
| AMD GPU storage                  | (not covered yet)                   |

## Building Examples

All examples require: CUDA Toolkit (12.0+), cuFile library, and nvidia-fs kernel module.

```bash
# Configure
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90"

# Build all examples
cmake --build . -j$(nproc)

# Run individual examples (may need sudo for NVMe device access)
sudo ./01-driver-init
sudo ./02-sync-read-write /mnt/nvme/testfile
sudo ./03-async-read-write /mnt/nvme/testfile
sudo ./04-batch-io /mnt/nvme/testfile
sudo ./05-end-to-end-pipeline /mnt/nvme/testfile
sudo ./06-alignment-check /mnt/nvme
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

## Extending This Skill

To add coverage for additional cuFile/NVMe topics:

1. Add a new `.md` file in `references/` following the existing pattern:
   - Concept overview (1-2 paragraphs)
   - Key API/struct/parameter tables
   - Code snippets in CUDA C
   - Common errors and pitfalls
   - Cross-reference to related reference files

2. Add corresponding code in `examples/` if applicable:
   - Self-contained `.cu` file with clear compile/run instructions in header comment
   - Use `common/cufile_utils.h` for shared helpers
   - Add to `examples/CMakeLists.txt`

3. Update the reference file table and example table in `SKILL.md`

4. Consider adding entries in the trigger keyword list in `SKILL.md` frontmatter

## Related External Resources

- NVIDIA GPUDirect Storage Documentation: https://docs.nvidia.com/gpudirect-storage/
- cuFile API Reference: https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html
- GDS Troubleshooting Guide: https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html
- CUDA Toolkit (includes libcufile): https://developer.nvidia.com/cuda-downloads
- GPUDirect RDMA Documentation: https://docs.nvidia.com/cuda/gpudirect-rdma/
- NVIDIA Magnum IO: https://developer.nvidia.com/magnum-io

## Related Skills

- `nvme-programming` — NVMe protocol fundamentals (queue model, PRP/SGL, doorbells). Use when you need to understand what cuFile does under the hood or debug at the NVMe protocol level.
- `cuda-knowledge` — CUDA API reference (streams, memory types, `cudaMalloc`). Use when writing GPU kernels that consume/produce data for cuFile.
- `cuda-samples` — Working CUDA code patterns. Use when building end-to-end GPU compute + cuFile I/O pipelines.
