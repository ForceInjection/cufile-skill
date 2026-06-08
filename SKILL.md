---
name: cufile-skill
description: >
  NVIDIA cuFile / GPUDirect Storage (GDS) 高性能编程 — GPU 与 NVMe SSD 之间的直接 DMA 数据传输，
  绕过 CPU 内存，实现最低延迟和最高吞吐。涵盖同步/异步/批量 I/O、缓冲区注册、
  cufile.json 调优、GDS 兼容性检测与故障排查。
  Use when writing, debugging, or tuning cuFile applications that need direct NVMe-to-GPU data transfer.
  Covers driver lifecycle, buffer/handle registration, sync/async/batch I/O, cufile.json configuration,
  performance tuning (alignment, IO size, batch aggregation), and GDS fallback detection.
  Triggers on: cuFile, cufile, GPUDirect Storage, GDS, GPU Direct Storage, GPU+NVMe, nvidia-fs,
  cuFileRead, cuFileWrite, cuFileBufRegister, cuFileHandleRegister, gdscheck, gdsio, gdscopyme,
  CUDA stream IO, GPU storage, GPU data pipeline, direct DMA, NVMe to GPU, GPU I/O acceleration,
  GPU直通存储, GPU数据流水线, 批量IO, 异步IO, O_DIRECT, cufile.json, libcufile.
---

# cuFile / GPUDirect Storage 高性能编程

## Core Philosophy

**GDS is a DMA shortcut.** GPUDirect Storage enables the NVMe controller to DMA data directly into GPU memory via PCIe peer-to-peer (P2P) transfers. The CPU does not touch the data — no bounce buffer, no `cudaMemcpy`, no kernel-mode copies.

**Compatibility mode is THE #1 pitfall.** cuFile silently falls back to a CPU bounce buffer (two PCIe hops: NVMe → CPU → GPU) when GDS prerequisites aren't met. Your code still works, but performance is 2–3× worse. Always verify with `cuFileDriverGetProperties()` and `gdscheck` before measuring performance.

**Registration has cost. Reuse has value.** `cuFileBufRegister` and `cuFileHandleRegister` pin pages and set up DMA mappings — these are not free. Register once at init, reuse for the lifetime of the workload. Never register/deregister inside a hot loop.

**The API is simple — the performance is in the details.** cuFile has only ~15 core functions. The hard part is alignment, IO sizing, stream ordering, batch aggregation, and knowing whether GDS is actually engaged.

## GPUDirect Storage: 30-Second Architecture

```text
Traditional I/O (no GDS):
  NVMe SSD ──PCIe──► CPU DRAM (bounce buffer) ──PCIe──► GPU DRAM
  ─ Two PCIe hops, CPU memory bandwidth consumed, CPU involved

GPUDirect Storage (GDS enabled):
  NVMe SSD ──PCIe P2P──► GPU DRAM (BAR-mapped)
  ─ Single PCIe hop, zero CPU memory, CPU only submits commands

Compatibility mode (GDS unavailable, fallback):
  NVMe SSD ──PCIe──► CPU DRAM ──PCIe──► GPU DRAM
  ─ Looks the same to your code, but 2–3× worse throughput
```

**How it works under the hood:**

1. `cuFileBufRegister(devPtr, size)` → Pins GPU virtual address range, creates a BAR mapping visible to the PCIe root complex
2. `cuFileHandleRegister(fd, path)` → Verifies the file is on a GDS-compatible filesystem, creates a GDS file descriptor
3. `cuFileRead(file_handle, devPtr, size, offset, 0)` → Submits an NVMe read command with the GPU physical address as the data pointer. The NVMe controller DMAs directly to GPU memory.
4. NVMe completion interrupt → cuFile library processes completion → GPU buffer has the data

**Key requirement:** The NVMe SSD and GPU must be on the same PCIe root complex with PCIe Access Control Services (ACS) disabled for P2P. `gdscheck` validates this topology.

## cuFile API Lifecycle: The 5-Step Pattern

Every cuFile application follows this exact sequence. Get the order wrong = runtime errors.

```text
Step 1: cuFileDriverOpen()
        └── Initialize GDS driver, negotiate API version
Step 2: Allocate & Register GPU Buffers
        ├── cudaMalloc(&devPtr, size)
        └── cuFileBufRegister(devPtr, size, 0)
Step 3: Open & Register File
        ├── fd = open(path, O_DIRECT | O_RDONLY)
        └── cuFileHandleRegister(&file_handle, &descr)
Step 4: Perform I/O
        ├── cuFileRead(file_handle, devPtr, size, offset, 0)
        ├── cuFileWrite(file_handle, devPtr, size, offset, 0)
        ├── cuFileReadAsync(file_handle, devPtr, size, offset, stream)
        └── cuFileWriteAsync(file_handle, devPtr, size, offset, stream)
Step 5: Cleanup (reverse order of setup)
        ├── cuFileHandleDeregister(file_handle)
        ├── close(fd)
        ├── cuFileBufDeregister(devPtr)
        ├── cudaFree(devPtr)
        └── cuFileDriverClose()
```

**Critical ordering rule:** Cleanup in strict reverse order. Deregister the handle BEFORE closing the file descriptor. Deregister the buffer BEFORE freeing GPU memory. Otherwise you get `CU_FILE_INVALID_HANDLE` or worse — silent memory corruption if the GPU frees a page the DMA engine still references.

## API Quick Reference

### Driver & Lifecycle

| Function                            | Purpose                                   | Perf Impact     |
| ----------------------------------- | ----------------------------------------- | --------------- |
| `cuFileDriverOpen()`                | Initialize cuFile driver, negotiate API   | One-time cost   |
| `cuFileDriverClose()`               | Shutdown cuFile driver                    | Cleanup only    |
| `cuFileDriverGetProperties()`       | Query driver version and GDS capabilities | Diagnostic tool |
| `cuFileIsGpuDirectStorageCapable()` | Check if platform can support GDS         | Diagnostic tool |
| `cuFileIsGpuDirectStorageEnabled()` | Check if GDS is currently engaged         | Diagnostic tool |

### Buffer Management

| Function                   | Purpose                                    | Perf Impact                     |
| -------------------------- | ------------------------------------------ | ------------------------------- |
| `cuFileBufRegister()`      | Register GPU buffer for DMA                | **Critical** — enables GDS path |
| `cuFileBufDeregister()`    | Unregister GPU buffer                      | Cleanup only                    |
| `cuFileBufSetAllocFlags()` | Set allocation flags for registered buffer | Advanced tuning                 |

### File Handle Management

| Function                   | Purpose                              | Perf Impact                     |
| -------------------------- | ------------------------------------ | ------------------------------- |
| `cuFileHandleRegister()`   | Register file descriptor for GDS I/O | **Critical** — enables GDS path |
| `cuFileHandleDeregister()` | Unregister file handle               | Cleanup only                    |

### Synchronous I/O

| Function        | Purpose                        | Perf Impact           |
| --------------- | ------------------------------ | --------------------- |
| `cuFileRead()`  | Read from file into GPU buffer | Blocks calling thread |
| `cuFileWrite()` | Write GPU buffer to file       | Blocks calling thread |

### Asynchronous I/O (CUDA Streams)

| Function                 | Purpose                                   | Perf Impact                     |
| ------------------------ | ----------------------------------------- | ------------------------------- |
| `cuFileReadAsync()`      | Read into GPU buffer on a CUDA stream     | Overlaps I/O with compute       |
| `cuFileWriteAsync()`     | Write GPU buffer to file on a CUDA stream | Overlaps I/O with compute       |
| `cuFileStreamRegister()` | Associate a CUDA stream with cuFile       | Required before first async I/O |

### Batch I/O

| Function                   | Purpose                              | Perf Impact                           |
| -------------------------- | ------------------------------------ | ------------------------------------- |
| `cuFileBatchIOSetUp()`     | Create batch I/O handle              | Setup cost, amortized across requests |
| `cuFileBatchIOSubmit()`    | Submit multiple I/O requests at once | **Critical** — 2-10× throughput gain  |
| `cuFileBatchIOGetStatus()` | Check completion status of batch     | Polling overhead                      |
| `cuFileBatchIOCancel()`    | Cancel pending batch requests        | Error recovery                        |
| `cuFileBatchIODestroy()`   | Tear down batch I/O handle           | Cleanup only                          |

### Error & Diagnostic Utilities

| Function                 | Purpose                                   |
| ------------------------ | ----------------------------------------- |
| `cuFileGetErrorString()` | Convert error code to human-readable text |
| `cuFileOpError()`        | Query last error for a specific operation |

## Performance Tuning: The Complete Checklist

### Before You Start: Verify GDS is Working

**This is the single most important step.** Many "perf tuning" sessions end with "GDS was never engaged."

```bash
# 1. Check GDS toolchain
gdscheck -p              # Platform check: GPU topology, PCIe P2P, ACS state
gdscheck -f /mnt/nvme    # Filesystem check: mount options, O_DIRECT support
gdsio -f /mnt/nvme/test  # Functional test: write + read + verify

# 2. Verify in code
CUfileDrvProps_t props;
cuFileDriverGetProperties(&props);
printf("GDS capable: %s\n", props.is_gds_capable ? "YES" : "NO");
printf("GDS enabled:  %s\n", props.is_gds_enabled ? "YES" : "NO");
printf("Version:      %d.%d\n", props.major, props.minor);
```

### Performance Tuning Checklist (ordered by impact)

- [ ] **Buffer Alignment** — GPU buffers must be 4KB-aligned for GDS. Use `cuMemAlloc` (not `cudaMalloc`) for guaranteed alignment. Misaligned buffers silently fall back to compat mode.
- [ ] **IO Size** — Minimum 4KB for GDS path. For high throughput, use IO sizes ≥ 1MB. Small IOs (≤ 64KB) have high per-operation overhead. Batch them.
- [ ] **File Open Flags** — Files MUST be opened with `O_DIRECT` for the GDS path. Without it, you get compat mode. Also use `O_RDONLY` or `O_WRONLY` — O_RDWR has additional locking overhead.
- [ ] **Buffer Registration Strategy** — Register buffers once at initialization, reuse for thousands of IOs. Registration pins GPU pages — pinning is expensive. Unregistration unpins. A hot loop that registers/unregisters is wasting 90%+ of its time in pinning.
- [ ] **Batch I/O** — For workloads with many small IOs, use `cuFileBatchIOSetUp` + `cuFileBatchIOSubmit`. Reduces per-IO submission overhead by 2–10×. See `references/batch-io.md`.
- [ ] **Async I/O + CUDA Stream Overlap** — Use `cuFileReadAsync` on one stream while your kernel runs on another. This hides I/O latency behind compute. See `references/async-io.md`.
- [ ] **CUDA Stream Registration** — Call `cuFileStreamRegister()` once per stream before first async I/O. Don't re-register the same stream.
- [ ] **Thread to File Mapping** — For multi-file workloads, dedicate one CPU thread per file. cuFile handles are not thread-safe for concurrent operations on the same handle.
- [ ] **NUMA Affinity** — CPU threads submitting I/O should be on the NUMA node closest to the PCIe root complex serving the NVMe device. Use `numactl --cpunodebind=N` or `sched_setaffinity()`.
- [ ] **cufile.json Tuning** — The global configuration file at `/etc/cufile.json` (or `$CUFILE_CONFIG`) controls GDS behavior. Key parameters:
  - `max_direct_io_size`: Maximum IO size for direct GDS path (default: 1GB)
  - `max_device_cache_size`: Maximum device-internal cache for GDS (default: 128MB)
  - `max_device_pinned_mem_size`: Maximum device pinned memory (default: unlimited)
  - `enable_compat_mode`: Allow fallback to compat mode (default: true — turn off to get hard errors instead of silent fallback)
- [ ] **Filesystem Choice** — GDS requires: ext4, xfs, or GPFS with specific mount options. NFS/CIFS/FUSE are NOT supported. Verify with `gdscheck -f <mountpoint>`.
- [ ] **PCIe Topology** — GPU and NVMe must be on the same PCIe root complex with ACS disabled for P2P. Multi-socket systems often have separate root complexes — if GPU is on socket 0 and NVMe on socket 1, GDS may not work.

### IO Size Guidelines

| Workload                      | Recommended IO Size | Batch?           | Reasoning                                      |
| ----------------------------- | ------------------- | ---------------- | ---------------------------------------------- |
| Small random reads (4KB-64KB) | 4KB-64KB            | **Yes**          | Batch amortizes per-IO cost 2–5×               |
| Medium sequential reads       | 256KB-1MB           | Optional         | Large enough that per-IO cost is low           |
| Large sequential reads        | 1MB-16MB            | No               | Single IO saturates PCIe bandwidth             |
| Checkpoint / model save       | 16MB-256MB          | No               | Max throughput via single large write          |
| Data pipeline (train+ingest)  | 1MB-8MB             | Yes (double buf) | Batch N reads while kernel processes batch N-1 |

## Configuration: cufile.json Reference

The cuFile configuration file controls global GDS behavior. Default location: `/etc/cufile.json`. Override with environment variable `CUFILE_CONFIG`.

### Complete Key Reference

```json
{
  "properties": {
    "max_device_cache_size": 134217728,
    "max_device_pinned_mem_size": 0,
    "max_direct_io_size": 1073741824,
    "max_device_allocated_mem_size": 0,
    "enable_compat_mode": true,
    "profiling": {
      "enable": false,
      "trace": false
    },
    "fs": {
      "ext4": {
        "max_direct_io_chunk_size": 0
      },
      "xfs": {
        "max_direct_io_chunk_size": 0
      }
    }
  }
}
```

| Key                                  | Default   | Description                                                  | Tuning Guidance                                                 |
| ------------------------------------ | --------- | ------------------------------------------------------------ | --------------------------------------------------------------- |
| `max_device_cache_size`              | 128MB     | GDS-internal device cache size in bytes                      | Increase to 256MB-512MB for large sequential workloads          |
| `max_device_pinned_mem_size`         | 0 (unlim) | Maximum GPU memory GDS can pin (bytes). 0 = unlimited        | Set a limit to prevent GDS from starving compute of GPU memory  |
| `max_direct_io_size`                 | 1GB       | Maximum single IO request size for GDS path (bytes)          | Increase to 8GB-16GB for checkpoint workloads                   |
| `max_device_allocated_mem_size`      | 0 (unlim) | Maximum GPU memory GDS can allocate for internal use (bytes) | Leave at 0 unless memory-constrained                            |
| `enable_compat_mode`                 | true      | Silently fall back to CPU bounce buffer if GDS unavailable   | Set to `false` during development to catch GDS-not-engaged bugs |
| `profiling.enable`                   | false     | Enable cuFile internal profiling                             | Enable for performance debugging only; has overhead             |
| `profiling.trace`                    | false     | Enable verbose tracing                                       | Enable only for troubleshooting                                 |
| `fs.<type>.max_direct_io_chunk_size` | 0 (auto)  | Per-filesystem maximum chunk size for direct I/O             | Tune per storage type: NVMe vs SSD vs HDD                       |

### Per-Filesystem Tuning

```json
{
  "properties": {
    "fs": {
      "ext4": {
        "max_direct_io_chunk_size": 8388608
      },
      "xfs": {
        "max_direct_io_chunk_size": 16777216
      }
    }
  }
}
```

### GDS Debug Configuration

During development, set these to catch silent fallbacks:

```json
{
  "properties": {
    "enable_compat_mode": false,
    "profiling": {
      "enable": true,
      "trace": false
    }
  }
}
```

With `enable_compat_mode: false`, operations that can't use GDS will fail with an explicit error instead of silently falling back. This is essential for verifying that your alignment, IO size, and file configuration are correct.

## Common Performance Traps

| Symptom                                            | Likely Cause                                  | Investigation                                                 | Fix                                                              |
| -------------------------------------------------- | --------------------------------------------- | ------------------------------------------------------------- | ---------------------------------------------------------------- |
| Throughput 2-3× lower than expected                | GDS not engaged (compat mode)                 | Check `cuFileIsGpuDirectStorageEnabled()` / `gdscheck` output | Fix alignment, enable O_DIRECT, check PCIe topology              |
| `cuFileBufRegister` fails                          | Buffer not 4KB-aligned or GPU memory full     | Check alignment with `(uintptr_t)ptr % 4096`                  | Use `cuMemAlloc` for guaranteed alignment                        |
| `cuFileHandleRegister` fails with error            | File not on supported filesystem              | `gdscheck -f /mnt/nvme`                                       | Re-mount with correct options or use `enable_compat_mode: true`  |
| `cuFileRead` returns fewer bytes than requested    | Partial read (normal for POSIX, rare for GDS) | Check return value against requested size                     | Loop to issue remaining bytes (POSIX semantics)                  |
| Async ops seem sequential                          | GDS not enabled → falls back to CPU sync      | Check stream overlap with CUDA events                         | Verify GDS in code; compat mode async is effectively synchronous |
| Throughput plateaus despite increasing IO size     | Hit `max_direct_io_size` limit                | Check cufile.json                                             | Increase `max_direct_io_size`                                    |
| High CPU usage during I/O                          | Compat mode — CPU is copying data             | `cuFileIsGpuDirectStorageEnabled()` returns false             | Fix GDS prerequisites                                            |
| `cuFileBatchIOSubmit` slower than individual reads | Batch not amortizing overhead                 | Check batch size: too few IOs per batch                       | Increase batch size to ≥16, ensure IOs are large enough          |
| Intermittent `CU_FILE_INVALID_HANDLE`              | Handle use-after-close or thread contention   | Check if another thread is using the same handle concurrently | Dedicated handle per thread, or add synchronization              |
| Memory allocation fails after registering buffers  | GPU memory pinned by GDS                      | `nvidia-smi` shows reserved memory                            | Deregister unused buffers, set `max_device_pinned_mem_size`      |

## Error Handling

### Error Model

cuFile returns `CUfileError_t` from most functions. Check `.err` field — 0 means success. Any non-zero value is an error.

```c
CUfileError_t status;
status = cuFileRead(fh, devPtr, size, offset, 0);
if (status.err != 0) {
    fprintf(stderr, "cuFileRead failed: %s (err=%d)\n",
            cuFileGetErrorString(status), status.err);
}
```

### Common Error Codes

| Error Constant              | Value | Meaning                                       | Recovery                              |
| --------------------------- | ----- | --------------------------------------------- | ------------------------------------- |
| `CU_FILE_SUCCESS`           | 0     | Success                                       | —                                     |
| `CU_FILE_NOT_FOUND`         | 2     | File or path not found                        | Check file path                       |
| `CU_FILE_INVALID_HANDLE`    | 5     | Handle is invalid or closed                   | Check handle lifecycle, thread safety |
| `CU_FILE_INVALID_PARAMETER` | 6     | Bad parameter (NULL pointer, bad alignment)   | Validate all parameters               |
| `CU_FILE_DISK_FULL`         | 12    | No space left on device                       | Free disk space                       |
| `CU_FILE_INTERNAL_ERROR`    | 14    | Internal driver error                         | Check dmesg, reinitialize driver      |
| `CU_FILE_NOT_REGISTERED`    | 15    | Buffer not registered with cuFile             | Call `cuFileBufRegister` first        |
| `CU_FILE_NOT_OPENED`        | 16    | File not opened with `O_DIRECT`               | Re-open file with correct flags       |
| `CU_FILE_NOT_SUPPORTED`     | 18    | Operation not supported (e.g., GDS not avail) | Check GDS prerequisites               |

### Error Handling Pattern

```c
CUfileError_t cuFileCheck(CUfileError_t status, const char *op) {
    if (status.err != 0) {
        fprintf(stderr, "ERROR: %s failed: %s (err=%d)\n",
                op, cuFileGetErrorString(status), status.err);
        exit(EXIT_FAILURE);
    }
    return status;
}

// Usage:
cuFileCheck(cuFileRead(fh, devPtr, size, offset, 0), "cuFileRead");
```

### GDS Status Diagnostic Pattern

Always include this diagnostic block in your initialization:

```c
void check_gds_status() {
    CUfileDrvProps_t props;
    CUfileError_t st = cuFileDriverGetProperties(&props);
    if (st.err != 0) {
        fprintf(stderr, "WARNING: cuFileDriverGetProperties failed: %s\n",
                cuFileGetErrorString(st));
        return;
    }
    printf("cuFile driver: v%d.%d\n", props.major, props.minor);
    printf("GDS capable:  %s\n", props.is_gds_capable ? "YES" : "NO");
    printf("GDS enabled:   %s\n", props.is_gds_enabled ? "YES" : "NO");
    printf("Max IO size:   %lu bytes (%.1f GB)\n",
           props.max_device_direct_io_size,
           props.max_device_direct_io_size / 1e9);
    if (!props.is_gds_capable) {
        printf("\nACTION REQUIRED: GDS not capable on this system.\n");
        printf("  Run: gdscheck -p\n");
    }
    if (props.is_gds_capable && !props.is_gds_enabled) {
        printf("\nACTION REQUIRED: GDS capable but not enabled.\n");
        printf("  Check /etc/cufile.json: enable_compat_mode\n");
    }
}
```

## Integration Patterns

### Pattern 1: Double-Buffered Prefetch Pipeline

The most common high-performance pattern. While the GPU kernel processes batch N, cuFile reads batch N+1 into the other buffer.

```text
Time ─────────────────────────────────────────────────►

Stream 1 (I/O):   [Read B0]            [Read B1]            [Read B2]
Stream 2 (Kernel):          [Process B0]         [Process B1]         [Process B2]

Result: I/O latency is hidden behind compute. Throughput = max(IO_bw, compute_bw),
        not IO_bw + compute_bw.
```

See `examples/05-end-to-end-pipeline.cu` for complete code.

### Pattern 2: Checkpoint/Restart

Model weights live in GPU memory. cuFile writes them directly to NVMe without a CPU copy.

```c
// Save checkpoint: GPU memory → NVMe (no CPU involved)
cuFileWrite(model_fh, model_weights_gpu, model_size, checkpoint_offset, 0);

// Restore checkpoint: NVMe → GPU memory (no CPU involved)
cuFileRead(model_fh, model_weights_gpu, model_size, checkpoint_offset, 0);
```

Performance: With GDS, 40GB model checkpoint writes at 20+ GB/s on a PCIe Gen4 x16 NVMe. Traditional CPU-bounce path caps at ~12 GB/s (limited by CPU memory bandwidth).

### Pattern 3: Multi-GPU Multi-NVMe Striping

For extreme throughput, stripe data across multiple NVMe devices, each serving a dedicated GPU.

```text
GPU 0 ◄──PCIe P2P──► NVMe 0  (stream 0 data)
GPU 1 ◄──PCIe P2P──► NVMe 1  (stream 1 data)
GPU 2 ◄──PCIe P2P──► NVMe 2  (stream 2 data)
GPU 3 ◄──PCIe P2P──► NVMe 3  (stream 3 data)
```

Key constraint: Each GPU-NVMe pair must be on the SAME PCIe root complex. `gdscheck -p` reveals the topology.

### Pattern 4: Streaming Ingest with In-Place Processing

Data lands in GPU memory via cuFile, then a CUDA kernel processes it in-place — zero copies, zero CPU involvement.

```c
// Step 1: Read data directly into GPU memory
cuFileReadAsync(fh, gpu_buffer, chunk_size, offset, stream1);

// Step 2: No cudaMemcpy needed — data is already on GPU
// Step 3: Launch kernel on stream2 (waits on stream1 completion)
cudaStreamWaitEvent(stream2, done_event);
process_kernel<<<grid, block, 0, stream2>>>(gpu_buffer, chunk_size);

// Step 4: Write results out (if needed)
cuFileWriteAsync(out_fh, gpu_buffer, result_size, out_offset, stream2);
```

## Relationship with NVMe Programming Skill

cuFile is a **higher-level abstraction** over NVMe. You do NOT construct NVMe commands, manage PRP lists, or ring doorbell registers — cuFile handles all of that internally.

**When to use cuFile (this skill) vs NVMe (nvme-programming skill):**

| Scenario                                       | Use                                                                                          |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------- |
| I need to read/write data between NVMe and GPU | **cuFile** — this skill                                                                      |
| I need to read/write data between NVMe and CPU | **nvme-programming** — use ioctl/SPDK                                                        |
| I need to understand NVMe protocol details     | **nvme-programming** — PRP, SQE, CQE                                                         |
| I need to debug a cuFile performance issue     | Start here; drop to nvme-programming for low-level diagnostics                               |
| I'm building a storage system with GPU compute | **cuFile** for data path; **nvme-programming** for understanding what cuFile does underneath |

**What cuFile handles for you (you don't need to know these to use cuFile):**

- PRP/SGL construction for GPU physical addresses
- NVMe command submission and completion polling
- Doorbell register management
- Queue pair allocation and management
- PCIe P2P BAR mapping setup

**What you still need to understand (covered in this skill):**

- O_DIRECT and filesystem requirements
- Buffer alignment and registration
- IO size vs throughput tradeoffs
- Batch submission strategies
- CUDA stream synchronization
- GDS vs compat mode detection

For deep NVMe understanding (when you need to debug at the protocol level), see the `nvme-programming` skill.

## Reference Files

Detailed reference material is organized by topic — load the relevant file when going deep:

| File                                   | Use When                                                                    |
| -------------------------------------- | --------------------------------------------------------------------------- |
| `references/api-reference.md`          | Looking up function signatures, struct definitions, enum values             |
| `references/driver-lifecycle.md`       | Multi-threaded driver init, version negotiation, CUDA context requirements  |
| `references/buffer-management.md`      | Buffer registration strategies, alignment guarantees, large buffer handling |
| `references/file-handle-management.md` | Supported filesystems, mount requirements, O_DIRECT semantics               |
| `references/sync-io.md`                | Synchronous I/O tuning, partial reads, error recovery                       |
| `references/async-io.md`               | CUDA stream-based async I/O, stream ordering, overlapping patterns          |
| `references/batch-io.md`               | Batch I/O API deep dive, batch sizing formula, cancellation semantics       |
| `references/performance-tuning.md`     | End-to-end tuning workflow, benchmarking methodology, profiler integration  |
| `references/configuration.md`          | Complete cufile.json reference, per-filesystem tuning, debug flags          |
| `references/error-handling.md`         | All error codes, error classification, recovery strategies                  |
| `references/integration-patterns.md`   | Production patterns: pipelining, striping, multi-GPU, checkpoint/restore    |
| `references/hardware-requirements.md`  | GPU/NVMe/PCIe compatibility, topology requirements, ACS/P2P details         |
| `references/comparison-spdk.md`        | cuFile vs SPDK vs standard buffered I/O — when to use which                 |
| `references/quick-reference.md`        | One-page cheat sheet: types, API layering, error codes, GDS requirements    |

## Code Examples

Ready-to-compile CUDA C examples using the cuFile API:

| File                                 | Demonstrates                                                                       |
| ------------------------------------ | ---------------------------------------------------------------------------------- |
| `examples/01-driver-init.cu`         | `cuFileDriverOpen/Close`, properties query, GDS capability/enabled checks          |
| `examples/02-sync-read-write.cu`     | Buffer registration, O_DIRECT file open, `cuFileRead/Write` with data verification |
| `examples/03-async-read-write.cu`    | CUDA stream registration, `cuFileReadAsync/WriteAsync`, stream synchronization     |
| `examples/04-batch-io.cu`            | Batch I/O setup, submission, status polling, cancel, destroy                       |
| `examples/05-end-to-end-pipeline.cu` | Double-buffered prefetch: async read + kernel compute + async write                |
| `examples/06-alignment-check.cu`     | Validate buffer alignment, file offset alignment, IO size alignment for GDS        |

Shared helper code in `examples/common/`:

- `cufile_utils.h` / `cufile_utils.cu` — `check_gds_available()`, `check_alignment()`, `print_error()`, `measure_bandwidth()`

## Search Patterns

When working with the NVIDIA cuFile documentation or source:

```bash
# Find API function documentation
grep -r "cuFileRead\|cuFileWrite\|cuFileBufRegister" /path/to/cufile-docs/

# Find error codes
grep -r "CU_FILE_\|CUfileError_t" /path/to/cufile-docs/

# Find configuration reference
grep -r "cufile.json\|max_direct_io_size\|max_device_cache_size" /path/to/cufile-docs/

# Find GDS version history
grep -r "GDS\|GPUDirect Storage\|gdscheck\|gdsio" /path/to/cufile-docs/

# Find examples in NVIDIA's repo
grep -r "cuFileRead\|cuFileHandleRegister" /usr/local/cuda/gds/samples/
```

## Quick Reference

One-page cheat sheet covering core types, API layering, error codes, GDS requirements, cufile.json keys, IO size selection, and performance diagnostics. See [`references/quick-reference.md`](references/quick-reference.md).
