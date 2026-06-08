# cuFile Integration Patterns

## Overview

Production cuFile applications combine direct I/O with GPU compute in end-to-end pipelines. This reference covers proven patterns for training data ingestion, model checkpoint/restore, multi-GPU setups, and GPU-direct data processing.

## Pattern 1: Double-Buffered Prefetch Pipeline

**Use case:** Training pipeline where data must be on GPU before each training step. I/O is overlapped with compute.

```c
typedef struct {
    CUfileDescr_t fh;
    CUdeviceptr buf[2];       // Double buffer
    size_t buf_size;
    cudaStream_t io_stream;
    cudaStream_t compute_stream;
    cudaEvent_t io_done[2];
    cudaEvent_t compute_done[2];
    size_t total_chunks;
    size_t current_chunk;
    int initialized;
} PrefetchPipeline;

int prefetch_init(PrefetchPipeline *p, const char *filepath,
                  size_t buf_size, size_t total_size) {
    p->buf_size = buf_size;
    p->total_chunks = total_size / buf_size;

    // Allocate double buffers
    for (int i = 0; i < 2; i++) {
        cuMemAlloc(&p->buf[i], buf_size);
        cuFileBufRegister(p->buf[i], buf_size, 0);
        cudaEventCreate(&p->io_done[i]);
        cudaEventCreate(&p->compute_done[i]);
    }

    // Open and register file
    int fd = open(filepath, O_DIRECT | O_RDONLY);
    p->fh.cookie = (CUfileDriverCookie)(uintptr_t)fd;
    cuFileHandleRegister(&p->fh, NULL);

    // Set up streams
    cudaStreamCreate(&p->io_stream);
    cudaStreamCreate(&p->compute_stream);
    cuFileStreamRegister(p->fh, p->io_stream);

    // Kick off first chunk read
    cuFileReadAsync(p->fh, p->buf[0], buf_size, 0, 0, p->io_stream);
    cudaEventRecord(p->io_done[0], p->io_stream);

    p->current_chunk = 0;
    p->initialized = 1;
    return 0;
}

// Get the next chunk of data (blocks until available)
CUdeviceptr prefetch_next(PrefetchPipeline *p) {
    int cur = p->current_chunk % 2;
    int next = (p->current_chunk + 1) % 2;

    // Wait for current chunk's I/O to finish
    cudaEventSynchronize(p->io_done[cur]);

    // Prefetch next chunk (while caller processes current)
    if (p->current_chunk + 1 < p->total_chunks) {
        // Make sure next buffer isn't in use
        if (p->current_chunk >= 1) {
            cudaEventSynchronize(p->compute_done[next]);
        }
        cuFileReadAsync(p->fh, p->buf[next], p->buf_size,
                        (p->current_chunk + 1) * p->buf_size,
                        0, p->io_stream);
        cudaEventRecord(p->io_done[next], p->io_stream);
    }

    p->current_chunk++;
    return p->buf[cur];
}

void prefetch_mark_processed(PrefetchPipeline *p) {
    int cur = (p->current_chunk - 1) % 2;
    cudaEventRecord(p->compute_done[cur], p->compute_stream);
}
```

## Pattern 2: Checkpoint / Restore

**Use case:** Save and restore model weights directly between GPU memory and NVMe, minimizing checkpoint latency.

```c
typedef struct {
    CUfileDescr_t fh;
    char filepath[256];
    size_t model_size;
} CheckpointManager;

// Save checkpoint: GPU → NVMe
int checkpoint_save(CheckpointManager *cm, CUdeviceptr model_weights) {
    ssize_t written = cuFileWrite(cm->fh, model_weights,
                                  cm->model_size, 0, 0);
    if (written != cm->model_size) {
        fprintf(stderr, "Checkpoint write failed: %zd / %zu bytes\n",
                written, cm->model_size);
        return -1;
    }
    return 0;
}

// Restore checkpoint: NVMe → GPU
int checkpoint_restore(CheckpointManager *cm, CUdeviceptr model_weights) {
    ssize_t read_bytes = cuFileRead(cm->fh, model_weights,
                                    cm->model_size, 0, 0);
    if (read_bytes != cm->model_size) {
        fprintf(stderr, "Checkpoint read failed: %zd / %zu bytes\n",
                read_bytes, cm->model_size);
        return -1;
    }
    return 0;
}

// Async checkpoint save (non-blocking for training)
int checkpoint_save_async(CheckpointManager *cm, CUdeviceptr model_weights,
                          cudaStream_t stream) {
    CUfileError_t st = cuFileWriteAsync(cm->fh, model_weights,
                                        cm->model_size, 0, 0, stream);
    if (st.err != CU_FILE_SUCCESS) {
        fprintf(stderr, "Async checkpoint failed: %s\n",
                cuFileGetErrorString(st));
        return -1;
    }
    return 0;
    // Training can continue on other streams — checkpoint writes in background
}
```

**Performance note:** For large models (40GB+), GDS checkpoint save throughput can exceed 25 GB/s, completing in < 2 seconds. CPU-bounce checkpoint would take 4-6 seconds.

## Pattern 3: Multi-GPU Multi-NVMe Striping

**Use case:** Maximum I/O throughput by pairing each GPU with a dedicated NVMe device.

```c
typedef struct {
    int gpu_id;
    CUcontext ctx;
    CUfileDescr_t fh;
    CUdeviceptr buffer;
    size_t buffer_size;
    pthread_t thread;
} GPUStorageNode;

void* gpu_stripe_worker(void *arg) {
    GPUStorageNode *node = (GPUStorageNode*)arg;
    cuCtxSetCurrent(node->ctx);

    // Each thread reads its portion of the data file
    size_t chunk_size = node->buffer_size;
    off_t base_offset = node->gpu_id * node->buffer_size * num_chunks;

    for (size_t i = 0; i < num_chunks; i++) {
        cuFileRead(node->fh, node->buffer, chunk_size,
                   base_offset + i * chunk_size, 0);
        // Process on this GPU
        process_kernel<<<grid, block>>>(node->buffer, chunk_size);
    }

    cudaDeviceSynchronize();
    return NULL;
}

int multi_gpu_striping(const char *file_pattern, int num_gpus,
                       size_t total_size_per_gpu) {
    GPUStorageNode *nodes = calloc(num_gpus, sizeof(GPUStorageNode));

    for (int g = 0; g < num_gpus; g++) {
        nodes[g].gpu_id = g;
        cuCtxCreate(&nodes[g].ctx, 0, g);
        cuCtxSetCurrent(nodes[g].ctx);

        // Each GPU gets its own buffer
        cuMemAlloc(&nodes[g].buffer, total_size_per_gpu);
        cuFileBufRegister(nodes[g].buffer, total_size_per_gpu, 0);

        // Each GPU reads from a dedicated file (or offset in shared file)
        char path[256];
        snprintf(path, sizeof(path), file_pattern, g);
        int fd = open(path, O_DIRECT | O_RDONLY);
        nodes[g].fh.cookie = (CUfileDriverCookie)(uintptr_t)fd;
        cuFileHandleRegister(&nodes[g].fh, NULL);

        nodes[g].buffer_size = total_size_per_gpu / num_chunks;
    }

    // Launch worker threads
    for (int g = 0; g < num_gpus; g++) {
        pthread_create(&nodes[g].thread, NULL, gpu_stripe_worker, &nodes[g]);
    }

    // Wait for all
    for (int g = 0; g < num_gpus; g++) {
        pthread_join(nodes[g].thread, NULL);
    }

    // Cleanup...
    return 0;
}
```

**Requirement:** Each GPU-NVMe pair must be on the same PCIe root complex. Verify with `gdscheck -p`.

## Pattern 4: Streaming Ingest with In-Place Processing

**Use case:** Data flows NVMe → GPU → processed → NVMe without ever touching CPU memory.

```c
void streaming_pipeline(CUfileDescr_t in_fh, CUfileDescr_t out_fh,
                        CUdeviceptr buffer, size_t chunk_size,
                        size_t total_size, cudaStream_t stream) {
    size_t num_chunks = total_size / chunk_size;

    for (size_t i = 0; i < num_chunks; i++) {
        // 1. Read raw data into GPU memory
        cuFileRead(in_fh, buffer, chunk_size, i * chunk_size, 0);

        // 2. Process in-place (data already on GPU)
        preprocess_kernel<<<grid, block, 0, stream>>>(buffer, chunk_size);
        cudaStreamSynchronize(stream);

        // 3. Write results (still on GPU)
        cuFileWrite(out_fh, buffer, chunk_size, i * chunk_size, 0);
    }
}
```

**This eliminates:**

- `cudaMemcpy` host→device (data arrives on GPU via DMA)
- `cudaMemcpy` device→host (results stay on GPU)
- CPU memory allocation for staging buffers
- CPU cycles for data transformation

## Pattern 5: Graceful GDS Degradation

**Use case:** Application that prefers GDS but operates correctly in compat mode.

```c
typedef enum { IO_MODE_GDS, IO_MODE_COMPAT, IO_MODE_POSIX } IOMode;

typedef struct {
    IOMode mode;
    CUfileDescr_t cu_fh;    // Only valid for GDS/COMPAT
    int posix_fd;           // Only valid for POSIX mode
    void *cpu_buffer;       // Only valid for COMPAT/POSIX
} AdaptiveFileHandle;

ssize_t adaptive_read(AdaptiveFileHandle *af, CUdeviceptr devPtr,
                      size_t size, off_t offset) {
    switch (af->mode) {
    case IO_MODE_GDS:
        // Direct DMA: NVMe → GPU
        return cuFileRead(af->cu_fh, devPtr, size, offset, 0);

    case IO_MODE_COMPAT:
        // cuFile compat: NVMe → CPU → GPU (but still cuFile API)
        return cuFileRead(af->cu_fh, devPtr, size, offset, 0);

    case IO_MODE_POSIX:
        // Fallback: NVMe → CPU buffer → cudaMemcpy → GPU
        ssize_t n = pread(af->posix_fd, af->cpu_buffer, size, offset);
        if (n > 0) {
            cudaMemcpy((void*)devPtr, af->cpu_buffer, n,
                       cudaMemcpyHostToDevice);
        }
        return n;
    }
    return -1;
}
```

## Memory Layout Considerations

### Optimal Buffer Layout for GDS

```text
GOOD: Contiguous, page-aligned GPU allocation
┌────────────────────────────────────────────┐
│████████████████████████████████████████████│ ← Single cuMemAlloc + cuFileBufRegister
└────────────────────────────────────────────┘

BAD: Fragmented allocation (multiple small cudaMalloc calls)
┌────┬────┬────┬────┬────┬────┬────┬────┐
│ B0 │ B1 │ B2 │ B3 │ B4 │ B5 │ B6 │ B7 │ ← Each needs separate register
└────┴────┴────┴────┴────┴────┴────┴────┘

FIX: Single large allocation with offsets
┌────────────────────────────────────────────┐
│ B0 │ B1 │ B2 │ B3 │ B4 │ B5 │ B6 │ B7 │ ← One register, offset-based access
└────────────────────────────────────────────┘
```

## Throughput Expectations by Pattern

| Pattern                     | Throughput vs. Theoretical | CPU Usage   | Latency      |
| --------------------------- | -------------------------- | ----------- | ------------ |
| Sync single-file read       | 70-85% of PCIe BW          | Very low    | Per-IO       |
| Double-buffer prefetch      | 85-95% of PCIe BW          | Very low    | Hidden       |
| Multi-GPU striping (4 GPUs) | 3-4× single-GPU            | Very low    | Per-stripe   |
| Checkpoint save (async)     | 80-90% of PCIe BW          | Very low    | Non-blocking |
| POSIX fallback              | 20-40% of PCIe BW          | Medium-High | Per-IO       |

See also:

- `async-io.md` for double-buffering implementation details
- `batch-io.md` for batching many small I/Os in pipeline stages
- `performance-tuning.md` for benchmarking these patterns
