# cuFile Synchronous I/O

## Overview

Synchronous I/O (`cuFileRead` / `cuFileWrite`) is the simplest cuFile pattern. The calling thread blocks until the I/O completes. For high-throughput workloads, synchronous I/O can saturate PCIe bandwidth — but only if alignment, IO size, and thread mapping are correct.

## cuFileRead: Complete Usage

```c
ssize_t cuFileRead(
    CUfileDescr_t fh,       // Registered file handle
    CUdeviceptr   devPtr,   // GPU buffer (registered)
    size_t        size,     // Bytes to read
    off_t         offset,   // File offset (4KB-aligned for GDS)
    off_t         devPtr_offset  // Offset within GPU buffer (0 = start)
);
```

### Return Value Semantics

`cuFileRead` returns `ssize_t`, NOT `CUfileError_t`. This is different from most cuFile functions.

| Return Value | Meaning                                                                 |
| ------------ | ----------------------------------------------------------------------- |
| `> 0`        | Number of bytes actually read (may be less than `size`)                 |
| `0`          | End of file reached                                                     |
| `-1`         | Error — check `errno` or use `cuFileOpError()` for cuFile-specific info |

### Handling Partial Reads

Like POSIX `read()`, cuFileRead can return fewer bytes than requested. Always handle this:

```c
ssize_t cuFileReadFull(CUfileDescr_t fh, CUdeviceptr devPtr,
                       size_t size, off_t offset) {
    size_t total_read = 0;
    while (total_read < size) {
        ssize_t n = cuFileRead(fh,
                               devPtr + total_read,
                               size - total_read,
                               offset + total_read,
                               0);
        if (n == 0) {
            fprintf(stderr, "EOF after %zu bytes (expected %zu)\n",
                    total_read, size);
            break;
        }
        if (n < 0) {
            fprintf(stderr, "cuFileRead error after %zu bytes\n", total_read);
            return -1;
        }
        total_read += n;
    }
    return total_read;
}
```

**Note on GDS and partial reads:** With GDS, partial reads are rare — the NVMe controller DMAs the full requested size or fails entirely. Partial reads are more common in compat mode (CPU page cache interaction).

## cuFileWrite: Complete Usage

```c
ssize_t cuFileWrite(
    CUfileDescr_t fh,
    CUdeviceptr   devPtr,   // GPU buffer (registered)
    size_t        size,     // Bytes to write
    off_t         offset,   // File offset (4KB-aligned for GDS)
    off_t         devPtr_offset
);
```

Same return semantics as `cuFileRead`. Handle partial writes:

```c
ssize_t cuFileWriteFull(CUfileDescr_t fh, CUdeviceptr devPtr,
                        size_t size, off_t offset) {
    size_t total_written = 0;
    while (total_written < size) {
        ssize_t n = cuFileWrite(fh,
                                devPtr + total_written,
                                size - total_written,
                                offset + total_written,
                                0);
        if (n < 0) {
            fprintf(stderr, "cuFileWrite error after %zu bytes\n", total_written);
            return -1;
        }
        total_written += n;
    }
    return total_written;
}
```

## Synchronous I/O with Verify

A common pattern: read data into GPU, verify integrity with a CUDA kernel:

```c
// Read data
ssize_t bytes = cuFileRead(fh, devPtr, expected_size, offset, 0);
if (bytes != expected_size) {
    fprintf(stderr, "Short read: %zd / %zu\n", bytes, expected_size);
    exit(EXIT_FAILURE);
}

// Verify checksum on GPU (no data copy needed!)
verify_checksum_kernel<<<grid, block>>>(devPtr, expected_size, expected_checksum);
cudaDeviceSynchronize();
```

## Performance Characteristics

### GDS Path (GPU Direct DMA)

| Metric               | Typical Value        | Notes                                         |
| -------------------- | -------------------- | --------------------------------------------- |
| Latency (4KB)        | ~10-15 µs            | PCIe round-trip + NVMe command processing     |
| Latency (1MB)        | ~50-100 µs           | DMA setup dominates for large transfers       |
| Throughput (1MB IO)  | 20-28 GB/s (Gen4 x4) | Near PCIe line rate                           |
| Throughput (16MB IO) | 25-30 GB/s (Gen4 x4) | Saturates PCIe bandwidth                      |
| CPU utilization      | < 1% per IO thread   | CPU only submits commands, doesn't touch data |

### Compat Mode (CPU Bounce Buffer)

| Metric               | Typical Value        | Notes                           |
| -------------------- | -------------------- | ------------------------------- |
| Latency (4KB)        | ~20-30 µs            | Two PCIe hops + CPU copy        |
| Throughput (1MB IO)  | 10-15 GB/s           | Limited by CPU memory bandwidth |
| Throughput (16MB IO) | 12-18 GB/s           | Cannot saturate PCIe            |
| CPU utilization      | 10-30% per IO thread | CPU is copying every byte       |

## Optimal IO Size Selection

```c
// Determine optimal IO size based on driver properties
CUfileDrvProps_t props;
cuFileDriverGetProperties(&props);

size_t io_size;
if (props.is_gds_enabled) {
    // GDS path: large IOs are efficient
    io_size = 16 * 1024 * 1024; // 16MB
} else {
    // Compat mode: medium IOs are better (CPU cache effects)
    io_size = 2 * 1024 * 1024; // 2MB
}

// Cap at max_direct_io_size
if (io_size > props.max_device_direct_io_size) {
    io_size = props.max_device_direct_io_size;
}
```

## Thread-to-File Mapping for Multi-File Workloads

```c
typedef struct {
    CUfileDescr_t fh;
    CUdeviceptr devPtr;
    size_t chunk_size;
    size_t total_size;
    int thread_id;
} io_worker_arg_t;

void* io_worker(void *arg) {
    io_worker_arg_t *a = (io_worker_arg_t*)arg;
    size_t num_chunks = a->total_size / a->chunk_size;

    for (size_t i = 0; i < num_chunks; i++) {
        off_t offset = i * a->chunk_size;
        ssize_t bytes = cuFileRead(a->fh, a->devPtr + offset,
                                   a->chunk_size, offset, 0);
        if (bytes != a->chunk_size) {
            fprintf(stderr, "[Thread %d] Short read at chunk %zu\n",
                    a->thread_id, i);
        }
    }
    return NULL;
}

// Spawn one thread per file
pthread_t threads[NUM_FILES];
io_worker_arg_t args[NUM_FILES];
for (int i = 0; i < NUM_FILES; i++) {
    args[i] = (io_worker_arg_t){
        .fh = file_handles[i],
        .devPtr = gpu_buffers[i],
        .chunk_size = 16 * 1024 * 1024,
        .total_size = file_sizes[i],
        .thread_id = i,
    };
    pthread_create(&threads[i], NULL, io_worker, &args[i]);
}
for (int i = 0; i < NUM_FILES; i++) {
    pthread_join(threads[i], NULL);
}
```

## Common Sync I/O Pitfalls

| Pitfall                                     | Consequence                    | Fix                                              |
| ------------------------------------------- | ------------------------------ | ------------------------------------------------ |
| Not handling partial read/write             | Data silently truncated        | Always loop on `cuFileRead`/`cuFileWrite` return |
| Using IO size > `max_device_direct_io_size` | Falls back to compat mode      | Cap IO size at property value                    |
| IO size not 4KB-aligned in GDS path         | Falls back to compat mode      | Round up to 4KB multiple                         |
| File offset not 4KB-aligned                 | `CU_FILE_INVALID_OFFSET`       | Align offset to 4KB                              |
| Multiple threads on same handle             | Undefined behavior, corruption | One thread per handle, or add locking            |
| Calling sync I/O from CUDA stream callback  | Deadlock or timeout            | Use async I/O for stream contexts                |

See also:

- `async-io.md` for streaming I/O patterns
- `batch-io.md` for high-throughput multi-IO submission
- `performance-tuning.md` for complete tuning workflow
- `api-reference.md` for function signatures
