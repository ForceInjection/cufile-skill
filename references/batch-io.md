# cuFile Batch I/O

## Overview

Batch I/O (`cuFileBatchIOSetUp` / `cuFileBatchIOSubmit`) amortizes the per-I/O submission overhead by grouping multiple I/O operations into a single submission call. For workloads with many small-to-medium I/Os, batch I/O can improve throughput 2–10× compared to individual `cuFileRead`/`cuFileWrite` calls.

**When batch I/O helps:**

- Many I/Os (dozens to thousands per second)
- Small-to-medium IO sizes (4KB–512KB)
- Random or scattered access patterns

**When batch I/O doesn't help:**

- Few large I/Os (> 1MB each) — per-IO overhead is already negligible
- Single sequential scan — use sync or async I/O
- Already saturating PCIe bandwidth — batching won't increase throughput

## Batch I/O Lifecycle

```text
cuFileBatchIOSetUp(batch_id, max_ios)
    │
    ├──► cuFileBatchIOSubmit(batch_id, num_ios, io_params)
    │       │
    │       ├──► cuFileBatchIOGetStatus(batch_id, min, max, &num_done)
    │       │       └──► Process completed I/Os
    │       │
    │       └──► [Optional] cuFileBatchIOCancel(batch_id)
    │
    └──► cuFileBatchIODestroy(batch_id)
```

## Complete Batch I/O Example

```c
#include <cufile.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define BATCH_SIZE   64
#define IO_SIZE      (256 * 1024)  // 256KB per IO
#define NUM_BATCHES  100

void batch_read_example(CUfileDescr_t fh, CUdeviceptr devPtr,
                        size_t total_size) {
    int batch_id = 42;  // User-defined identifier
    CUfileError_t status;

    // Step 1: Set up batch I/O
    status = cuFileBatchIOSetUp(fh, batch_id, BATCH_SIZE, 0);
    if (status.err != CU_FILE_SUCCESS) {
        fprintf(stderr, "cuFileBatchIOSetUp failed: %s\n",
                cuFileGetErrorString(status));
        exit(EXIT_FAILURE);
    }

    // Step 2: Prepare I/O parameters
    CUfileIOParams_t io_params[BATCH_SIZE];
    CUfileIOStatus_t io_status[BATCH_SIZE];

    size_t num_chunks = total_size / IO_SIZE;
    size_t chunks_processed = 0;

    while (chunks_processed < num_chunks) {
        // Build batch
        int batch_count = 0;
        for (int i = 0; i < BATCH_SIZE && chunks_processed + i < num_chunks; i++) {
            io_params[i] = (CUfileIOParams_t){
                .fh = fh.handle,
                .devPtr = devPtr + (chunks_processed + i) * IO_SIZE,
                .file_offset = (chunks_processed + i) * IO_SIZE,
                .devPtr_offset = 0,
                .size = IO_SIZE,
                .cookie = (void*)(uintptr_t)(chunks_processed + i), // Track chunk ID
                .opcode = CUFILE_READ,
            };
            batch_count++;
        }

        // Step 3: Submit batch
        status = cuFileBatchIOSubmit(batch_id, batch_count, io_params, 0);
        if (status.err != CU_FILE_SUCCESS) {
            fprintf(stderr, "cuFileBatchIOSubmit failed: %s\n",
                    cuFileGetErrorString(status));
            break;
        }

        // Step 4: Wait for completions
        int num_completed = 0;
        status = cuFileBatchIOGetStatus(batch_id,
                                         batch_count,  // Wait for ALL I/Os
                                         batch_count,  // Report all
                                         io_status,
                                         &num_completed);
        if (status.err != CU_FILE_SUCCESS) {
            fprintf(stderr, "cuFileBatchIOGetStatus failed: %s\n",
                    cuFileGetErrorString(status));
            break;
        }

        // Step 5: Check individual I/O results
        for (int i = 0; i < num_completed; i++) {
            if (io_status[i].err != CU_FILE_SUCCESS) {
                size_t chunk_id = (size_t)io_status[i].cookie;
                fprintf(stderr, "Batch I/O [%d] (chunk %zu) failed: %d\n",
                        i, chunk_id, io_status[i].err);
            } else if (io_status[i].ret != IO_SIZE) {
                fprintf(stderr, "Batch I/O [%d] short read: %zd / %zu\n",
                        i, io_status[i].ret, IO_SIZE);
            }
        }

        chunks_processed += batch_count;
    }

    // Step 6: Cleanup
    cuFileBatchIODestroy(batch_id);
}
```

## Batch I/O with Non-Blocking Polling

For maximum throughput, poll for completions while building the next batch:

```c
while (chunks_processed < num_chunks) {
    // Check for completions from previous batch (non-blocking)
    int num_done = 0;
    cuFileBatchIOGetStatus(batch_id, 0, BATCH_SIZE, io_status, &num_done);

    // Process completed I/Os
    for (int i = 0; i < num_done; i++) {
        // Handle completion...
    }

    // Submit next batch while previous is still in-flight
    if (inflight_count < MAX_INFLIGHT) {
        // Build and submit next batch...
    }
}
```

## Batch vs Individual vs Async I/O

| Aspect                | Batch I/O                    | Individual Sync I/O    | Async I/O (Stream)           |
| --------------------- | ---------------------------- | ---------------------- | ---------------------------- |
| Submission overhead   | Amortized across batch       | Per-call               | Per-call (lower on GPU)      |
| Best for IO size      | 4KB – 512KB                  | > 1MB                  | > 1MB (with compute overlap) |
| CPU usage             | Low (single submission call) | Higher (many syscalls) | Low (GPU-driven)             |
| Completion model      | Polling (`GetStatus`)        | Blocking               | Stream synchronization       |
| Overlap with compute? | No (CPU polls completions)   | No                     | Yes (CUDA streams)           |
| Complexity            | Medium                       | Low                    | Medium-High                  |

## Batch Sizing Guidelines

```c
// Optimal batch size depends on IO size:
size_t io_size = 256 * 1024; // 256KB

int batch_size;
if (io_size <= 4096)       batch_size = 128;
else if (io_size <= 65536)  batch_size = 64;
else if (io_size <= 262144) batch_size = 32;
else if (io_size <= 1048576) batch_size = 16;
else                         batch_size = 8;  // Large IOs — less benefit

// Rule of thumb: batch_total_size ≈ 4-8MB
// Adjust based on measurement
```

## Batch I/O Flags

```c
// Default: driver may coalesce or reorder for efficiency
cuFileBatchIOSubmit(batch_id, num_ios, io_params, 0);

// Direct submit: bypass batch optimization, submit directly to hardware
cuFileBatchIOSubmit(batch_id, num_ios, io_params, CUFILE_BATCH_DIRECT_SUBMIT);
```

When to use `CUFILE_BATCH_DIRECT_SUBMIT`:

- Very low latency requirements (skip driver-side batching)
- Debugging batch I/O issues
- Small number of large I/Os

## Error Handling in Batch I/O

Batch I/O has two error levels:

1. **Submission-level errors**: `cuFileBatchIOSubmit` returns an error → entire batch failed
2. **Per-I/O errors**: Individual I/O errors in `CUfileIOStatus_t.err` → only that I/O failed

```c
// Handle both levels
status = cuFileBatchIOSubmit(batch_id, batch_count, io_params, 0);
if (status.err != CU_FILE_SUCCESS) {
    // Entire batch failed — may need to retry or abort
    fprintf(stderr, "Batch submission failed: %s\n", cuFileGetErrorString(status));
} else {
    // Check individual I/Os
    for (int i = 0; i < num_completed; i++) {
        if (io_status[i].err != CU_FILE_SUCCESS) {
            // Individual I/O failed — partial retry possible
            fprintf(stderr, "I/O %d failed: err=%d, bytes=%zd\n",
                    io_status[i].index, io_status[i].err, io_status[i].ret);
        }
    }
}
```

## Batch Cancel

Cancel pending I/Os in a batch:

```c
// Cancel all pending I/Os
cuFileBatchIOCancel(batch_id);

// Wait for cancelled I/Os to report status
int num_done = 0;
CUfileIOStatus_t io_status[BATCH_SIZE];
cuFileBatchIOGetStatus(batch_id, 0, BATCH_SIZE, io_status, &num_done);

// Cancelled I/Os will have err = CU_FILE_CANCELED
```

## Performance Tuning Checklist for Batch I/O

- [ ] **Batch size matches IO size** — Use the sizing table above. Too small: overhead not amortized. Too large: latency spikes.
- [ ] **Batch total size 4-8MB** — Sweet spot for amortizing submission cost without excessive queuing delay.
- [ ] **Non-blocking polling for high throughput** — Overlap batch building with I/O completion processing.
- [ ] **Check GDS status** — Batch I/O without GDS means batches of CPU copies. Verify with `cuFileIsGpuDirectStorageEnabled()`.
- [ ] **Buffer reuse** — All I/Os in a batch should target pre-registered buffers. Registration per-batch-I/O kills performance.
- [ ] **Cookie for tracking** — Use the `.cookie` field to map completions back to application-level contexts.
- [ ] **Monitor inflight depth** — Don't submit more batches than the NVMe device can handle concurrently (typically 256 commands per queue).

## Common Batch I/O Pitfalls

| Pitfall                                    | Consequence                         | Fix                                                  |
| ------------------------------------------ | ----------------------------------- | ---------------------------------------------------- |
| Batch too small (batch_size < 8)           | No measurable throughput gain       | Increase batch size or use sync I/O directly         |
| Batch too large (batch_size > 256)         | High latency, queue overflow        | Reduce batch size, check `MAXCMD` from NVMe Identify |
| Not checking individual I/O status         | Silent data corruption              | Always check `io_status[i].err` and `.ret`           |
| Reusing `CUfileIOParams_t` during inflight | Data race, corrupted I/O parameters | Copy `io_params` if reusing during inflight          |
| Mixing batch and non-batch I/O on same fh  | Undefined ordering                  | Use either batch or individual I/O per handle        |

See also:

- `sync-io.md` for synchronous I/O
- `async-io.md` for stream-based async I/O
- `performance-tuning.md` for end-to-end tuning
- `api-reference.md` for function signatures
