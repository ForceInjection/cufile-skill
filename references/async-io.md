# cuFile Asynchronous I/O

## Overview

Asynchronous I/O (`cuFileReadAsync` / `cuFileWriteAsync`) uses CUDA streams to overlap data movement with compute. The I/O is submitted to a CUDA stream and completes asynchronously — your code continues immediately while the DMA transfer happens in the background.

This is the key to hiding I/O latency behind GPU compute. In a well-tuned pipeline, data is always ready before the kernel needs it, making the I/O effectively "free."

## Prerequisites

1. CUDA streams (`cudaStreamCreate` or `cuStreamCreate`)
2. Stream must be registered with cuFile: `cuFileStreamRegister(fh, stream)`
3. GPU buffers must be registered: `cuFileBufRegister`
4. File handle must be registered: `cuFileHandleRegister`

## Complete Async I/O Pattern

```c
#include <cufile.h>
#include <cuda_runtime.h>

// 1. Create CUDA streams
cudaStream_t io_stream, compute_stream;
cudaStreamCreate(&io_stream);
cudaStreamCreate(&compute_stream);

// 2. Register streams with cuFile (once per stream-handle pair)
cuFileStreamRegister(fh, io_stream);
// compute_stream doesn't need registration if it won't do I/O

// 3. Submit async read
CUfileError_t status = cuFileReadAsync(
    fh,           // Registered file handle
    devPtr,       // GPU buffer (registered)
    size,         // Bytes to read
    offset,       // File offset
    0,            // devPtr offset
    io_stream     // CUDA stream
);
if (status.err != CU_FILE_SUCCESS) {
    fprintf(stderr, "cuFileReadAsync failed: %s\n",
            cuFileGetErrorString(status));
    exit(EXIT_FAILURE);
}

// 4. Submit compute work on a different stream
// Events synchronize: compute starts AFTER read completes
cudaEvent_t read_done;
cudaEventCreate(&read_done);
cudaEventRecord(read_done, io_stream);        // Record when read finishes
cudaStreamWaitEvent(compute_stream, read_done, 0); // Compute waits for read

process_kernel<<<grid, block, 0, compute_stream>>>(devPtr, size);

// 5. Async write results
cudaEvent_t compute_done;
cudaEventCreate(&compute_done);
cudaEventRecord(compute_done, compute_stream);

// Write waits for compute to finish
cudaStreamWaitEvent(io_stream, compute_done, 0);
cuFileWriteAsync(fh, devPtr, result_size, result_offset, 0, io_stream);

// 6. Synchronize (only when results are needed)
cudaStreamSynchronize(io_stream); // Wait for write to complete
```

## Stream Ordering Semantics

Operations on the same CUDA stream are ordered in FIFO sequence:

```c
// These three operations execute in order on io_stream:
cuFileReadAsync(fh, buf_A, size, offset_A, 0, io_stream);   // 1st
cuFileReadAsync(fh, buf_B, size, offset_B, 0, io_stream);   // 2nd (starts after 1st finishes)
cuFileWriteAsync(fh, buf_C, size, offset_C, 0, io_stream);  // 3rd (starts after 2nd finishes)
```

Operations on DIFFERENT streams can execute concurrently:

```c
// These can run in parallel:
cuFileReadAsync(fh, buf_0, size, off_0, 0, stream_0);
cuFileReadAsync(fh, buf_1, size, off_1, 0, stream_1);  // May overlap with stream_0
```

## Double-Buffered Prefetch Pipeline

The most important async I/O pattern. Two GPU buffers, two streams:

```c
void double_buffer_pipeline(CUfileDescr_t fh, CUdeviceptr buf[2],
                            size_t buf_size, size_t total_chunks) {
    cudaStream_t io_stream, compute_stream;
    cudaStreamCreate(&io_stream);
    cudaStreamCreate(&compute_stream);
    cuFileStreamRegister(fh, io_stream);

    cudaEvent_t io_done[2], compute_done[2];
    for (int i = 0; i < 2; i++) {
        cudaEventCreate(&io_done[i]);
        cudaEventCreate(&compute_done[i]);
    }

    // Prefetch first chunk
    cuFileReadAsync(fh, buf[0], buf_size, 0, 0, io_stream);
    cudaEventRecord(io_done[0], io_stream);

    for (size_t chunk = 0; chunk < total_chunks; chunk++) {
        int cur = chunk % 2;       // Current buffer (0 or 1)
        int next = (chunk + 1) % 2; // Next buffer

        // Wait for current chunk's I/O to complete
        cudaStreamWaitEvent(compute_stream, io_done[cur], 0);

        // Process current chunk (on compute stream)
        process_kernel<<<grid, block, 0, compute_stream>>>(buf[cur], buf_size);
        cudaEventRecord(compute_done[cur], compute_stream);

        // Prefetch next chunk (on I/O stream, while compute runs)
        if (chunk + 1 < total_chunks) {
            // Ensure next buffer is free before reading into it
            if (chunk >= 1) {
                cudaStreamWaitEvent(io_stream, compute_done[next], 0);
            }
            cuFileReadAsync(fh, buf[next], buf_size,
                            (chunk + 1) * buf_size, 0, io_stream);
            cudaEventRecord(io_done[next], io_stream);
        }
    }

    // Final synchronization
    cudaStreamSynchronize(compute_stream);
    cudaStreamSynchronize(io_stream);

    // Cleanup
    for (int i = 0; i < 2; i++) {
        cudaEventDestroy(io_done[i]);
        cudaEventDestroy(compute_done[i]);
    }
    cudaStreamDestroy(io_stream);
    cudaStreamDestroy(compute_stream);
}
```

## Multi-Stream I/O for Bandwidth Saturation

For workloads where I/O is the bottleneck (not compute), use multiple I/O streams:

```c
#define NUM_IO_STREAMS 4

cudaStream_t io_streams[NUM_IO_STREAMS];
for (int s = 0; s < NUM_IO_STREAMS; s++) {
    cudaStreamCreate(&io_streams[s]);
    cuFileStreamRegister(fh, io_streams[s]);
}

// Submit reads round-robin across streams
for (size_t i = 0; i < num_chunks; i++) {
    int stream_idx = i % NUM_IO_STREAMS;
    cuFileReadAsync(fh, devPtr + i * chunk_size, chunk_size,
                    i * chunk_size, 0, io_streams[stream_idx]);
}

// Wait for all
for (int s = 0; s < NUM_IO_STREAMS; s++) {
    cudaStreamSynchronize(io_streams[s]);
}
```

**When to use multi-stream I/O:**

- IO-bound workloads (compute time < IO time)
- Multiple NVMe devices (one stream per device)
- Large sequential reads where the NVMe device supports multiple queues

**When NOT to use multi-stream I/O:**

- Single NVMe device with limited queue depth — extra streams add overhead
- Compute-bound workloads — hide I/O behind compute instead
- Random I/O patterns — NVMe command ordering matters

## Async I/O and GPU Kernel Coordination

### Pattern: Wait on stream before kernel launch

```c
// Launch kernel that depends on I/O completion
cudaStreamWaitEvent(kernel_stream, io_done_event, 0);
my_kernel<<<grid, block, 0, kernel_stream>>>(devPtr, size);
```

### Pattern: Use cuFile I/O as a stream dependency

```c
// I/O completes → triggers kernel → triggers next I/O
cuFileReadAsync(fh, buf, size, offset, 0, stream);
my_kernel<<<grid, block, 0, stream>>>(buf, size);  // Implicitly waits for read
cuFileWriteAsync(fh, buf, size, offset, 0, stream); // Implicitly waits for kernel
```

### Pattern: CUDA graph integration

cuFile async ops can be part of CUDA graphs on supported hardware (Hopper+):

```c
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
cuFileReadAsync(fh, buf, size, offset, 0, stream);
my_kernel<<<grid, block, 0, stream>>>(buf, size);
cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
cudaGraphLaunch(instance, stream);
```

## Performance Characteristics

| Pattern                         | Throughput vs Sync | Latency Hiding      | CPU Usage    |
| ------------------------------- | ------------------ | ------------------- | ------------ |
| Single stream async             | Same as sync       | None (no overlap)   | Same as sync |
| Double-buffered (IO + compute)  | = max(IO, compute) | Full overlap        | Low          |
| Multi-stream I/O (no compute)   | Up to 1.3× sync    | None                | Low          |
| Async with no GDS (compat mode) | Same as sync       | Limited (CPU-bound) | High         |

**Key insight:** Async I/O shines most when GDS is enabled. In compat mode, the CPU is still copying data, so true overlap is limited by CPU core availability.

## Common Async I/O Pitfalls

| Pitfall                                       | Consequence                      | Fix                                                      |
| --------------------------------------------- | -------------------------------- | -------------------------------------------------------- |
| Not calling `cuFileStreamRegister`            | `CU_FILE_INVALID_STREAM` error   | Register stream before first async I/O                   |
| Using same buffer for concurrent reads        | Data corruption (race condition) | Use separate buffers for concurrent operations           |
| Forgetting `cudaStreamSynchronize` on cleanup | Driver close hangs               | Sync all streams before `cuFileDriverClose`              |
| Async I/O + `cudaMemcpy` on same buffer       | Data corruption                  | Use events to serialize access                           |
| Stream depth too deep (too many pending IOs)  | NVMe command queue overflow      | Limit in-flight I/Os to NVMe queue depth (typically 256) |
| Compat mode async: expecting true overlap     | I/O effectively synchronous      | Verify GDS is enabled before relying on async overlap    |
| CUDA graph + cuFile on pre-Hopper GPU         | Not supported                    | Use stream-ordered I/O on pre-Hopper GPUs                |

## When to Use Async vs Sync vs Batch

| Scenario                                     | Best Choice               |
| -------------------------------------------- | ------------------------- |
| Simple read/write, no compute overlap needed | Sync I/O                  |
| I/O + GPU compute in same pipeline           | **Async I/O**             |
| Many small I/Os, no compute overlap          | Batch I/O                 |
| Many small I/Os + GPU compute                | Batch + Async             |
| Single large sequential read                 | Sync I/O                  |
| Checkpoint (save model while training)       | **Async I/O**             |
| Low-latency small random reads               | Sync I/O (least overhead) |

See also:

- `sync-io.md` for synchronous I/O patterns
- `batch-io.md` for batch I/O API
- `integration-patterns.md` for end-to-end pipeline design
- `api-reference.md` for function signatures
