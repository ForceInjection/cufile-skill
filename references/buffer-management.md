# cuFile Buffer Management

## Overview

GPU buffer registration is THE prerequisite for GPUDirect Storage. Without `cuFileBufRegister`, every cuFileRead/Write goes through a CPU bounce buffer — silently. This reference covers registration strategies, alignment requirements, and memory management patterns.

## Why Registration is Necessary

cuFile needs to:

1. **Pin GPU pages** — DMA engines require physically contiguous, non-pageable memory
2. **Create BAR mappings** — Expose GPU physical addresses to the PCIe root complex so the NVMe controller can target them
3. **Set up IOMMU entries** — If IOMMU is enabled, cuFile must create IOMMU mappings for the NVMe device to access GPU memory

Without registration, these steps don't happen → the NVMe controller can't DMA into GPU memory → cuFile falls back to CPU bounce buffer.

## Buffer Registration Basics

```c
#include <cufile.h>
#include <cuda_runtime.h>

// Step 1: Allocate GPU memory with guaranteed alignment
CUdeviceptr devPtr;
size_t buf_size = 128 * 1024 * 1024; // 128 MB

// Prefer cuMemAlloc over cudaMalloc — it provides guaranteed alignment
cuMemAlloc(&devPtr, buf_size);

// Step 2: Register the buffer for GDS
CUfileError_t status = cuFileBufRegister(devPtr, buf_size, 0);
if (status.err != CU_FILE_SUCCESS) {
    fprintf(stderr, "cuFileBufRegister failed: %s (err=%d)\n",
            cuFileGetErrorString(status), status.err);
    // Common causes:
    // - Buffer not 4KB-aligned
    // - Size too small (< 4KB)
    // - GPU memory exhausted
}

// ... perform I/O using devPtr ...

// Step 3: Deregister before freeing
cuFileBufDeregister(devPtr);

// Step 4: Free GPU memory
cuMemFree(devPtr);
```

## Alignment Requirements

| Aspect         | Minimum    | Recommended   | Notes                                              |
| -------------- | ---------- | ------------- | -------------------------------------------------- |
| Buffer address | 4KB (4096) | GPU page size | `cudaMalloc` typically aligns to 256B — NOT enough |
| Buffer size    | 4KB        | 1MB+          | Small buffers have high registration overhead/byte |
| reg offset     | 4KB        | 4KB           | Must match device-side alignment                   |

### Checking Alignment at Runtime

```c
bool is_aligned(CUdeviceptr ptr, size_t alignment) {
    return ((uintptr_t)ptr % alignment) == 0;
}

// Always validate before registering
if (!is_aligned(devPtr, 4096)) {
    fprintf(stderr, "ERROR: GPU buffer NOT 4KB-aligned!\n");
    fprintf(stderr, "  Address: 0x%lx\n", (uintptr_t)devPtr);
    fprintf(stderr, "  Offset from 4K: %lu bytes\n", (uintptr_t)devPtr % 4096);
    exit(EXIT_FAILURE);
}
```

### Guaranteeing Alignment

```c
// Method 1: cuMemAlloc (Driver API) — guaranteed page-aligned
CUdeviceptr devPtr;
cuMemAlloc(&devPtr, size);  // Always GPU-page-aligned

// Method 2: cudaMalloc with manual check
void *raw_ptr;
cudaMalloc(&raw_ptr, size);
if ((uintptr_t)raw_ptr % 4096 != 0) {
    // Handle misalignment — rare but possible
}

// Method 3: Overallocated + offset (extreme case)
void *overalloc;
size_t aligned_size = size + 4096;
cudaMalloc(&overalloc, aligned_size);
uintptr_t aligned_addr = ((uintptr_t)overalloc + 4095) & ~4095;
```

## Registration Strategy: Register Once, Use Many Times

Registration is expensive because it pins GPU pages. This is the single most important performance rule for buffer management.

### Anti-Pattern (DO NOT DO THIS)

```c
// WRONG: Register/deregister per I/O — 90%+ of time wasted in pinning
for (int i = 0; i < 10000; i++) {
    cuFileBufRegister(devPtr, buf_size, 0);
    cuFileRead(fh, devPtr, buf_size, i * buf_size, 0);
    cuFileBufDeregister(devPtr);
}
```

### Correct Pattern

```c
// RIGHT: Register once, reuse for all I/Os
cuFileBufRegister(devPtr, buf_size, 0);
for (int i = 0; i < 10000; i++) {
    cuFileRead(fh, devPtr, buf_size, i * buf_size, 0);
}
cuFileBufDeregister(devPtr);
```

## Large Buffer Registration

For large buffers (> 1GB), registration can take significant time (100ms+). This is expected — the driver is pinning many GPU pages.

### Chunked Registration for Very Large Memory Pools

If you have a multi-GB buffer pool and only access portions at a time:

```c
// For a 16GB pool, register in 1GB chunks
#define CHUNK_SIZE (1ULL << 30) // 1GB

for (size_t offset = 0; offset < total_pool_size; offset += CHUNK_SIZE) {
    CUdeviceptr chunk = base_ptr + offset;
    size_t chunk_sz = min(CHUNK_SIZE, total_pool_size - offset);
    cuFileBufRegister(chunk, chunk_sz, 0);
}
```

**Trade-off:** Chunked registration reduces initial registration time but increases the number of registered regions. Each region consumes driver resources. Balance based on your access pattern.

## Buffer Lifecycle Management

```c
typedef struct {
    CUdeviceptr ptr;
    size_t size;
    bool registered;
} GPUBuffer;

void gpu_buffer_init(GPUBuffer *buf, size_t size) {
    buf->size = size;
    buf->registered = false;
    CUresult res = cuMemAlloc(&buf->ptr, size);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "cuMemAlloc failed: %d\n", res);
        exit(EXIT_FAILURE);
    }
}

void gpu_buffer_register(GPUBuffer *buf) {
    if (buf->registered) return; // Already registered
    CUfileError_t st = cuFileBufRegister(buf->ptr, buf->size, 0);
    if (st.err != CU_FILE_SUCCESS) {
        fprintf(stderr, "cuFileBufRegister failed: %s\n", cuFileGetErrorString(st));
        exit(EXIT_FAILURE);
    }
    buf->registered = true;
}

void gpu_buffer_free(GPUBuffer *buf) {
    if (buf->registered) {
        cuFileBufDeregister(buf->ptr);
        buf->registered = false;
    }
    if (buf->ptr) {
        cuMemFree(buf->ptr);
        buf->ptr = 0;
    }
}
```

## Pinned Memory Limits

The cuFile driver limits total pinned GPU memory. Check with `cuFileDriverGetProperties()`:

```c
CUfileDrvProps_t props;
cuFileDriverGetProperties(&props);
printf("Max pinned memory: %lu bytes (%.1f GB)\n",
       props.max_device_pinned_mem_size,
       props.max_device_pinned_mem_size / 1e9);
```

If you exceed the limit, `cuFileBufRegister` returns `CU_FILE_MEMORY_ALLOCATION`. Either:

1. Increase `max_device_pinned_mem_size` in `/etc/cufile.json`
2. Reduce registered buffer count
3. Deregister unused buffers

## Common Pitfalls

| Pitfall                                    | Consequence                                 | Prevention                              |
| ------------------------------------------ | ------------------------------------------- | --------------------------------------- |
| Using `cudaMalloc` without alignment check | Buffer may be 256B-aligned, not 4KB         | Use `cuMemAlloc` or verify alignment    |
| Registering in hot loop                    | 90%+ time wasted in registration            | Register once at init                   |
| Forgetting to deregister before `cudaFree` | Undefined behavior, possible GPU crash      | Pair every register with deregister     |
| Registering buffers > pinned limit         | `CU_FILE_MEMORY_ALLOCATION` error           | Check `max_device_pinned_mem_size`      |
| Registering on wrong GPU (multi-GPU)       | I/O targets wrong GPU, data corruption      | `cuCtxSetCurrent()` before registration |
| Buffer resize without re-registration      | Partial buffer not registered → compat mode | Re-register after resize                |

See also:

- `api-reference.md` for function signatures
- `file-handle-management.md` for the other half of the registration pair
- `performance-tuning.md` for buffer sizing strategies
