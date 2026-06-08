# cuFile API Reference

## Complete Function Signatures, Structures, and Enumerations

This reference covers the cuFile API as of CUDA 12.x. Always verify against your installed version: check `/usr/local/cuda/include/cufile.h` or the NVIDIA GDS documentation.

## Core Types

### CUfileError_t — Function Return Type

```c
typedef struct CUfileError_s {
    CUfileOpError err;    // Error code (0 = success)
} CUfileError_t;
```

Every cuFile function returns `CUfileError_t`. Check `.err` for success/failure. Use `cuFileGetErrorString()` or `cuFileOpError()` for diagnostics.

### CUfileOpError — Error Enumeration

```c
typedef enum {
    CU_FILE_SUCCESS                = 0,   // Operation completed successfully
    CU_FILE_NOT_FOUND              = 2,   // File or path not found
    CU_FILE_INVALID_HANDLE         = 5,   // Handle is invalid, closed, or stale
    CU_FILE_INVALID_PARAMETER      = 6,   // Invalid parameter (NULL ptr, bad alignment)
    CU_FILE_INVALID_STREAM         = 8,   // Invalid CUDA stream
    CU_FILE_DISK_FULL              = 12,  // No space left on device
    CU_FILE_INTERNAL_ERROR         = 14,  // Internal driver error
    CU_FILE_NOT_REGISTERED         = 15,  // Buffer not registered with cuFile
    CU_FILE_NOT_OPENED             = 16,  // File not opened with required flags
    CU_FILE_NOT_SUPPORTED          = 18,  // Operation not supported (GDS not available)
    CU_FILE_IO_ERROR               = 19,  // I/O error during operation
    CU_FILE_INVALID_OFFSET         = 20,  // File offset is invalid or misaligned
    CU_FILE_INVALID_SIZE           = 21,  // I/O size is invalid or misaligned
    CU_FILE_INVALID_ALIGNMENT      = 22,  // Buffer or offset alignment invalid
    CU_FILE_MEMORY_ALLOCATION      = 23,  // Memory allocation failed
    CU_FILE_TIMEOUT                = 24,  // Operation timed out
    CU_FILE_CANCELED               = 25,  // Batch operation was canceled
    CU_FILE_DRIVER_NOT_INITIALIZED = 26,  // cuFileDriverOpen() not called
    CU_FILE_DRIVER_SHUTDOWN        = 27,  // Driver is shutting down
    CU_FILE_BATCH_ERROR            = 28,  // Batch I/O error
    CU_FILE_STREAM_ERROR           = 29,  // CUDA stream error
} CUfileOpError;
```

### CUfileDescr_t — File Handle Descriptor

```c
typedef struct CUfileDescr_s {
    CUfileHandle handle;     // Opaque file handle (returned by cuFileHandleRegister)
    CUfileDriverCookie cookie; // Opaque driver cookie
} CUfileDescr_t;
```

### CUfileDrvProps_t — Driver Properties

```c
typedef struct CUfileDrvProps_s {
    uint32_t major;                      // Driver major version
    uint32_t minor;                      // Driver minor version
    bool is_gds_capable;                 // Platform supports GDS
    bool is_gds_enabled;                 // GDS is currently enabled
    uint64_t max_device_pinned_mem_size; // Max pinned GPU memory (bytes)
    uint64_t max_device_direct_io_size;  // Max single GDS IO size (bytes)
    uint64_t max_device_cache_size;      // Max GDS device cache (bytes)
} CUfileDrvProps_t;
```

### CUfileIOParams_t — Batch I/O Parameters

```c
typedef struct CUfileIOParams_s {
    CUfileHandle fh;       // Registered file handle
    CUdeviceptr devPtr;    // GPU buffer address
    off_t file_offset;     // File offset (must be 4KB-aligned for GDS)
    off_t devPtr_offset;   // Offset within GPU buffer
    ssize_t size;          // I/O size in bytes
    void *cookie;          // User-defined cookie (returned in status)
    int opcode;            // CUFILE_READ or CUFILE_WRITE
} CUfileIOParams_t;
```

### CUfileIOStatus_t — Batch I/O Status

```c
typedef struct CUfileIOStatus_s {
    int index;             // Index of the I/O in the batch
    ssize_t ret;           // Bytes transferred (or negative on error)
    CUfileOpError err;     // Error code (CU_FILE_SUCCESS on success)
    void *cookie;          // User cookie from CUfileIOParams_t
} CUfileIOStatus_t;
```

## Driver & Lifecycle Functions

### cuFileDriverOpen()

```c
CUfileError_t cuFileDriverOpen(void);
```

Initialize the cuFile driver. Must be called exactly once, before any other cuFile function. Not thread-safe during initialization — call from a single thread, then other threads can use cuFile functions.

**Returns:** `CU_FILE_SUCCESS` on success, or error code.

**Common errors:**

- `CU_FILE_INTERNAL_ERROR` — nvidia-fs kernel module not loaded
- `CU_FILE_NOT_SUPPORTED` — CUDA driver version too old

### cuFileDriverClose()

```c
CUfileError_t cuFileDriverClose(void);
```

Shutdown the cuFile driver and release all resources. Must be called after all handles and buffers are deregistered. Not thread-safe during shutdown.

### cuFileDriverGetProperties()

```c
CUfileError_t cuFileDriverGetProperties(CUfileDrvProps_t *props);
```

Query driver version and GDS capabilities. **Always call this after `cuFileDriverOpen()`** to verify GDS status.

**Usage:**

```c
CUfileDrvProps_t props = {0};
cuFileDriverGetProperties(&props);
if (!props.is_gds_enabled) {
    // GDS not active — check prerequisites
}
```

## Buffer Management Functions

### cuFileBufRegister()

```c
CUfileError_t cuFileBufRegister(
    CUdeviceptr devPtr,     // GPU buffer address
    size_t      size,       // Buffer size in bytes
    int         flags       // Registration flags (reserved, must be 0)
);
```

Register a GPU buffer for DMA with the cuFile driver. This pins the GPU pages and creates BAR mappings visible to the PCIe root complex.

**Requirements:**

- `devPtr` must be 4KB-aligned for GDS path
- `size` must be a multiple of 4KB for optimal performance
- `flags` must be 0 (reserved for future use)
- The buffer must remain allocated until `cuFileBufDeregister()` is called

**Performance note:** Registration is expensive (~10-100 µs for large buffers). Register once, use many times. Never register/deregister per-I/O.

### cuFileBufDeregister()

```c
CUfileError_t cuFileBufDeregister(CUdeviceptr devPtr);
```

Unregister a previously registered GPU buffer. Must be called before `cudaFree(devPtr)`. After deregistration, the buffer can be freed or reused normally.

### cuFileBufSetAllocFlags()

```c
CUfileError_t cuFileBufSetAllocFlags(
    CUdeviceptr devPtr,
    int         flags
);
```

Set allocation flags for a registered buffer. Advanced usage — most applications should use `flags=0` with `cuFileBufRegister`.

## File Handle Management Functions

### cuFileHandleRegister()

```c
CUfileError_t cuFileHandleRegister(
    CUfileDescr_t *descr,  // [out] Handle descriptor
    CUfileDrvProps_t *props // [in] File properties
);
```

Register a POSIX file descriptor with the cuFile driver for GDS I/O.

**Requirements:**

- File must be opened with `O_DIRECT` for GDS path
- File must be on a supported filesystem (ext4, xfs, GPFS)
- Caller must have read/write permissions
- Not thread-safe: call from one thread per file handle

**Typical usage:**

```c
int fd = open("/mnt/nvme/data.bin", O_DIRECT | O_RDONLY);
CUfileDescr_t descr = {0};
descr.cookie = (CUfileDriverCookie)fd;  // Pass fd in cookie
cuFileHandleRegister(&descr, NULL);
```

### cuFileHandleDeregister()

```c
CUfileError_t cuFileHandleDeregister(CUfileDescr_t descr);
```

Unregister a file handle. Must be called BEFORE `close(fd)`.

## Synchronous I/O Functions

### cuFileRead()

```c
ssize_t cuFileRead(
    CUfileDescr_t fh,    // Registered file handle
    CUdeviceptr   devPtr, // GPU destination buffer (must be registered)
    size_t        size,   // Number of bytes to read
    off_t         offset, // File offset (must be 4KB-aligned for GDS)
    off_t         devPtr_offset // Offset within GPU buffer (typically 0)
);
```

Read data from a file directly into GPU memory.

**Returns:** Number of bytes actually read (may be less than `size` for partial reads), or -1 on error.

**Requirements:**

- `offset` must be 4KB-aligned for GDS path
- `devPtr` must be registered with `cuFileBufRegister`
- `size` should be a multiple of 4KB for optimal GDS performance
- The file must have been opened with `O_DIRECT | O_RDONLY` (or `O_RDWR`)

### cuFileWrite()

```c
ssize_t cuFileWrite(
    CUfileDescr_t fh,
    CUdeviceptr   devPtr,  // GPU source buffer (must be registered)
    size_t        size,    // Number of bytes to write
    off_t         offset,  // File offset (must be 4KB-aligned for GDS)
    off_t         devPtr_offset // Offset within GPU buffer (typically 0)
);
```

Write data from GPU memory directly to a file.

**Returns:** Number of bytes actually written, or -1 on error.

**Requirements:**

- Same alignment and registration requirements as `cuFileRead`
- File must be opened with `O_DIRECT | O_WRONLY` (or `O_RDWR`)

## Asynchronous I/O Functions

### cuFileReadAsync()

```c
CUfileError_t cuFileReadAsync(
    CUfileDescr_t fh,
    CUdeviceptr   devPtr,
    size_t        size,
    off_t         offset,
    off_t         devPtr_offset,
    CUstream      stream    // CUDA stream for async execution
);
```

Submit an async read on a CUDA stream. The read completes when all preceding operations on the stream complete.

**Requirements:**

- Same alignment and registration requirements as `cuFileRead`
- The stream must be registered via `cuFileStreamRegister()` before first use
- Multiple async operations on the same stream are ordered (FIFO)
- Operations on different streams may execute concurrently

### cuFileWriteAsync()

```c
CUfileError_t cuFileWriteAsync(
    CUfileDescr_t fh,
    CUdeviceptr   devPtr,
    size_t        size,
    off_t         offset,
    off_t         devPtr_offset,
    CUstream      stream
);
```

Submit an async write on a CUDA stream. Same semantics as `cuFileReadAsync`.

### cuFileStreamRegister()

```c
CUfileError_t cuFileStreamRegister(
    CUfileDescr_t fh,
    CUstream      stream
);
```

Associate a CUDA stream with a cuFile handle. Must be called once per stream-handle pair before first async I/O. Not needed for synchronous I/O.

## Batch I/O Functions

### cuFileBatchIOSetUp()

```c
CUfileError_t cuFileBatchIOSetUp(
    CUfileDescr_t fh,
    int           batch_id,    // User-defined batch identifier
    int           max_ios,     // Maximum number of concurrent I/Os in batch
    int           flags        // Batch flags (0 = default)
);
```

Initialize batch I/O for a file handle. Creates internal resources for batched submission.

**Parameters:**

- `batch_id`: Application-defined ID (used to identify this batch in status queries)
- `max_ios`: Maximum number of I/O operations per batch (typical: 16-256)
- `flags`: Reserved, must be 0

### cuFileBatchIOSubmit()

```c
CUfileError_t cuFileBatchIOSubmit(
    int                  batch_id,
    int                  num_ios,          // Number of I/Os in this batch
    CUfileIOParams_t    *io_params,        // Array of I/O parameters
    int                  flags             // CUFILE_BATCH_DIRECT_SUBMIT or 0
);
```

Submit a batch of I/O operations. All operations are submitted to the driver in a single call, reducing per-I/O submission overhead.

**Flags:**

- `0` — Normal submission (driver may coalesce or reorder)
- `CUFILE_BATCH_DIRECT_SUBMIT` — Submit directly to hardware without batch optimization

### cuFileBatchIOGetStatus()

```c
CUfileError_t cuFileBatchIOGetStatus(
    int              batch_id,
    int              min_ios,       // Min I/Os to wait for (0 = return immediately)
    int              max_ios,       // Max I/Os to report
    CUfileIOStatus_t *io_status,    // [out] Array of status structs
    int              *num_completed  // [out] Number actually completed
);
```

Poll or wait for batch I/O completions.

**Behavior:**

- `min_ios = 0`: Return immediately with any completed I/Os
- `min_ios = N`: Block until at least N I/Os complete
- `num_completed`: Actual number of completed I/Os (≤ `max_ios`)

### cuFileBatchIOCancel()

```c
CUfileError_t cuFileBatchIOCancel(int batch_id);
```

Cancel all pending I/Os in a batch. Already-completed I/Os are not affected.

### cuFileBatchIODestroy()

```c
CUfileError_t cuFileBatchIODestroy(int batch_id);
```

Release all resources associated with a batch. Must be called after all I/Os complete or are canceled.

## Error & Diagnostic Functions

### cuFileGetErrorString()

```c
const char* cuFileGetErrorString(CUfileError_t error);
```

Convert a `CUfileError_t` to a human-readable error string. Returns statically allocated string — do not free.

### cuFileOpError()

```c
CUfileOpError cuFileOpError(CUfileError_t error);
```

Extract the `CUfileOpError` enum value from a `CUfileError_t`. Equivalent to `error.err`.

## I/O Operation Constants

```c
#define CUFILE_READ   0
#define CUFILE_WRITE  1

// Batch I/O flags
#define CUFILE_BATCH_DIRECT_SUBMIT  (1 << 0)
```

## Size Constants

```c
#define CU_FILE_MIN_IO_SIZE       4096        // Minimum IO size for GDS (4KB)
#define CU_FILE_DEFAULT_IO_SIZE   (1 << 20)   // Default IO size (1MB)
#define CU_FILE_MAX_IO_SIZE       (1ULL << 31) // Maximum IO size (2GB)
#define CU_FILE_ALIGNMENT         4096         // Required alignment for GDS (4KB)
```

See also:

- `driver-lifecycle.md` for detailed init/shutdown patterns
- `buffer-management.md` for buffer registration strategies
- `error-handling.md` for complete error reference
