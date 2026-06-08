# cuFile Error Handling

## Overview

cuFile has a two-tier error model: most functions return `CUfileError_t` (check `.err`), while `cuFileRead`/`cuFileWrite` return `ssize_t` (check for -1). Understanding both tiers and knowing which errors are recoverable vs fatal is critical for production applications.

## Error Model

### Tier 1: CUfileError_t (Most Functions)

```c
CUfileError_t status = some_cufile_function(...);
if (status.err != CU_FILE_SUCCESS) {
    const char *msg = cuFileGetErrorString(status);
    fprintf(stderr, "Error: %s (code=%d)\n", msg, status.err);
}
```

Functions returning `CUfileError_t`:

- `cuFileDriverOpen/Close`
- `cuFileDriverGetProperties`
- `cuFileBufRegister/Deregister`
- `cuFileHandleRegister/Deregister`
- `cuFileReadAsync/WriteAsync`
- `cuFileBatchIOSetUp/Submit/GetStatus/Cancel/Destroy`
- `cuFileStreamRegister`

### Tier 2: ssize_t (cuFileRead/Write)

```c
ssize_t bytes = cuFileRead(fh, devPtr, size, offset, 0);
if (bytes < 0) {
    // Error — check errno or use cuFileOpError()
    perror("cuFileRead");
}
```

## Complete Error Code Reference

### Critical / Fatal Errors

These errors typically require application restart or configuration change:

| Error Code                       | Value | Meaning                          | Recovery                                    |
| -------------------------------- | ----- | -------------------------------- | ------------------------------------------- |
| `CU_FILE_INTERNAL_ERROR`         | 14    | Internal driver error            | Check dmesg, reinitialize driver            |
| `CU_FILE_DRIVER_NOT_INITIALIZED` | 26    | `cuFileDriverOpen()` not called  | Call `cuFileDriverOpen()` first             |
| `CU_FILE_DRIVER_SHUTDOWN`        | 27    | Driver is shutting down          | Wait for shutdown, re-initialize            |
| `CU_FILE_MEMORY_ALLOCATION`      | 23    | GPU or system memory exhausted   | Free memory, reduce buffer sizes            |
| `CU_FILE_NOT_SUPPORTED`          | 18    | Operation not supported (no GDS) | Enable compat mode or fix GDS prerequisites |

### Recoverable Errors

These errors can often be resolved without restart:

| Error Code                  | Value | Meaning                             | Recovery                                |
| --------------------------- | ----- | ----------------------------------- | --------------------------------------- |
| `CU_FILE_NOT_FOUND`         | 2     | File not found                      | Check path, create file if needed       |
| `CU_FILE_INVALID_HANDLE`    | 5     | Handle is invalid/stale             | Re-register file handle                 |
| `CU_FILE_INVALID_PARAMETER` | 6     | Bad parameter (NULL, misalignment)  | Fix parameter, retry                    |
| `CU_FILE_INVALID_STREAM`    | 8     | Invalid CUDA stream                 | Check stream creation, re-register      |
| `CU_FILE_DISK_FULL`         | 12    | No space on device                  | Free up space, retry                    |
| `CU_FILE_NOT_REGISTERED`    | 15    | Buffer not registered               | Call `cuFileBufRegister()` first        |
| `CU_FILE_NOT_OPENED`        | 16    | File not opened with required flags | Re-open with O_DIRECT                   |
| `CU_FILE_IO_ERROR`          | 19    | I/O error                           | Retry (if transient), check NVMe health |
| `CU_FILE_TIMEOUT`           | 24    | Operation timed out                 | Retry, increase timeout                 |
| `CU_FILE_CANCELED`          | 25    | Batch operation cancelled           | Re-submit cancelled I/Os                |
| `CU_FILE_BATCH_ERROR`       | 28    | Batch I/O error                     | Check individual I/O statuses           |
| `CU_FILE_STREAM_ERROR`      | 29    | CUDA stream error                   | Check stream state, recreate if needed  |

### Alignment Errors

| Error Code                  | Value | Meaning                            | Recovery                   |
| --------------------------- | ----- | ---------------------------------- | -------------------------- |
| `CU_FILE_INVALID_OFFSET`    | 20    | File offset invalid/misaligned     | Align offset to 4KB        |
| `CU_FILE_INVALID_SIZE`      | 21    | I/O size invalid/misaligned        | Round size to 4KB multiple |
| `CU_FILE_INVALID_ALIGNMENT` | 22    | Buffer or offset alignment invalid | Align buffer to 4KB        |

## Error Classification

```c
typedef enum {
    ERROR_FATAL,       // Restart required
    ERROR_RECOVERABLE, // Retry or fix parameter
    ERROR_CONFIG,      // Configuration issue
    ERROR_TRANSIENT,   // Likely to succeed on retry
} ErrorClass;

ErrorClass classify_cufile_error(CUfileOpError err) {
    switch (err) {
    case CU_FILE_INTERNAL_ERROR:
    case CU_FILE_DRIVER_NOT_INITIALIZED:
    case CU_FILE_DRIVER_SHUTDOWN:
    case CU_FILE_MEMORY_ALLOCATION:
        return ERROR_FATAL;

    case CU_FILE_NOT_FOUND:
    case CU_FILE_INVALID_HANDLE:
    case CU_FILE_INVALID_PARAMETER:
    case CU_FILE_NOT_REGISTERED:
    case CU_FILE_NOT_OPENED:
    case CU_FILE_INVALID_OFFSET:
    case CU_FILE_INVALID_SIZE:
    case CU_FILE_INVALID_ALIGNMENT:
        return ERROR_RECOVERABLE;

    case CU_FILE_NOT_SUPPORTED:
        return ERROR_CONFIG;

    case CU_FILE_DISK_FULL:
    case CU_FILE_IO_ERROR:
    case CU_FILE_TIMEOUT:
    case CU_FILE_CANCELED:
    case CU_FILE_STREAM_ERROR:
    case CU_FILE_BATCH_ERROR:
        return ERROR_TRANSIENT;

    default:
        return ERROR_FATAL;
    }
}
```

## Error Handling Patterns

### Pattern 1: Retry with Backoff

```c
#define MAX_RETRIES  3
#define RETRY_DELAY_US 100

ssize_t cuFileReadWithRetry(CUfileDescr_t fh, CUdeviceptr devPtr,
                            size_t size, off_t offset, int max_retries) {
    for (int attempt = 0; attempt < max_retries; attempt++) {
        ssize_t n = cuFileRead(fh, devPtr, size, offset, 0);
        if (n >= 0) return n;  // Success or EOF

        // Classify error
        CUfileOpError err = cuFileOpError((CUfileError_t){.err = errno});

        switch (classify_cufile_error(err)) {
        case ERROR_TRANSIENT:
            usleep(RETRY_DELAY_US * (1 << attempt));  // Exponential backoff
            continue;

        case ERROR_RECOVERABLE:
            fprintf(stderr, "Non-retryable error: %d\n", err);
            return -1;

        case ERROR_FATAL:
        case ERROR_CONFIG:
            fprintf(stderr, "Fatal error: %d\n", err);
            exit(EXIT_FAILURE);
        }
    }
    fprintf(stderr, "Max retries (%d) exceeded\n", max_retries);
    return -1;
}
```

### Pattern 2: Log and Continue (Batch I/O)

```c
for (int i = 0; i < num_completed; i++) {
    if (io_status[i].err != CU_FILE_SUCCESS) {
        size_t chunk_id = (size_t)io_status[i].cookie;
        ErrorClass cls = classify_cufile_error(io_status[i].err);

        if (cls == ERROR_FATAL) {
            fprintf(stderr, "FATAL: Chunk %zu failed: %d\n",
                    chunk_id, io_status[i].err);
            exit(EXIT_FAILURE);
        }

        if (cls == ERROR_TRANSIENT) {
            // Re-submit this specific I/O
            fprintf(stderr, "WARN: Chunk %zu transient error, retrying\n",
                    chunk_id);
            retry_queue[retry_count++] = chunk_id;
        }

        failed_count++;
    }
}
```

### Pattern 3: GDS Status Monitor

```c
// Periodic health check during long-running I/O
void gds_health_check(int interval_seconds) {
    while (running) {
        sleep(interval_seconds);

        CUfileDrvProps_t props;
        if (cuFileDriverGetProperties(&props).err == CU_FILE_SUCCESS) {
            if (!props.is_gds_enabled && props.is_gds_capable) {
                fprintf(stderr,
                        "WARNING: GDS disabled mid-run! "
                        "Performance degraded.\n");
            }
        }
    }
}
```

## Diagnostic Utilities

### Full Error Context Dump

```c
void dump_cufile_context() {
    // Driver state
    CUfileDrvProps_t props = {0};
    CUfileError_t st = cuFileDriverGetProperties(&props);
    printf("Driver: v%d.%d, GDS: %s/%s\n",
           props.major, props.minor,
           props.is_gds_capable ? "capable" : "not capable",
           props.is_gds_enabled ? "enabled" : "not enabled");

    // CUDA state
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp dev_props;
    cudaGetDeviceProperties(&dev_props, device);
    printf("GPU: %s, CC %d.%d, PCIe Gen%d x%d\n",
           dev_props.name,
           dev_props.major, dev_props.minor,
           dev_props.pciBusID, dev_props.pciDeviceID);
}
```

### Error-to-JSON for Monitoring

```c
void log_error_json(const char *operation, CUfileOpError err,
                    ssize_t bytes, double elapsed_ms) {
    printf("{\"event\":\"cufile_error\","
           "\"op\":\"%s\","
           "\"err_code\":%d,"
           "\"err_msg\":\"%s\","
           "\"bytes\":%zd,"
           "\"elapsed_ms\":%.2f}\n",
           operation, err,
           cuFileGetErrorString((CUfileError_t){.err = err}),
           bytes, elapsed_ms);
}
```

## Common Error Scenarios and Resolutions

| Scenario                                     | Error                       | Resolution Steps                                                      |
| -------------------------------------------- | --------------------------- | --------------------------------------------------------------------- |
| First `cuFileRead` after setup               | `CU_FILE_NOT_REGISTERED`    | Call `cuFileBufRegister` before I/O                                   |
| After `cudaFree` + re-`cudaMalloc`           | `CU_FILE_INVALID_HANDLE`    | Old buffer deregistered; re-register new buffer                       |
| File moved/renamed between register and read | `CU_FILE_NOT_FOUND`         | Deregister, re-open, re-register                                      |
| GDS was working, now throughput halved       | (no error, just slow)       | GDS disabled mid-run. Check `cuFileDriverGetProperties`               |
| Random errors on multi-threaded app          | `CU_FILE_INVALID_HANDLE`    | Multiple threads using same handle. Add locking or per-thread handles |
| All operations failing after sleep/resume    | `CU_FILE_INTERNAL_ERROR`    | System sleep may have invalidated GPU state. Re-initialize            |
| Large buffer registration fails              | `CU_FILE_MEMORY_ALLOCATION` | Check GPU free memory, check `max_device_pinned_mem_size`             |

See also:

- `api-reference.md` for function signatures and error codes
- `driver-lifecycle.md` for driver init errors
- `configuration.md` for `enable_compat_mode` interaction
