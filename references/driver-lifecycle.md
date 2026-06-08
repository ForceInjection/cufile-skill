# cuFile Driver Lifecycle

## Overview

The cuFile driver (`nvidia-fs.ko`) is a kernel module that enables GPUDirect Storage. The userspace library (`libcufile.so`) communicates with this driver to set up DMA mappings and submit I/O operations. Getting the driver lifecycle right is the first step to any cuFile application.

## Driver Initialization: The Complete Pattern

```c
#include <cufile.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

int init_cufile() {
    CUfileError_t status;

    // Step 1: Open the cuFile driver
    status = cuFileDriverOpen();
    if (status.err != CU_FILE_SUCCESS) {
        fprintf(stderr, "cuFileDriverOpen failed: %s\n",
                cuFileGetErrorString(status));
        return -1;
    }
    printf("cuFile driver opened.\n");

    // Step 2: Query driver properties (ALWAYS do this)
    CUfileDrvProps_t props = {0};
    status = cuFileDriverGetProperties(&props);
    if (status.err != CU_FILE_SUCCESS) {
        fprintf(stderr, "cuFileDriverGetProperties failed: %s\n",
                cuFileGetErrorString(status));
        cuFileDriverClose();
        return -1;
    }

    printf("Driver version: v%d.%d\n", props.major, props.minor);
    printf("GDS capable: %s\n", props.is_gds_capable ? "YES" : "NO");
    printf("GDS enabled:  %s\n", props.is_gds_enabled ? "YES" : "NO");
    printf("Max direct IO: %lu bytes (%.1f GB)\n",
           props.max_device_direct_io_size,
           props.max_device_direct_io_size / 1e9);
    printf("Max device cache: %lu bytes (%.1f MB)\n",
           props.max_device_cache_size,
           props.max_device_cache_size / 1e6);

    // Step 3: Interpret GDS status
    if (!props.is_gds_capable) {
        fprintf(stderr,
                "\n*** WARNING: GDS is NOT capable on this platform. ***\n"
                "    All I/O will use compatibility mode (CPU bounce buffer).\n"
                "    Run 'gdscheck -p' to diagnose.\n\n");
        // Application can still proceed — cuFile works in compat mode
    } else if (!props.is_gds_enabled) {
        fprintf(stderr,
                "\n*** WARNING: GDS is capable but NOT enabled. ***\n"
                "    Check /etc/cufile.json and nvidia-fs kernel module.\n"
                "    Run 'gdscheck' to diagnose.\n\n");
    }

    return 0;
}
```

## Driver Shutdown: The Complete Pattern

```c
void shutdown_cufile() {
    // Important: All handles must be deregistered and files closed
    // BEFORE calling cuFileDriverClose(). All buffers must be
    // deregistered and freed BEFORE calling cuFileDriverClose().

    CUfileError_t status = cuFileDriverClose();
    if (status.err != CU_FILE_SUCCESS) {
        fprintf(stderr, "cuFileDriverClose failed: %s\n",
                cuFileGetErrorString(status));
        // Don't exit here — the driver may recover on its own
    }
    printf("cuFile driver closed.\n");
}
```

## Full Application Lifecycle Pattern

```c
int main() {
    // 1. Initialize CUDA context (cuFile needs a CUDA context)
    CUcontext ctx;
    cuInit(0);
    cuCtxCreate(&ctx, 0, 0);

    // 2. Initialize cuFile driver
    if (init_cufile() != 0) {
        return EXIT_FAILURE;
    }

    // 3. Allocate and register GPU buffers
    CUdeviceptr devPtr;
    size_t buf_size = 128 * 1024 * 1024; // 128MB
    cuMemAlloc(&devPtr, buf_size);        // Use cuMemAlloc for guaranteed alignment
    cuFileBufRegister(devPtr, buf_size, 0);

    // 4. Open and register file
    int fd = open("/mnt/nvme/data.bin", O_DIRECT | O_RDWR);
    CUfileDescr_t fh = {0};
    fh.cookie = (CUfileDriverCookie)(uintptr_t)fd;
    cuFileHandleRegister(&fh, NULL);

    // 5. Perform I/O ...
    ssize_t bytes = cuFileRead(fh, devPtr, buf_size, 0, 0);
    printf("Read %zd bytes\n", bytes);

    // 6. Cleanup — REVERSE ORDER OF SETUP
    cuFileHandleDeregister(fh);
    close(fd);
    cuFileBufDeregister(devPtr);
    cuMemFree(devPtr);

    // 7. Shutdown cuFile driver
    shutdown_cufile();

    // 8. Destroy CUDA context
    cuCtxDestroy(ctx);

    return EXIT_SUCCESS;
}
```

## Multi-Threaded Driver Usage

The cuFile driver supports multi-threaded access AFTER initialization:

```c
// CORRECT: Single-threaded init, multi-threaded usage
void main_thread() {
    cuFileDriverOpen();  // Only thread calling this
    // ... register buffers and handles ...
    // Spawn worker threads
}

void worker_thread() {
    // OK: cuFileRead/Write can be called from any thread
    cuFileRead(fh, devPtr, size, offset, 0);
}
```

**Thread safety rules:**

- `cuFileDriverOpen()` / `cuFileDriverClose()` — NOT thread-safe. Call from single thread.
- `cuFileBufRegister()` / `cuFileBufDeregister()` — NOT thread-safe per buffer. Call from single thread per buffer.
- `cuFileHandleRegister()` / `cuFileHandleDeregister()` — NOT thread-safe per handle.
- `cuFileRead()` / `cuFileWrite()` — Thread-safe for DIFFERENT handles. A single handle should only be used by one thread at a time (no internal locking).
- `cuFileReadAsync()` / `cuFileWriteAsync()` — Thread-safe for different handles and streams. Operations on the same stream are ordered in FIFO.

## Version Negotiation

cuFile has an implicit version negotiation. Calling `cuFileDriverOpen()` links against whatever `libcufile.so` is installed. Use `cuFileDriverGetProperties()` to check the negotiated version:

```c
CUfileDrvProps_t props;
cuFileDriverGetProperties(&props);

if (props.major < 2 && props.minor < 6) {
    fprintf(stderr, "cuFile driver too old: %d.%d — need >= 1.6\n",
            props.major, props.minor);
}
```

## Relationship with CUDA Context

cuFile requires an active CUDA context on the calling thread. All operations (except `cuFileDriverOpen/Close`) must be called within a CUDA context.

**Implicit context (CUDA Runtime API):** If you use `cudaMalloc` / `cudaFree`, the CUDA runtime implicitly creates a primary context.

**Explicit context (CUDA Driver API):** If you use `cuMemAlloc` / `cuMemFree`, you must manage the context explicitly.

**Multi-GPU setup:** Call `cuCtxSetCurrent(ctx)` to switch the active GPU before registering buffers on that GPU.

```c
// Multi-GPU buffer registration
for (int gpu = 0; gpu < num_gpus; gpu++) {
    cuCtxSetCurrent(ctx[gpu]);
    cuMemAlloc(&devPtr[gpu], buf_size);
    cuFileBufRegister(devPtr[gpu], buf_size, 0);
}
```

## Common Driver Lifecycle Errors

| Error                                       | Cause                                                 | Fix                                                         |
| ------------------------------------------- | ----------------------------------------------------- | ----------------------------------------------------------- |
| `CU_FILE_DRIVER_NOT_INITIALIZED`            | `cuFileDriverOpen()` not called                       | Call `cuFileDriverOpen()` first                             |
| `CU_FILE_INTERNAL_ERROR` on driver open     | `nvidia-fs.ko` not loaded                             | `modprobe nvidia-fs`                                        |
| `CU_FILE_NOT_SUPPORTED` on driver open      | GPU driver too old (< 470.57)                         | Upgrade NVIDIA driver to 470.57+ (R470) or 525+ recommended |
| `cuFileDriverGetProperties` returns `false` | Platform doesn't support GDS (no ACS, wrong topology) | Check `gdscheck -p` output                                  |
| Driver close hangs                          | Outstanding async I/O not completed                   | `cudaStreamSynchronize()` all streams before closing        |
| Crash on driver close                       | Buffer freed before deregistering                     | Always deregister before freeing — check cleanup order      |

## Diagnostic Commands

```bash
# Check if nvidia-fs kernel module is loaded
lsmod | grep nvidia_fs

# Load it manually
sudo modprobe nvidia-fs

# Check GDS status in procfs
cat /proc/driver/nvidia-fs/gds

# Check GDS versions
strings /proc/driver/nvidia-fs/version

# Verify libcufile is installed (bundled with CUDA)
find /usr/local/cuda -name "libcufile*"
```

See also:

- `api-reference.md` for function signatures
- `configuration.md` for cufile.json tuning that affects driver behavior
- `error-handling.md` for complete error reference
