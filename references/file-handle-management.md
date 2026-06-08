# cuFile File Handle Management

## Overview

File handle registration is the second half of the GDS setup pair (buffer registration being the first). It binds a POSIX file descriptor to the cuFile driver, enabling direct DMA between the file's backing storage (NVMe) and registered GPU buffers.

## The O_DIRECT Requirement

**Files MUST be opened with `O_DIRECT` for the GDS path.** Without `O_DIRECT`, the kernel's page cache intervenes, and cuFile silently falls back to compatibility mode (CPU bounce buffer).

```c
// CORRECT: GDS path possible
int fd = open("/mnt/nvme/data.bin", O_DIRECT | O_RDONLY);

// WRONG: Will use compat mode
int fd = open("/mnt/nvme/data.bin", O_RDONLY);

// WRONG: Buffered I/O — compat mode
int fd = open("/mnt/nvme/data.bin", O_RDWR);
```

## Supported Filesystems

| Filesystem | GDS Support | Notes                                                           |
| ---------- | ----------- | --------------------------------------------------------------- |
| ext4       | ✅ Full     | Standard Linux ext4 with default mount options                  |
| xfs        | ✅ Full     | Recommended for large files, better concurrency                 |
| GPFS       | ✅ Full     | IBM Spectrum Scale, enterprise parallel filesystem              |
| NFS v3/v4  | ❌ None     | Network filesystem — no local NVMe                              |
| CIFS/SMB   | ❌ None     | Network filesystem                                              |
| FUSE       | ❌ None     | Userspace filesystem — no kernel DMA support                    |
| tmpfs      | ❌ None     | RAM-backed — no NVMe, use `cudaMemcpy` instead                  |
| ZFS        | ⚠️ Partial  | May work with specific configurations, not officially supported |

**Verification:**

```bash
# Check filesystem type
df -T /mnt/nvme

# Test specific filesystem
gdscheck -f /mnt/nvme
```

## File Handle Registration: The Complete Pattern

```c
#include <fcntl.h>
#include <cufile.h>

int fd = open("/mnt/nvme/data.bin", O_DIRECT | O_RDWR | O_CREAT, 0644);
if (fd < 0) {
    perror("open");
    exit(EXIT_FAILURE);
}

CUfileDescr_t fh = {0};
// The cookie can be used to pass context to cuFile.
// Convention: store the fd in cookie for later reference.
fh.cookie = (CUfileDriverCookie)(uintptr_t)fd;

CUfileError_t status = cuFileHandleRegister(&fh, NULL);
if (status.err != CU_FILE_SUCCESS) {
    fprintf(stderr, "cuFileHandleRegister failed: %s (err=%d)\n",
            cuFileGetErrorString(status), status.err);
    close(fd);
    exit(EXIT_FAILURE);
}

// fh.handle now contains the valid cuFile handle for I/O operations

// ... perform I/O ...

// Cleanup: deregister BEFORE closing the fd
cuFileHandleDeregister(fh);
close(fd);
```

## Handle Registration Parameters

```c
CUfileError_t cuFileHandleRegister(
    CUfileDescr_t *descr,   // [out] Populated handle descriptor
    CUfileDrvProps_t *props  // [in] Optional: override driver properties for this handle
);
```

- `descr`: Must be zero-initialized. `.cookie` can be set to any user-defined value.
- `props`: Usually `NULL` (use global driver properties). Non-`NULL` to override per-handle.

## File Open Flags: What Works

| Flag Combination                   | GDS Support | Use Case                                |
| ---------------------------------- | ----------- | --------------------------------------- |
| `O_DIRECT \| O_RDONLY`             | ✅          | Read-only data ingestion                |
| `O_DIRECT \| O_WRONLY`             | ✅          | Write-only (checkpoint, output)         |
| `O_DIRECT \| O_RDWR`               | ✅          | Read + write on same file               |
| `O_DIRECT \| O_RDWR \| O_CREAT`    | ✅          | Create or open for read/write           |
| `O_DIRECT \| O_RDWR \| O_SYNC`     | ⚠️          | Synchronous writes — higher latency     |
| `O_DIRECT \| O_RDONLY \| O_DIRECT` | ✅          | Same as above (O_DIRECT once is enough) |

**Avoid `O_SYNC`** unless you need write-through-to-media semantics. It adds per-write latency and reduces throughput significantly.

## Persisting Files for GDS

For newly created files, you may need to pre-allocate and set size:

```c
// Create and extend file to desired size
int fd = open("/mnt/nvme/data.bin", O_DIRECT | O_RDWR | O_CREAT, 0644);
ftruncate(fd, 10ULL * 1024 * 1024 * 1024); // 10 GB file
// Now register with cuFile
```

Without `ftruncate`, writing beyond EOF (current file size) may fail or trigger filesystem allocation on the write path, reducing throughput.

## Multiple File Handles

A cuFile application can register multiple file handles simultaneously:

```c
CUfileDescr_t fh_reads[4];
CUfileDescr_t fh_writes[4];

for (int i = 0; i < 4; i++) {
    char path[256];
    snprintf(path, sizeof(path), "/mnt/nvme/data_%d.bin", i);
    int fd = open(path, O_DIRECT | O_RDONLY);
    fh_reads[i].cookie = (CUfileDriverCookie)(uintptr_t)fd;
    cuFileHandleRegister(&fh_reads[i], NULL);
}

// Each handle can be used from a separate thread
```

**Thread safety rule:** One thread per handle. cuFile handles are NOT internally synchronized. If two threads call `cuFileRead` on the same handle simultaneously, behavior is undefined.

## Handle Lifecycle and Error Recovery

If `cuFileHandleRegister` fails:

1. The file descriptor IS still valid (you opened it successfully)
2. You can fall back to standard POSIX I/O with that fd
3. Close the fd normally

If `cuFileHandleRegister` succeeds but later I/O fails:

1. The handle may be in an error state
2. Deregister the handle → close fd → re-open → re-register
3. If errors persist, check filesystem health: `fsck /mnt/nvme`

## Common Pitfalls

| Pitfall                                   | Consequence                                | Prevention                                     |
| ----------------------------------------- | ------------------------------------------ | ---------------------------------------------- |
| Forgetting `O_DIRECT`                     | Silent compat mode, 2-3× worse throughput  | Always audit open() flags                      |
| Registering handle on NFS/CIFS mount      | GDS not available                          | Verify filesystem type with `df -T`            |
| File on different NUMA node than GPU/NVMe | Higher latency                             | Check PCIe topology with `gdscheck -p`         |
| Handle not deregistered before close      | Resource leak, possible driver error       | Always deregister before close                 |
| Using same handle from multiple threads   | Undefined behavior, data corruption        | One thread per handle, or add external locking |
| File is sparse or not pre-allocated       | Allocation during write reduces throughput | `ftruncate` before first I/O                   |

See also:

- `api-reference.md` for function signatures
- `buffer-management.md` for buffer registration (the other half)
- `performance-tuning.md` for filesystem-level tuning
- `configuration.md` for per-filesystem cufile.json settings
