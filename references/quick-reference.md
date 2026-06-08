# cuFile Quick Reference Card

## Core Types

```text
CUfileError_t     — { CUfileOpError err; } — function return type
CUfileDescr_t     — { CUfileHandle handle; CUfileDriverCookie cookie; } — file handle descriptor
CUfileDrvProps_t  — Driver properties (GDS capable, enabled, version, IO limits)
CUfileIOParams_t  — Batch I/O parameter struct
CUfileIOStatus_t  — Batch I/O completion status
CUfileOpError     — Enum: CU_FILE_SUCCESS, CU_FILE_NOT_FOUND, CU_FILE_INVALID_HANDLE, etc.
```

## API Layering

```text
Init:        cuFileDriverOpen() → cuFileDriverGetProperties()
Buffers:     cudaMalloc → cuFileBufRegister → ... → cuFileBufDeregister → cudaFree
Files:       open(O_DIRECT) → cuFileHandleRegister → ... → cuFileHandleDeregister → close
Sync I/O:    cuFileRead(fh, devPtr, size, offset, 0) / cuFileWrite(...)
Async I/O:   cuFileStreamRegister(stream) → cuFileReadAsync(fh, devPtr, size, offset, stream)
Batch I/O:   cuFileBatchIOSetUp → cuFileBatchIOSubmit → cuFileBatchIOGetStatus
Cleanup:     Always reverse order of setup
```

## Complete API Function Signature Summary

### Driver & Lifecycle

```c
CUfileError_t cuFileDriverOpen(void);
CUfileError_t cuFileDriverClose(void);
CUfileError_t cuFileDriverGetProperties(CUfileDrvProps_t *props);
```

### Buffer Management

```c
CUfileError_t cuFileBufRegister(CUdeviceptr devPtr, size_t size, int flags);
CUfileError_t cuFileBufDeregister(CUdeviceptr devPtr);
CUfileError_t cuFileBufSetAllocFlags(CUdeviceptr devPtr, int flags);
```

### File Handle Management

```c
CUfileError_t cuFileHandleRegister(CUfileDescr_t *descr, CUfileDrvProps_t *props);
CUfileError_t cuFileHandleDeregister(CUfileDescr_t descr);
```

### Synchronous I/O

```c
ssize_t cuFileRead(CUfileDescr_t fh, CUdeviceptr devPtr, size_t size,
                   off_t offset, off_t devPtr_offset);
ssize_t cuFileWrite(CUfileDescr_t fh, CUdeviceptr devPtr, size_t size,
                    off_t offset, off_t devPtr_offset);
```

### Asynchronous I/O

```c
CUfileError_t cuFileReadAsync(CUfileDescr_t fh, CUdeviceptr devPtr,
                              size_t size, off_t offset,
                              off_t devPtr_offset, CUstream stream);
CUfileError_t cuFileWriteAsync(CUfileDescr_t fh, CUdeviceptr devPtr,
                               size_t size, off_t offset,
                               off_t devPtr_offset, CUstream stream);
CUfileError_t cuFileStreamRegister(CUfileDescr_t fh, CUstream stream);
```

### Batch I/O

```c
CUfileError_t cuFileBatchIOSetUp(CUfileDescr_t fh, int batch_id,
                                 int max_ios, int flags);
CUfileError_t cuFileBatchIOSubmit(int batch_id, int num_ios,
                                  CUfileIOParams_t *io_params, int flags);
CUfileError_t cuFileBatchIOGetStatus(int batch_id, int min_ios,
                                     int max_ios, CUfileIOStatus_t *io_status,
                                     int *num_completed);
CUfileError_t cuFileBatchIOCancel(int batch_id);
CUfileError_t cuFileBatchIODestroy(int batch_id);
```

### Error Handling

```c
const char* cuFileGetErrorString(CUfileError_t error);
CUfileOpError cuFileOpError(CUfileError_t error);
```

## GDS Requirements Checklist

| Requirement    | Minimum                        | Recommended                    |
| -------------- | ------------------------------ | ------------------------------ |
| GPU            | Pascal (SM 6.0)                | Ampere+ (SM 8.0)               |
| NVMe           | PCIe Gen3 x4                   | PCIe Gen4 x4                   |
| Filesystem     | ext4, xfs, GPFS                | xfs with DAX                   |
| Kernel         | nvidia-fs.ko loaded            | nvidia-fs.ko loaded            |
| Topology       | GPU + NVMe on same PCIe root complex | ACS disabled for P2P     |
| File flags     | O_DIRECT on open               | O_DIRECT                       |
| Alignment      | 4KB (buffer + offset)          | GPU page size                  |
| IO size        | ≥ 4KB for GDS                  | ≥ 1MB for high throughput      |

## Diagnostic Commands

```bash
gdscheck -p                                                        # Platform readiness
gdscheck -f /mnt/nvme                                              # Filesystem readiness
gdsio -f /mnt/nvme/testfile                                        # Functional test (write + read + verify)
sudo lspci -vvv | grep -i "ACS"                                    # ACS status on all devices
sudo lspci -vvv -s <BDF> | grep "ACSCtl"                           # ACS on specific device
lspci -t -v                                                        # PCIe tree (verify same root complex)
nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current --format=csv  # GPU PCIe status
cat /proc/driver/nvidia-fs/gds                                     # GDS kernel module status
modinfo nvidia-fs | grep version                                   # nvidia-fs version
bash scripts/check_gds.sh /mnt/nvme                                # Automated GDS readiness check
```

## Error Classification Quick Reference

| Error Code                       | Value | Class                    | Recovery                         |
| -------------------------------- | ----- | ------------------------ | -------------------------------- |
| `CU_FILE_SUCCESS`                | 0     | —                        | —                                |
| `CU_FILE_NOT_FOUND`              | 2     | Recoverable              | Check path, create file          |
| `CU_FILE_INVALID_HANDLE`         | 5     | Recoverable              | Re-register handle               |
| `CU_FILE_INVALID_PARAMETER`      | 6     | Recoverable              | Fix parameter, retry             |
| `CU_FILE_DISK_FULL`              | 12    | Transient                | Free space, retry                |
| `CU_FILE_INTERNAL_ERROR`         | 14    | **Fatal**                | Check dmesg, reinitialize        |
| `CU_FILE_NOT_REGISTERED`         | 15    | Recoverable              | Call cuFileBufRegister first     |
| `CU_FILE_NOT_OPENED`             | 16    | Recoverable              | Re-open with O_DIRECT            |
| `CU_FILE_NOT_SUPPORTED`          | 18    | Config                   | Enable compat or fix GDS reqs    |
| `CU_FILE_IO_ERROR`               | 19    | Transient                | Retry, check NVMe health         |
| `CU_FILE_INVALID_OFFSET`         | 20    | Recoverable              | Align to 4KB                     |
| `CU_FILE_INVALID_SIZE`           | 21    | Recoverable              | Round to 4KB multiple            |
| `CU_FILE_INVALID_ALIGNMENT`      | 22    | Recoverable              | Align buffer to 4KB              |
| `CU_FILE_MEMORY_ALLOCATION`      | 23    | **Fatal**                | Free memory, reduce buffers      |
| `CU_FILE_TIMEOUT`                | 24    | Transient                | Retry, increase timeout          |
| `CU_FILE_CANCELED`               | 25    | Transient                | Re-submit cancelled I/Os         |
| `CU_FILE_DRIVER_NOT_INITIALIZED` | 26    | **Fatal**                | Call cuFileDriverOpen first      |

## cufile.json Key Parameters

```json
{
  "properties": {
    "max_device_cache_size": 134217728,
    "max_device_pinned_mem_size": 0,
    "max_direct_io_size": 1073741824,
    "max_device_allocated_mem_size": 0,
    "enable_compat_mode": true
  }
}
```

| Parameter                    | Default     | Description                                    | Tuning Hint                        |
| ---------------------------- | ----------- | ---------------------------------------------- | ---------------------------------- |
| `max_device_cache_size`      | 128 MB      | GDS device-side cache                          | 256-512 MB for sequential workload |
| `max_device_pinned_mem_size` | 0 (unlim)   | Max GPU memory pinned for GDS                  | Set limit for multi-tenant GPU     |
| `max_direct_io_size`         | 1 GB        | Max single direct IO size                      | 4-16 GB for checkpoint workloads   |
| `enable_compat_mode`         | true        | Fallback to CPU bounce buffer                  | Set `false` during development     |

## IO Size Selection

| Workload                     | IO Size         | Batch?   | Expected Throughput (Gen4 x4) |
| ---------------------------- | --------------- | -------- | ---------------------------- |
| Small random reads           | 4KB – 64KB      | ✅ Yes   | 1-5 GB/s (batch dependent)   |
| Medium sequential            | 256KB – 1MB     | Optional | 15-25 GB/s                   |
| Large sequential             | 1MB – 16MB      | No       | 25-30 GB/s                   |
| Checkpoint / model save      | 16MB – 256MB    | No       | 25-30 GB/s                   |
| Data pipeline (train+ingest) | 1MB – 8MB       | ✅ Yes   | 20-28 GB/s (with overlap)    |

## Common Performance Pitfalls: Quick Diagnose

| Symptom                                  | First Check                                  | Fix                                    |
| ---------------------------------------- | -------------------------------------------- | -------------------------------------- |
| Low throughput (2-3× below expected)     | `cuFileDriverGetProperties()` → is_gds_enabled | O_DIRECT, alignment, cufile.json     |
| `cuFileBufRegister` fails                | `(uintptr_t)ptr % 4096`                      | Use `cuMemAlloc` for guaranteed alignment |
| `cuFileHandleRegister` fails             | `df -T /mnt/nvme`                             | ext4/xfs required, NOT NFS            |
| Throughput plateaus at high IO size      | `props.max_device_direct_io_size`             | Increase in cufile.json                |
| High CPU during I/O                      | Compat mode active                             | Fix GDS prerequisites                  |
| Async ops seem sequential                | Compat mode active (CPU can't truly overlap)   | Verify GDS is enabled                  |

See also:
- `api-reference.md` for full function signatures and struct definitions
- `error-handling.md` for complete error code reference and recovery strategies
- `configuration.md` for full cufile.json reference
- `performance-tuning.md` for end-to-end tuning workflow
