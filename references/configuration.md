# cuFile Configuration: cufile.json Reference

## Overview

`cufile.json` is the global configuration file for GPUDirect Storage. It controls GDS enable/disable, memory limits, IO size thresholds, and filesystem-specific tuning. The default location is `/etc/cufile.json`. Override with the environment variable `CUFILE_CONFIG`.

## Configuration File Location

```bash
# Default location
/etc/cufile.json

# Environment variable override
export CUFILE_CONFIG=/path/to/custom/cufile.json

# Per-user configuration (if supported by driver version)
~/.cufile.json
```

Priority: `CUFILE_CONFIG` env var > `/etc/cufile.json` > built-in defaults.

## Complete Configuration Reference

```json
{
  "properties": {
    "max_device_cache_size": 134217728,
    "max_device_pinned_mem_size": 0,
    "max_direct_io_size": 1073741824,
    "max_device_allocated_mem_size": 0,
    "enable_compat_mode": true,
    "profiling": {
      "enable": false,
      "trace": false
    },
    "fs": {
      "ext4": {
        "max_direct_io_chunk_size": 0
      },
      "xfs": {
        "max_direct_io_chunk_size": 0
      },
      "gpfs": {
        "max_direct_io_chunk_size": 0
      }
    }
  }
}
```

## Property Reference

### `max_device_cache_size`

```json
"max_device_cache_size": 134217728  // 128 MB (default)
```

**Purpose:** Maximum size of the GPU-side cache used by GDS to buffer data for small or unaligned I/O operations.

**Tuning guide:**

- **128 MB (default)**: Good for general workloads
- **256-512 MB**: Large sequential workloads with high throughput — larger cache reduces NVMe round-trips
- **1-2 GB**: Extreme throughput workloads (checkpoint/restore, model loading)
- **0** (disabled): No GDS cache — forces every I/O to go directly to NVMe. Useful for debugging.

**Trade-off:** Larger cache = higher throughput for medium I/Os, but consumes GPU memory that could be used for compute.

### `max_device_pinned_mem_size`

```json
"max_device_pinned_mem_size": 0  // 0 = unlimited (default)
```

**Purpose:** Maximum GPU memory (in bytes) that the GDS driver can pin for DMA. Pinning prevents the GPU memory manager from swapping or migrating those pages.

**Tuning guide:**

- **0 (unlimited)**: Let GDS use as much GPU memory as needed
- **8 GB (8589934592)**: Reserve 8GB for GDS on a 40GB GPU — leaves 32GB for compute
- **16 GB (17179869184)**: Large checkpoint workloads

**When to set a limit:**

- Multi-tenant GPU (MIG, MPS) — prevent GDS from starving other tenants
- GPU memory-constrained workloads — ensure compute always has enough memory
- Debugging GPU OOM issues — rule out GDS as the cause

### `max_direct_io_size`

```json
"max_direct_io_size": 1073741824  // 1 GB (default)
```

**Purpose:** Maximum single I/O request size that uses the GDS direct path. Requests larger than this are split or use compat mode.

**Tuning guide:**

- **1 GB (default)**: Suitable for most workloads
- **4-8 GB**: Checkpoint/snapshot workloads with very large single writes
- **16 GB (17179869184)**: Maximum supported in recent drivers

**Note:** Even with a large `max_direct_io_size`, the actual I/O may be limited by the NVMe device's Maximum Data Transfer Size (MDTS) from NVMe Identify. GDS splits large requests internally.

### `max_device_allocated_mem_size`

```json
"max_device_allocated_mem_size": 0  // 0 = unlimited (default)
```

**Purpose:** Maximum GPU memory the GDS driver can allocate for internal use (not pin user buffers).

**Tuning guide:** Typically leave at 0 (unlimited). Only set a limit if profiling shows unexpected GPU memory usage and you suspect GDS internal allocations.

### `enable_compat_mode`

```json
"enable_compat_mode": true  // default: true
```

**Purpose:** When GDS cannot be used (alignment, filesystem, topology issues), silently fall back to CPU bounce buffer.

**Development setting:**

```json
"enable_compat_mode": false
```

With compat mode disabled, operations that can't use GDS will return an explicit error (`CU_FILE_NOT_SUPPORTED`). This is THE most useful setting during development — it catches misconfiguration immediately.

**Production setting:**

```json
"enable_compat_mode": true
```

Allows the application to function even if GDS prerequisites aren't met (degraded performance, but correct behavior).

### `profiling.enable`

```json
"profiling": {
  "enable": false,   // default: false
  "trace": false     // default: false
}
```

**`profiling.enable`**: Enable cuFile internal performance profiling. Collects latency histograms, throughput statistics, and error counts. Has a small overhead (~1-3%).

**`profiling.trace`**: Enable verbose tracing of every cuFile API call. Generates very large logs. Only enable for targeted debugging sessions.

### Filesystem-Specific Settings

```json
"fs": {
  "ext4": {
    "max_direct_io_chunk_size": 0
  },
  "xfs": {
    "max_direct_io_chunk_size": 0
  }
}
```

**`max_direct_io_chunk_size`**: Per-filesystem maximum chunk size for direct I/O (bytes). 0 = use default.

**Tuning guide:**

- ext4: 8-16 MB (ext4 direct I/O can fragment large requests)
- xfs: 16-32 MB (xfs handles large direct I/O more efficiently)
- GPFS: Consult IBM documentation, typically 16-64 MB

## Configuration Templates

### Development / Debugging

```json
{
  "properties": {
    "enable_compat_mode": false,
    "profiling": {
      "enable": true,
      "trace": false
    },
    "max_device_cache_size": 0
  }
}
```

This configuration:

- Fails loudly when GDS isn't working (instead of silent fallback)
- Enables profiling for performance analysis
- Disables cache to measure raw NVMe performance

### Production: AI Training Data Pipeline

```json
{
  "properties": {
    "max_device_cache_size": 268435456,
    "max_direct_io_size": 4294967296,
    "max_device_pinned_mem_size": 0,
    "enable_compat_mode": true,
    "fs": {
      "xfs": {
        "max_direct_io_chunk_size": 33554432
      }
    }
  }
}
```

- 256 MB cache: Buffers training data batches
- 4 GB max IO: Handles large sample files
- Unlimited pinned memory: Training GPU is dedicated to this workload
- xfs optimization: 32MB chunks for efficient large reads

### Production: HPC Checkpoint/Restore

```json
{
  "properties": {
    "max_device_cache_size": 536870912,
    "max_direct_io_size": 17179869184,
    "max_device_pinned_mem_size": 17179869184,
    "enable_compat_mode": true,
    "fs": {
      "gpfs": {
        "max_direct_io_chunk_size": 67108864
      }
    }
  }
}
```

- 512 MB cache: Reduces NVMe round-trips
- 16 GB max IO: Full model checkpoint in single operation
- 16 GB pinned limit: Reserve half of a 32GB GPU for GDS
- GPFS tuning: 64MB chunks for parallel filesystem

### Production: Real-Time Inference (Low Latency)

```json
{
  "properties": {
    "max_device_cache_size": 0,
    "max_direct_io_size": 134217728,
    "enable_compat_mode": false
  }
}
```

- Cache disabled: Eliminates cache eviction jitter
- 128 MB max IO: Limits per-operation latency
- Compat mode disabled: Fails fast on misconfiguration

## Validating Configuration

```bash
# Check if cufile.json is valid JSON
python3 -m json.tool /etc/cufile.json > /dev/null && echo "Valid" || echo "Invalid JSON"

# Reload configuration (requires driver restart)
sudo rmmod nvidia-fs && sudo modprobe nvidia-fs

# Verify config took effect
gdscheck -v
```

## Environment Variables

| Variable          | Purpose                   | Example                   |
| ----------------- | ------------------------- | ------------------------- |
| `CUFILE_CONFIG`   | Override config file path | `/home/user/.cufile.json` |
| `CUFILE_LOGGING`  | Enable detailed logging   | `1` or `0`                |
| `CUFILE_LOG_FILE` | Custom log file path      | `/tmp/cufile_debug.log`   |

See also:

- `performance-tuning.md` for applying these settings in a tuning workflow
- `driver-lifecycle.md` for driver init that reads these settings
- `error-handling.md` for GDS-not-available errors when `enable_compat_mode: false`
