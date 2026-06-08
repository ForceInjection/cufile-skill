# cuFile Performance Tuning

## Overview

cuFile performance tuning follows a systematic workflow. The most common failure mode is not "bad tuning" — it's "GDS was never enabled." Always start with verification, then optimize.

## Performance Tuning Workflow

```text
Phase 1: Verify GDS Status
  ├── gdscheck -p          # Platform: GPU + NVMe + PCIe topology
  ├── gdscheck -f /mnt/nvme # Filesystem: mount options, O_DIRECT support
  └── Code: cuFileDriverGetProperties() → is_gds_enabled

Phase 2: Fix Basic Prerequisites
  ├── Buffer alignment (4KB)
  ├── File offset alignment (4KB)
  ├── IO size ≥ 4KB, ideally ≥ 1MB
  └── O_DIRECT on file open

Phase 3: Measure Baseline
  ├── Sequential read throughput (gdsio or custom benchmark)
  ├── Sequential write throughput
  ├── Random read IOPS (if applicable)
  └── CPU utilization during I/O

Phase 4: Apply Optimizations (in order of impact)
  ├── 1. Increase IO size to 1-16MB
  ├── 2. Enable batch I/O (if many small I/Os)
  ├── 3. Use async I/O with compute overlap
  ├── 4. Multi-stream I/O (if IO-bound)
  ├── 5. NUMA binding of I/O threads
  └── 6. Tune cufile.json parameters

Phase 5: Profile and Iterate
  └── Compare against theoretical PCIe bandwidth
```

## Phase 1: Verification — Is GDS Actually Working?

This is the single most important phase. Skipping it means you might be tuning compat mode.

### Automated Verification Code

```c
int verify_gds_working(CUfileDescr_t fh, CUdeviceptr devPtr, size_t size) {
    CUfileDrvProps_t props;
    cuFileDriverGetProperties(&props);

    printf("=== GDS Verification ===\n");

    // Check 1: Driver properties
    printf("[%s] GDS capable\n",
           props.is_gds_capable ? "PASS" : "FAIL");
    printf("[%s] GDS enabled\n",
           props.is_gds_enabled ? "PASS" : "FAIL");

    if (!props.is_gds_capable) {
        printf("\nGDS not capable. Check:\n");
        printf("  1. GPU is Pascal+ (SM 6.0+)\n");
        printf("  2. nvidia-fs kernel module loaded\n");
        printf("  3. GPU + NVMe on same PCIe root complex\n");
        printf("  4. ACS disabled for P2P\n");
        printf("  Run: gdscheck -p\n");
        return -1;
    }

    if (!props.is_gds_enabled) {
        printf("\nGDS capable but not enabled. Check:\n");
        printf("  1. /etc/cufile.json: enable_compat_mode should be true\n");
        printf("  2. Filesystem supports O_DIRECT\n");
        printf("  3. File opened with O_DIRECT\n");
        printf("  Run: gdscheck -f <mountpoint>\n");
        return -1;
    }

    // Check 2: Alignment
    bool aligned = ((uintptr_t)devPtr % 4096 == 0);
    printf("[%s] Buffer alignment (4KB)\n", aligned ? "PASS" : "FAIL");

    // Check 3: Functional I/O test
    ssize_t bytes = cuFileRead(fh, devPtr, size, 0, 0);
    printf("[%s] Functional I/O (%zd bytes)\n",
           (bytes == size) ? "PASS" : "FAIL", bytes);

    return (props.is_gds_enabled && aligned && bytes == size) ? 0 : -1;
}
```

## Phase 2-3: Baseline Measurement

```c
#include <time.h>

double measure_read_throughput(CUfileDescr_t fh, CUdeviceptr devPtr,
                               size_t io_size, size_t total_size, int repeat) {
    double best_bw = 0;
    size_t num_ios = total_size / io_size;

    for (int r = 0; r < repeat; r++) {
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        for (size_t i = 0; i < num_ios; i++) {
            ssize_t n = cuFileRead(fh, devPtr, io_size, i * io_size, 0);
            if (n != io_size) {
                fprintf(stderr, "Short read at iteration %zu\n", i);
                return -1;
            }
        }

        clock_gettime(CLOCK_MONOTONIC, &end);
        double elapsed = (end.tv_sec - start.tv_sec) +
                         (end.tv_nsec - start.tv_nsec) / 1e9;
        double bw = (total_size / 1e9) / elapsed; // GB/s
        if (bw > best_bw) best_bw = bw;
        printf("  Run %d: %.2f GB/s\n", r + 1, bw);
    }

    printf("Best: %.2f GB/s\n", best_bw);
    return best_bw;
}
```

## Phase 4: Optimization — Ordered by Impact

### 1. IO Size Optimization

```c
void find_optimal_io_size(CUfileDescr_t fh, CUdeviceptr devPtr,
                          size_t max_size) {
    printf("=== IO Size Sweep ===\n");
    printf("Size       Throughput\n");
    printf("--------   ----------\n");

    // Sweep from 4KB to max_size, doubling each step
    for (size_t sz = 4096; sz <= max_size; sz *= 2) {
        size_t num_ios = max_size / sz;
        struct timespec t1, t2;
        clock_gettime(CLOCK_MONOTONIC, &t1);

        for (size_t i = 0; i < num_ios; i++) {
            cuFileRead(fh, devPtr, sz, i * sz, 0);
        }

        clock_gettime(CLOCK_MONOTONIC, &t2);
        double elapsed = (t2.tv_sec - t1.tv_sec) + (t2.tv_nsec - t1.tv_nsec) / 1e9;
        double bw = (max_size / 1e9) / elapsed;

        printf("%-10s %.2f GB/s\n", format_size(sz), bw);
    }
}
```

**Expected results with GDS:**

- 4KB: ~1-2 GB/s (per-IO overhead dominant)
- 64KB: ~10-15 GB/s
- 256KB: ~20-25 GB/s
- 1MB: ~25-28 GB/s
- 4MB+: ~28-30 GB/s (saturating PCIe Gen4 x4)

### 2. Batch I/O (many small I/Os)

If IO size must be small (4KB-256KB), batch I/O is the biggest win:

```c
// Without batch: 500K IOPS max (individual submission overhead)
// With batch:    2M+ IOPS (amortized submission overhead)
```

See `batch-io.md` for the complete implementation.

### 3. Async I/O with Compute Overlap

If your workload has GPU compute between I/Os, async I/O can hide latency entirely:

```c
// Sequential (no overlap):
//   [Read 100ms] → [Compute 50ms] → Total: 150ms per iteration

// Async (overlapped):
//   [Read 100ms]
//   [          Compute 50ms]  → Total: 100ms per iteration
```

See `async-io.md` for the complete implementation.

### 4. Thread Affinity and NUMA

```c
#include <sched.h>
#include <numa.h>

void bind_thread_to_pcie_numa(int cpu) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);

    // Also bind memory allocations to local NUMA node
    int numa_node = numa_node_of_cpu(cpu);
    numa_set_preferred(numa_node);
}
```

**Finding the right NUMA node:**

```bash
# Find NVMe PCIe address
lspci | grep "Non-Volatile"

# Find NUMA node for that PCIe address
cat /sys/bus/pci/devices/0000:17:00.0/numa_node

# Bind your I/O thread to a CPU on that node
numactl --cpunodebind=0 --membind=0 ./my_app
```

### 5. cufile.json Tuning

```json
{
  "properties": {
    "max_device_cache_size": 268435456,
    "max_direct_io_size": 8589934592,
    "enable_compat_mode": true
  }
}
```

See `configuration.md` for the full reference.

## Theoretical Maximums and Realistic Targets

| PCIe Generation | Lanes | Theoretical BW | Realistic GDS Target | Compat Mode Target |
| --------------- | ----- | -------------- | -------------------- | ------------------ |
| Gen3            | x4    | 3.94 GB/s      | 3.2-3.5 GB/s         | 2.0-2.5 GB/s       |
| Gen3            | x16   | 15.75 GB/s     | 12-14 GB/s           | 7-10 GB/s          |
| Gen4            | x4    | 7.88 GB/s      | 6.5-7.2 GB/s         | 4-5 GB/s           |
| Gen4            | x16   | 31.5 GB/s      | 25-28 GB/s           | 12-18 GB/s         |
| Gen5            | x4    | 15.75 GB/s     | 13-14 GB/s           | 8-10 GB/s          |

**Note:** Realistic targets account for PCIe protocol overhead (~10-15% for GDS, ~30-50% for compat mode).

## Benchmarking Methodology

### Using gdsio (built-in GDS benchmarking tool)

```bash
# Sequential read benchmark
gdsio -f /mnt/nvme/testfile -s 1G -i 1M -x 0

# Sequential write benchmark
gdsio -f /mnt/nvme/testfile -s 1G -i 1M -x 1 -w 1

# Random read benchmark
gdsio -f /mnt/nvme/testfile -s 1G -i 4K -x 0 -r 1

# Flags:
#   -s SIZE    Total data size
#   -i SIZE    IO size per operation
#   -x FLAGS   GDS flags (0=auto, 1=force GDS, 2=force compat)
#   -w 1       Write mode (0=read, 1=write)
#   -r 1       Random I/O mode
```

### Custom Benchmark Structure

```c
// For reproducible benchmarks:
// 1. Drop filesystem caches before each run
//    echo 3 > /proc/sys/vm/drop_caches
// 2. Pre-allocate file with ftruncate()
// 3. Warm up with 2-3 iterations (not measured)
// 4. Measure 5-10 iterations, take the best
// 5. Report bandwidth in GB/s (not MB/s or Gbps)

double benchmark_cufile_read(CUfileDescr_t fh, CUdeviceptr devPtr,
                             size_t io_size, size_t total_size,
                             int warmup, int measure) {
    // Warmup (not measured)
    for (int i = 0; i < warmup; i++) {
        for (size_t off = 0; off < total_size; off += io_size) {
            cuFileRead(fh, devPtr, io_size, off, 0);
        }
    }

    // Measurement
    double best_bw = 0;
    for (int r = 0; r < measure; r++) {
        struct timespec t1, t2;
        clock_gettime(CLOCK_MONOTONIC, &t1);

        for (size_t off = 0; off < total_size; off += io_size) {
            cuFileRead(fh, devPtr, io_size, off, 0);
        }

        clock_gettime(CLOCK_MONOTONIC, &t2);
        double elapsed = (t2.tv_sec - t1.tv_sec) +
                         (t2.tv_nsec - t1.tv_nsec) / 1e9;
        double bw = (total_size / 1e9) / elapsed;
        if (bw > best_bw) best_bw = bw;
    }
    return best_bw;
}
```

## Common Performance Anti-Patterns

| Anti-Pattern                               | Impact                                | Fix                                                   |
| ------------------------------------------ | ------------------------------------- | ----------------------------------------------------- |
| Register/deregister buffer per I/O         | 10-100× slower — pinning overhead     | Register once at init, reuse                          |
| Using `cudaMalloc` without alignment check | 256B alignment → compat mode          | Use `cuMemAlloc` or check alignment                   |
| Forgetting `O_DIRECT`                      | Always compat mode                    | Add `O_DIRECT` to `open()`                            |
| IO size = 512B or 4KB repeatedly           | Per-IO overhead > transfer time       | Batch or combine into larger I/Os                     |
| Single-threaded for multi-file workloads   | PCIe bandwidth underutilized          | One thread per file, pinned to NUMA                   |
| No warmup before benchmarking              | First run includes cache allocation   | Always warm up 2-3 iterations                         |
| Not dropping filesystem cache              | Cached reads show memory BW, not NVMe | `echo 3 > /proc/sys/vm/drop_caches` before read tests |
| Measuring with `gettimeofday`              | Low resolution (~1ms)                 | Use `clock_gettime(CLOCK_MONOTONIC, ...)`             |

See also:

- `batch-io.md` for batch I/O optimization
- `async-io.md` for compute-I/O overlap
- `configuration.md` for cufile.json tuning
- `comparison-spdk.md` for cuFile vs alternatives
