/*
 * cufile_utils.cu — Shared helper implementations for cuFile examples
 *
 * Based on cuFile API v1.13 (CUDA 12.8). Uses CUFILE_ERRSTR macro
 * for error strings and nvfs.dstatusflags for GDS capability checks.
 */

#include "cufile_utils.h"
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

/* ── GDS Status Check ──────────────────────────────────────────── */

int check_gds_available(void) {
    CUfileDrvProps_t props = {0};
    CUfileError_t st = cuFileDriverGetProperties(&props);

    if (st.err != CU_FILE_SUCCESS) {
        fprintf(stderr, "ERROR: cuFileDriverGetProperties failed: %s\n",
                CUFILE_ERRSTR(st.err));
        return -1;
    }

    int nvme_ok  = gds_nvme_supported(&props);
    int p2p_ok   = gds_nvme_p2p_supported(&props);
    int compat   = gds_compat_mode_allowed(&props);

    printf("=== GDS Driver Status ===\n");
    printf("  nvfs version:    v%d.%d\n",
           props.nvfs.major_version, props.nvfs.minor_version);
    printf("  NVMe supported:  %s\n", nvme_ok ? "YES" : "NO");
    printf("  NVMe P2P (GDS):  %s\n", p2p_ok  ? "YES" : "NO");
    printf("  Compat mode:     %s\n", compat  ? "ALLOWED" : "DISABLED");
    printf("  Max direct IO:   %s\n",
           format_size(props.nvfs.max_direct_io_size));
    printf("  Max device cache: %s\n",
           format_size(props.max_device_cache_size));

    if (!nvme_ok) {
        printf("\n*** WARNING: NVMe not detected by nvidia-fs driver. ***\n");
        printf("    Run 'gdscheck -p' to diagnose.\n");
    } else if (!p2p_ok) {
        printf("\n*** WARNING: NVMe P2P (GDS) NOT supported. ***\n");
        printf("    Check PCIe topology: GPU+NVMe on same root complex, ACS disabled.\n");
        printf("    I/O will use compatibility mode (CPU bounce buffer).\n");
    } else {
        printf("\nNVMe P2P (GDS) is available.\n");
    }

    return (p2p_ok) ? 0 : -1;
}

/* ── Alignment Check ───────────────────────────────────────────── */

int check_alignment(const void *ptr, size_t size, off_t offset) {
    int ok = 1;

    if ((uintptr_t)ptr % 4096 != 0) {
        fprintf(stderr, "ALIGNMENT ERROR: Buffer %p is not 4KB-aligned "
                "(offset=%lu bytes)\n",
                ptr, (uintptr_t)ptr % 4096);
        ok = 0;
    }

    if (size % 4096 != 0) {
        fprintf(stderr, "ALIGNMENT WARNING: Size %s is not a multiple of 4KB\n",
                format_size(size));
        /* Not a hard error — cuFile handles this, just not optimally */
    }

    if (offset % 4096 != 0) {
        fprintf(stderr, "ALIGNMENT ERROR: File offset %lld is not 4KB-aligned "
                "(offset=%lld bytes)\n",
                (long long)offset, (long long)(offset % 4096));
        ok = 0;
    }

    return ok;
}

/* ── Error Handling ────────────────────────────────────────────── */

void cuFileCheck(CUfileError_t status, const char *operation) {
    if (status.err != CU_FILE_SUCCESS) {
        fprintf(stderr, "cuFile ERROR: %s failed: %s (err=%d)\n",
                operation, CUFILE_ERRSTR(status.err), status.err);
        exit(EXIT_FAILURE);
    }
}

void cuFileCheckIO(ssize_t ret, ssize_t expected, const char *operation) {
    if (ret < 0) {
        int err_code = (int)(-ret);
        fprintf(stderr, "cuFile IO ERROR: %s failed: %s (code=%d)\n",
                operation, CUFILE_ERRSTR(err_code), err_code);
        exit(EXIT_FAILURE);
    }
    if (expected > 0 && ret != expected) {
        fprintf(stderr, "cuFile IO WARNING: %s short transfer: "
                "%zd / %zd bytes\n", operation, ret, expected);
    }
}

/* ── Timing & Measurement ──────────────────────────────────────── */

double measure_bandwidth(size_t total_bytes, double elapsed_seconds) {
    if (elapsed_seconds <= 0) return 0;
    return (total_bytes / 1e9) / elapsed_seconds;  // GB/s
}

const char* format_size(size_t bytes) {
    static char buf[32];
    const char *units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit_idx = 0;
    double size = (double)bytes;

    while (size >= 1024.0 && unit_idx < 4) {
        size /= 1024.0;
        unit_idx++;
    }

    if (unit_idx == 0) {
        snprintf(buf, sizeof(buf), "%zu B", bytes);
    } else {
        snprintf(buf, sizeof(buf), "%.2f %s", size, units[unit_idx]);
    }
    return buf;
}

/* ── File Helpers ──────────────────────────────────────────────── */

int open_direct(const char *path, int flags, mode_t mode) {
    int fd = open(path, flags | O_DIRECT, mode);
    if (fd < 0) {
        perror("open(O_DIRECT)");
        fprintf(stderr, "  Path: %s\n", path);
        fprintf(stderr, "  Hint: Check filesystem supports O_DIRECT "
                "(ext4/xfs, not NFS)\n");
        exit(EXIT_FAILURE);
    }
    return fd;
}

void preallocate_file(int fd, size_t size) {
    if (ftruncate(fd, size) != 0) {
        perror("ftruncate");
        fprintf(stderr, "  Failed to pre-allocate %s\n", format_size(size));
        exit(EXIT_FAILURE);
    }
}
