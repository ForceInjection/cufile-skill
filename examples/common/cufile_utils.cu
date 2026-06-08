/*
 * cufile_utils.cu — Shared helper implementations for cuFile examples
 *
 * Compile as part of each example or as a shared object.
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
                cuFileGetErrorString(st));
        return -1;
    }

    printf("=== GDS Driver Status ===\n");
    printf("  Version:         v%d.%d\n", props.major, props.minor);
    printf("  GDS capable:     %s\n", props.is_gds_capable ? "YES" : "NO");
    printf("  GDS enabled:      %s\n", props.is_gds_enabled ? "YES" : "NO");
    printf("  Max direct IO:   %s\n", format_size(props.max_device_direct_io_size));
    printf("  Max device cache: %s\n", format_size(props.max_device_cache_size));

    if (!props.is_gds_capable) {
        printf("\n*** WARNING: GDS is NOT capable on this platform. ***\n");
        printf("    Run 'gdscheck -p' to diagnose. IO will use compat mode.\n");
    } else if (!props.is_gds_enabled) {
        printf("\n*** WARNING: GDS capable but NOT enabled. ***\n");
        printf("    Check /etc/cufile.json and nvidia-fs module.\n");
    } else {
        printf("\nGDS is fully operational.\n");
    }

    return (props.is_gds_enabled) ? 0 : -1;
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
                operation, cuFileGetErrorString(status), status.err);
        exit(EXIT_FAILURE);
    }
}

void cuFileCheckIO(ssize_t ret, ssize_t expected, const char *operation) {
    if (ret < 0) {
        fprintf(stderr, "cuFile IO ERROR: %s failed (ret=%zd)\n",
                operation, ret);
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
