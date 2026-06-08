/*
 * cufile_utils.h — Shared helper functions for cuFile examples
 *
 * Based on cuFile API v1.13 (CUDA 12.8). Key API differences from
 * newer versions are documented in references/api-reference.md.
 *
 * Provides:
 *   - check_gds_available(): Verify GDS driver status via nvfs.dstatusflags
 *   - check_alignment(): Validate buffer and offset alignment
 *   - cuFileCheck(): Error-checking wrapper using CUFILE_ERRSTR
 *   - measure_bandwidth(): Simple throughput measurement
 *   - format_size(): Human-readable size formatting
 */

#ifndef CUFILE_UTILS_H
#define CUFILE_UTILS_H

#include <cufile.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── GDS Status Check ──────────────────────────────────────────── */

/**
 * Verify GDS driver status and print diagnostic information.
 * Returns 0 if NVMe P2P is supported, -1 otherwise.
 *
 * Uses nvfs.dstatusflags (CU_FILE_NVME_P2P_SUPPORTED bit 11)
 * and nvfs.dcontrolflags (CU_FILE_ALLOW_COMPAT_MODE bit 1).
 */
int check_gds_available(void);

/* ── Alignment Check ───────────────────────────────────────────── */

/**
 * Check if a pointer/size/offset satisfies 4KB alignment for GDS.
 * Returns 1 if aligned, 0 otherwise.
 */
int check_alignment(const void *ptr, size_t size, off_t offset);

/* ── Error Handling ────────────────────────────────────────────── */

/**
 * Check cuFile error status and exit on failure.
 * Uses CUFILE_ERRSTR macro for error string.
 */
void cuFileCheck(CUfileError_t status, const char *operation);

/**
 * Check cuFile read/write return value and exit on failure.
 */
void cuFileCheckIO(ssize_t ret, ssize_t expected, const char *operation);

/* ── Timing & Measurement ──────────────────────────────────────── */

/**
 * Simple throughput measurement for a block of I/O.
 * Returns bandwidth in GB/s.
 */
double measure_bandwidth(size_t total_bytes, double elapsed_seconds);

/**
 * Format a byte size to human-readable string (e.g., "128 MB").
 * Returns pointer to static buffer.
 */
const char* format_size(size_t bytes);

/* ── File Helpers ──────────────────────────────────────────────── */

/**
 * Open a file with O_DIRECT and proper error handling.
 * Returns file descriptor on success, exits on failure.
 */
int open_direct(const char *path, int flags, mode_t mode);

/**
 * Pre-allocate a file to a specific size using ftruncate.
 */
void preallocate_file(int fd, size_t size);

/* ── GDS Diagnostic Helpers ────────────────────────────────────── */

/**
 * Check if NVMe P2P DMA is supported (bit 11 in nvfs.dstatusflags).
 */
static inline int gds_nvme_p2p_supported(const CUfileDrvProps_t *props) {
    return (props->nvfs.dstatusflags & (1u << CU_FILE_NVME_P2P_SUPPORTED)) != 0;
}

/**
 * Check if NVMe is supported (bit 4 in nvfs.dstatusflags).
 */
static inline int gds_nvme_supported(const CUfileDrvProps_t *props) {
    return (props->nvfs.dstatusflags & (1u << CU_FILE_NVME_SUPPORTED)) != 0;
}

/**
 * Check if compatibility mode is allowed (bit 1 in nvfs.dcontrolflags).
 */
static inline int gds_compat_mode_allowed(const CUfileDrvProps_t *props) {
    return (props->nvfs.dcontrolflags & (1u << CU_FILE_ALLOW_COMPAT_MODE)) != 0;
}

#ifdef __cplusplus
}
#endif

#endif /* CUFILE_UTILS_H */
