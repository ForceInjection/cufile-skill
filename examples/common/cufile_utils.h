/*
 * cufile_utils.h — Shared helper functions for cuFile examples
 *
 * Provides:
 *   - check_gds_available(): Verify GDS driver status
 *   - check_alignment(): Validate buffer and offset alignment
 *   - cuFileCheck(): Error-checking wrapper
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
 * Returns 0 if GDS is fully operational, -1 otherwise.
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

#ifdef __cplusplus
}
#endif

#endif /* CUFILE_UTILS_H */
