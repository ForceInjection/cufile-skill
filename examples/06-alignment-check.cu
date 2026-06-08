/*
 * 06-alignment-check.cu — GDS Alignment & Readiness Diagnostic
 *
 * Demonstrates:
 *   1. Check GPU buffer alignment for GDS (4KB minimum)
 *   2. Check cudaMalloc vs cuMemAlloc alignment differences
 *   3. Check file offset alignment
 *   4. Check IO size alignment
 *   5. Comprehensive GDS readiness diagnostic
 *
 * Compile: nvcc -O2 -o 06-alignment-check 06-alignment-check.cu common/cufile_utils.cu -lcufile -lcuda
 * Run:     sudo ./06-alignment-check /mnt/nvme
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <cufile.h>
#include <cuda_runtime.h>
#include "common/cufile_utils.h"

/* ── Alignment Test Structures ─────────────────────────────────── */

typedef struct {
    const char *name;
    int passed;
    const char *detail;
} AlignmentTest;

/* ── Individual Tests ──────────────────────────────────────────── */

/* Test 1: cuMemAlloc alignment */
int test_cuMemAlloc_alignment(void) {
    printf("Test 1: cuMemAlloc alignment...\n");
    CUdeviceptr ptr;
    size_t sizes[] = {4096, 65536, 1048576, 16777216, 134217728}; // 4K to 128M

    int all_pass = 1;
    for (int i = 0; i < 5; i++) {
        cuMemAlloc(&ptr, sizes[i]);
        int aligned = ((uintptr_t)ptr % 4096 == 0);
        printf("  %s: %p → %s\n",
               format_size(sizes[i]), (void *)(uintptr_t)ptr,
               aligned ? "4KB-aligned ✓" : "NOT ALIGNED ✗");
        if (!aligned) all_pass = 0;
        cuMemFree(ptr);
    }
    printf("  Result: %s\n\n", all_pass ? "PASS" : "FAIL");
    return all_pass;
}

/* Test 2: cudaMalloc alignment */
int test_cudaMalloc_alignment(void) {
    printf("Test 2: cudaMalloc alignment (should be ≥ 256B)...\n");
    void *ptr;
    size_t sizes[] = {4096, 65536, 1048576};

    int all_4k = 1;
    for (int i = 0; i < 3; i++) {
        cudaMalloc(&ptr, sizes[i]);
        int aligned_4k = ((uintptr_t)ptr % 4096 == 0);
        int aligned_256 = ((uintptr_t)ptr % 256 == 0);
        printf("  %s: %p → 4KB: %s, 256B: %s\n",
               format_size(sizes[i]), ptr,
               aligned_4k ? "YES" : "NO",
               aligned_256 ? "YES" : "NO");
        if (!aligned_4k) all_4k = 0;
        cudaFree(ptr);
    }

    if (!all_4k) {
        printf("  ⚠  cudaMalloc does NOT guarantee 4KB alignment.\n");
        printf("     Use cuMemAlloc for guaranteed GDS-compatible alignment.\n");
    }
    printf("  Result: %s\n\n", all_4k ? "ALL 4KB-aligned" : "SOME not 4KB-aligned");
    return all_4k;
}

/* Test 3: File offset alignment */
int test_file_offset_alignment(const char *mountpoint) {
    printf("Test 3: File offset alignment...\n");

    char filepath[512];
    snprintf(filepath, sizeof(filepath), "%s/alignment_test.bin", mountpoint);

    int fd = open(filepath, O_DIRECT | O_RDWR | O_CREAT, 0644);
    if (fd < 0) {
        printf("  SKIP: Cannot open %s (filesystem may not support O_DIRECT)\n",
               filepath);
        printf("  Run: gdscheck -f %s\n\n", mountpoint);
        return -1; // Not a failure, just can't test
    }

    ftruncate(fd, 65536);

    /* Test various offsets */
    off_t test_offsets[] = {0, 4096, 8192, 1, 511, 4095, 4097};
    const char *expected[] = {"PASS","PASS","PASS","FAIL","FAIL","FAIL","FAIL"};

    int all_pass = 1;
    for (int i = 0; i < 7; i++) {
        int is_aligned = (test_offsets[i] % 4096 == 0);
        printf("  offset=%6lld → %s (GDS: %s)\n",
               (long long)test_offsets[i],
               is_aligned ? "4KB-aligned" : "MISALIGNED ",
               is_aligned ? "OK" : "REQUIRES COMPAT MODE");
        if (!is_aligned) all_pass = 0;
    }

    close(fd);
    unlink(filepath);
    printf("  GDS requires file offset %% 4096 == 0\n");
    printf("  Result: Misaligned offsets detected (expected)\n\n");
    return all_pass;
}

/* Test 4: IO Size alignment */
int test_io_size_alignment(void) {
    printf("Test 4: IO size alignment...\n");

    size_t test_sizes[] = {512, 1024, 2048, 4095, 4096, 8192, 65535, 65536};
    for (int i = 0; i < 8; i++) {
        int aligned = (test_sizes[i] % 4096 == 0);
        printf("  IO size=%s → %s\n",
               format_size(test_sizes[i]),
               aligned ? "GDS OK" : "Compat mode (not 4KB multiple)");
    }
    printf("  GDS works best with IO sizes that are multiples of 4KB.\n");
    printf("  Sub-4KB IOs use compatibility mode.\n\n");
    return 0;
}

/* Test 5: GDS registration test */
int test_gds_registration(const char *mountpoint) {
    printf("Test 5: Full GDS registration test...\n");

    /* Driver */
    CUfileError_t st = cuFileDriverOpen();
    if (st.err != CU_FILE_SUCCESS) {
        printf("  FAIL: cuFileDriverOpen: %s\n", cuFileGetErrorString(st));
        return 0;
    }

    CUfileDrvProps_t props;
    cuFileDriverGetProperties(&props);
    printf("  GDS capable: %s\n",
           props.is_gds_capable ? "YES" : "NO");
    printf("  GDS enabled:  %s\n",
           props.is_gds_enabled ? "YES" : "NO");

    if (!props.is_gds_capable) {
        printf("  SKIP: GDS not capable — remaining tests would fail.\n");
        cuFileDriverClose();
        return -1;
    }

    /* Buffer registration */
    CUdeviceptr devPtr;
    cuMemAlloc(&devPtr, 1048576);
    st = cuFileBufRegister(devPtr, 1048576, 0);
    printf("  Buffer registration: %s\n",
           st.err == CU_FILE_SUCCESS ? "PASS" : "FAIL");

    /* File registration */
    char filepath[512];
    snprintf(filepath, sizeof(filepath), "%s/gds_test.bin", mountpoint);
    int fd = open(filepath, O_DIRECT | O_RDWR | O_CREAT, 0644);
    if (fd < 0) {
        printf("  SKIP: Cannot open file with O_DIRECT\n");
        cuFileBufDeregister(devPtr);
        cuMemFree(devPtr);
        cuFileDriverClose();
        return -1;
    }
    ftruncate(fd, 1048576);

    CUfileDescr_t fh = {0};
    fh.cookie = (CUfileDriverCookie)(uintptr_t)fd;
    st = cuFileHandleRegister(&fh, NULL);
    printf("  File registration: %s\n",
           st.err == CU_FILE_SUCCESS ? "PASS" : "FAIL");

    /* Functional I/O test */
    ssize_t n = cuFileRead(fh, devPtr, 4096, 0, 0);
    printf("  Functional read (4KB): %s (%zd bytes)\n",
           (n == 4096) ? "PASS" : "FAIL", n);

    /* Cleanup */
    cuFileHandleDeregister(fh);
    close(fd);
    unlink(filepath);
    cuFileBufDeregister(devPtr);
    cuMemFree(devPtr);
    cuFileDriverClose();
    printf("\n");
    return (n == 4096);
}

/* ── Main ─────────────────────────────────────────────────────── */

int main(int argc, char *argv[]) {
    const char *mountpoint = (argc > 1) ? argv[1] : "/mnt/nvme";

    printf("=== GDS Alignment & Readiness Diagnostic ===\n");
    printf("Target mount point: %s\n\n", mountpoint);

    int tests_passed = 0, tests_total = 0;

    /* Test 1 */
    if (test_cuMemAlloc_alignment() > 0) tests_passed++;
    tests_total++;

    /* Test 2 */
    if (test_cudaMalloc_alignment() >= 0) tests_passed++;
    tests_total++;

    /* Test 3 */
    {
        int r = test_file_offset_alignment(mountpoint);
        if (r >= 0) { tests_passed += r ? 1 : 0; tests_total++; }
    }

    /* Test 4 */
    test_io_size_alignment();
    tests_passed++; tests_total++; // informational test

    /* Test 5 */
    {
        int r = test_gds_registration(mountpoint);
        if (r >= 0) { tests_passed += r ? 1 : 0; tests_total++; }
    }

    /* ── Summary ──────────────────────────────────────────── */
    printf("=== Summary ===\n");
    printf("Tests passed: %d / %d\n\n", tests_passed, tests_total);

    printf("GDS Alignment Requirements:\n");
    printf("  ✓ GPU buffer:  4KB-aligned (use cuMemAlloc)\n");
    printf("  ✓ File offset:  4KB-aligned\n");
    printf("  ✓ IO size:      4KB multiple (≥ 4KB for GDS path)\n");
    printf("  ✓ O_DIRECT:     Required on file open\n");
    printf("  ✓ Filesystem:   ext4, xfs, or GPFS\n\n");

    printf("Quick fix checklist:\n");
    printf("  1. Replace cudaMalloc → cuMemAlloc for guaranteed alignment\n");
    printf("  2. Ensure all file offsets are 4KB multiples\n");
    printf("  3. Round IO sizes up to next 4KB boundary\n");
    printf("  4. Always open files with O_DIRECT\n");
    printf("  5. Run 'gdscheck -p && gdscheck -f %s' for full diagnosis\n",
           mountpoint);

    printf("\n=== Example Complete ===\n");
    return (tests_passed == tests_total) ? EXIT_SUCCESS : EXIT_FAILURE;
}
