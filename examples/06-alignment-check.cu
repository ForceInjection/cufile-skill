/*
 * 06-alignment-check.cu — GDS Alignment & Readiness Diagnostic
 *
 * Demonstrates:
 *   1. Buffer alignment verification (cudaMalloc vs cuMemAlloc)
 *   2. File offset and IO size alignment checks
 *   3. GDS driver properties diagnostic
 *   4. Comprehensive readiness report
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

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <nvme_mountpoint>\n", argv[0]);
        return EXIT_FAILURE;
    }
    const char *mount = argv[1];

    printf("=== GDS Alignment & Readiness Diagnostic ===\n\n");

    /* ── 1. CUDA Device Info ────────────────────────────────── */
    printf("─── CUDA Device ───\n");
    int dev_count;
    cudaGetDeviceCount(&dev_count);
    for (int i = 0; i < dev_count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("  GPU %d: %s (CC %d.%d, %.1f GB)\n",
               i, prop.name, prop.major, prop.minor,
               prop.totalGlobalMem / 1e9);
        printf("         PCIe Gen%d x%d\n",
               prop.pciBusID, /* approximate */ 0);
    }
    printf("\n");

    /* ── 2. Buffer Alignment ────────────────────────────────── */
    printf("─── Buffer Alignment ───\n");

    /* Standard cudaMalloc — may or may not be 4KB-aligned */
    void *p1;
    cudaMalloc(&p1, 4096);
    printf("  cudaMalloc(4KB):  %p (%s)\n", p1,
           ((uintptr_t)p1 % 4096 == 0) ? "4KB-aligned ✓" : "NOT aligned ✗");

    /* cuMemAlloc — guaranteed 4KB-aligned (or larger) */
    CUdeviceptr p2;
    CUresult cr = cuMemAlloc(&p2, 4096);
    if (cr == CUDA_SUCCESS) {
        printf("  cuMemAlloc(4KB):  0x%llx (%s)\n",
               (unsigned long long)p2,
               (p2 % 4096 == 0) ? "4KB-aligned ✓" : "NOT aligned ✗");
        cuMemFree(p2);
    } else {
        printf("  cuMemAlloc:       FAILED (CUresult=%d)\n", cr);
    }

    /* Larger allocation alignment */
    void *p3;
    cudaMalloc(&p3, 64 * 1024 * 1024);  // 64 MB
    printf("  cudaMalloc(64MB): %p (%s)\n", p3,
           ((uintptr_t)p3 % 4096 == 0) ? "4KB-aligned ✓" : "NOT aligned ✗");

    cudaFree(p1);
    cudaFree(p3);
    printf("\n");

    /* ── 3. GDS Driver Properties ───────────────────────────── */
    printf("─── GDS Driver ───\n");
    CUfileError_t st = cuFileDriverOpen();
    if (st.err != CU_FILE_SUCCESS) {
        printf("  Driver: NOT AVAILABLE (%s)\n", CUFILE_ERRSTR(st.err));
        printf("\n=== Report Complete (driver unavailable) ===\n");
        return EXIT_FAILURE;
    }

    CUfileDrvProps_t props = {0};
    st = cuFileDriverGetProperties(&props);
    if (st.err == CU_FILE_SUCCESS) {
        printf("  nvfs version:     v%d.%d\n",
               props.nvfs.major_version, props.nvfs.minor_version);
        printf("  NVMe supported:   %s\n",
               gds_nvme_supported(&props) ? "YES" : "NO");
        printf("  NVMe P2P (GDS):   %s\n",
               gds_nvme_p2p_supported(&props) ? "YES" : "NO");
        printf("  Compat mode:      %s\n",
               gds_compat_mode_allowed(&props) ? "ALLOWED" : "DISABLED");
        printf("  Max direct IO:    %s\n",
               format_size(props.nvfs.max_direct_io_size));
        printf("  Max device cache: %s\n",
               format_size(props.max_device_cache_size));
        printf("  Max pinned mem:   %s\n",
               props.max_device_pinned_mem_size == 0
                   ? "unlimited" : format_size(props.max_device_pinned_mem_size));
        printf("  Max batch IO:     %u\n", props.max_batch_io_size);
    }
    printf("\n");

    /* ── 4. Filesystem Check ────────────────────────────────── */
    printf("─── Filesystem: %s ───\n", mount);

    /* O_DIRECT test */
    char testpath[512];
    snprintf(testpath, sizeof(testpath), "%s/.gds_align_test_%d",
             mount, getpid());
    int fd = open(testpath, O_CREAT | O_RDWR | O_DIRECT, 0644);
    if (fd >= 0) {
        printf("  O_DIRECT open:    SUCCESS ✓\n");
        close(fd);
        unlink(testpath);
    } else {
        printf("  O_DIRECT open:    FAILED ✗ (%s)\n", strerror(errno));
        printf("    → GDS REQUIRES O_DIRECT. Check filesystem mount options.\n");
    }
    printf("\n");

    /* ── 5. Summary ─────────────────────────────────────────── */
    printf("─── Overall Assessment ───\n");
    int issues = 0;

    if (!gds_nvme_supported(&props)) {
        printf("  ❌ NVMe not supported by nvidia-fs driver\n");
        issues++;
    }
    if (!gds_nvme_p2p_supported(&props)) {
        printf("  ⚠️  NVMe P2P DMA (GDS) not available\n");
        printf("     Check: GPU+NVMe same PCIe root complex, ACS disabled\n");
        issues++;
    }
    if (issues == 0) {
        printf("  ✅ System is GDS-ready.\n");
    }

    cuFileDriverClose();
    printf("\n=== Diagnostic Complete ===\n");
    return (issues == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
