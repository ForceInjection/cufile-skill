/*
 * 01-driver-init.cu — cuFile Driver Lifecycle & Properties Query
 *
 * Demonstrates:
 *   1. cuFileDriverOpen() — initialize the GDS driver
 *   2. cuFileDriverGetProperties() — query driver version and GDS capabilities
 *   3. Interpreting GDS status via nvfs.dstatusflags / dcontrolflags
 *   4. cuFileDriverClose() — clean shutdown
 *
 * Compile: nvcc -O2 -o 01-driver-init 01-driver-init.cu common/cufile_utils.cu -lcufile -lcuda
 * Run:     ./01-driver-init
 */

#include <stdio.h>
#include <stdlib.h>
#include <cufile.h>
#include <cuda_runtime.h>
#include "common/cufile_utils.h"

int main(void) {
    printf("=== cuFile Driver Lifecycle Example ===\n\n");

    /* ── Step 1: Check CUDA environment ──────────────────────── */
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        fprintf(stderr, "ERROR: No CUDA-capable GPU found.\n");
        return EXIT_FAILURE;
    }

    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("GPU %d: %s (CC %d.%d)\n",
               i, prop.name, prop.major, prop.minor);

        if (prop.major < 6) {
            fprintf(stderr,
                    "WARNING: GPU %d has CC %d.%d — GDS requires "
                    "Pascal (SM 6.0) or newer.\n",
                    i, prop.major, prop.minor);
        }
    }
    printf("\n");

    /* ── Step 2: Open cuFile driver ──────────────────────────── */
    printf("Opening cuFile driver...\n");
    CUfileError_t status = cuFileDriverOpen();
    if (status.err != CU_FILE_SUCCESS) {
        fprintf(stderr, "ERROR: cuFileDriverOpen failed: %s (err=%d)\n",
                CUFILE_ERRSTR(status.err), status.err);
        fprintf(stderr, "\nTroubleshooting:\n");
        fprintf(stderr, "  1. Is nvidia-fs module loaded? "
                "→ lsmod | grep nvidia_fs\n");
        fprintf(stderr, "  2. Is NVIDIA driver ≥ 470.57? "
                "→ nvidia-smi\n");
        fprintf(stderr, "  3. Is libcufile installed? "
                "→ find /usr/local/cuda -name 'libcufile*'\n");
        return EXIT_FAILURE;
    }
    printf("Driver opened successfully.\n\n");

    /* ── Step 3: Query and display driver properties ─────────── */
    CUfileDrvProps_t props = {0};
    status = cuFileDriverGetProperties(&props);
    if (status.err != CU_FILE_SUCCESS) {
        fprintf(stderr,
                "ERROR: cuFileDriverGetProperties failed: %s\n",
                CUFILE_ERRSTR(status.err));
        cuFileDriverClose();
        return EXIT_FAILURE;
    }

    int nvme_ok = gds_nvme_supported(&props);
    int p2p_ok  = gds_nvme_p2p_supported(&props);
    int compat  = gds_compat_mode_allowed(&props);

    printf("=== Driver Properties ===\n");
    printf("  nvfs version:          v%d.%d\n",
           props.nvfs.major_version, props.nvfs.minor_version);
    printf("  NVMe supported:         %s\n",
           nvme_ok ? "YES ✓" : "NO ✗");
    printf("  NVMe P2P DMA (GDS):     %s\n",
           p2p_ok  ? "YES ✓" : "NO ✗");
    printf("  Compat mode allowed:    %s\n",
           compat  ? "YES" : "NO (hard errors on fallback)");
    printf("  Max direct IO size:    %s\n",
           format_size(props.nvfs.max_direct_io_size));
    printf("  Max device cache:      %s\n",
           format_size(props.max_device_cache_size));
    printf("  Max pinned mem:        %s\n",
           props.max_device_pinned_mem_size == 0
               ? "unlimited"
               : format_size(props.max_device_pinned_mem_size));
    printf("  Max batch IO size:     %u\n",
           props.max_batch_io_size);

    /* ── Step 4: Interpret GDS status ────────────────────────── */
    printf("\n=== Status Assessment ===\n");

    if (!nvme_ok) {
        printf("⚠️  NVMe not detected by nvidia-fs driver.\n");
        printf("   All I/O will use compatibility mode (CPU bounce buffer).\n");
        printf("\n   To enable GDS:\n");
        printf("   1. Run: gdscheck -p\n");
        printf("   2. Verify NVMe SSD is present and nvidia-fs is loaded\n");
    } else if (!p2p_ok) {
        printf("⚠️  NVMe detected but P2P DMA (GDS) NOT available.\n");
        printf("   I/O will use compatibility mode (CPU bounce buffer).\n");
        printf("\n   To enable GDS P2P:\n");
        printf("   1. Verify GPU and NVMe on same PCIe root complex\n");
        printf("   2. Check ACS is disabled for PCIe P2P\n");
        printf("   3. Run: gdscheck -p\n");
    } else {
        printf("✅ GDS NVMe P2P DMA is available!\n");
        printf("   Maximum direct IO: %s per operation\n",
               format_size(props.nvfs.max_direct_io_size));
    }

    /* ── Step 5: Cleanup ─────────────────────────────────────── */
    printf("\nClosing cuFile driver...\n");
    status = cuFileDriverClose();
    if (status.err != CU_FILE_SUCCESS) {
        fprintf(stderr, "WARNING: cuFileDriverClose failed: %s\n",
                CUFILE_ERRSTR(status.err));
    } else {
        printf("Driver closed successfully.\n");
    }

    printf("\n=== Example Complete ===\n");
    return EXIT_SUCCESS;
}
