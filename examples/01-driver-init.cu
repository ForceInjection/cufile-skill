/*
 * 01-driver-init.cu — cuFile Driver Lifecycle & Properties Query
 *
 * Demonstrates:
 *   1. cuFileDriverOpen() — initialize the GDS driver
 *   2. cuFileDriverGetProperties() — query driver version and GDS capabilities
 *   3. Interpreting GDS status (capable vs enabled)
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
                cuFileGetErrorString(status), status.err);
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
                cuFileGetErrorString(status));
        cuFileDriverClose();
        return EXIT_FAILURE;
    }

    printf("=== Driver Properties ===\n");
    printf("  API version:           v%d.%d\n",
           props.major, props.minor);
    printf("  GDS capable:           %s\n",
           props.is_gds_capable ? "YES ✓" : "NO ✗");
    printf("  GDS enabled:            %s\n",
           props.is_gds_enabled ? "YES ✓" : "NO ✗");
    printf("  Max direct IO size:    %s\n",
           format_size(props.max_device_direct_io_size));
    printf("  Max device cache:      %s\n",
           format_size(props.max_device_cache_size));
    printf("  Max pinned mem:        %s\n",
           props.max_device_pinned_mem_size == 0
               ? "unlimited"
               : format_size(props.max_device_pinned_mem_size));

    /* ── Step 4: Interpret GDS status ────────────────────────── */
    printf("\n=== Status Assessment ===\n");

    if (!props.is_gds_capable) {
        printf("⚠️  GDS is NOT capable on this platform.\n");
        printf("   All I/O will use compatibility mode (CPU bounce buffer).\n");
        printf("\n   To enable GDS:\n");
        printf("   1. Run: gdscheck -p\n");
        printf("   2. Verify GPU (SM ≥ 6.0) and NVMe on same PCIe root complex\n");
        printf("   3. Check ACS is disabled for PCIe P2P\n");
    } else if (!props.is_gds_enabled) {
        printf("⚠️  GDS is CAPABLE but NOT enabled.\n");
        printf("   Check:\n");
        printf("   1. /etc/cufile.json → enable_compat_mode should be true\n");
        printf("   2. Run: gdscheck -f <your-nvme-mountpoint>\n");
        printf("   3. Filesystem must support O_DIRECT (ext4/xfs)\n");
    } else {
        printf("✅ GDS is fully operational!\n");
        printf("   Maximum throughput: up to %s per IO\n",
               format_size(props.max_device_direct_io_size));
    }

    /* ── Step 5: Cleanup ─────────────────────────────────────── */
    printf("\nClosing cuFile driver...\n");
    status = cuFileDriverClose();
    if (status.err != CU_FILE_SUCCESS) {
        fprintf(stderr, "WARNING: cuFileDriverClose failed: %s\n",
                cuFileGetErrorString(status));
        /* Non-fatal — driver may recover */
    } else {
        printf("Driver closed successfully.\n");
    }

    printf("\n=== Example Complete ===\n");
    return EXIT_SUCCESS;
}
