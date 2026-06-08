/*
 * 02-sync-read-write.cu — Synchronous cuFile Read/Write with Verification
 *
 * Demonstrates:
 *   1. Full cuFile lifecycle: driver → buffer → file → I/O → cleanup
 *   2. cuFileBufRegister / cuFileBufDeregister — GPU buffer management
 *   3. cuFileHandleRegister / cuFileHandleDeregister — file handle management
 *   4. cuFileRead / cuFileWrite — synchronous I/O with partial read handling
 *   5. Data verification using a GPU kernel
 *
 * Compile: nvcc -O2 -o 02-sync-read-write 02-sync-read-write.cu common/cufile_utils.cu -lcufile -lcuda
 * Run:     sudo ./02-sync-read-write /mnt/nvme/testfile
 *
 * WARNING: This example writes to the file. Use a test file on a GDS-capable
 *          filesystem (ext4/xfs with O_DIRECT support).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <cufile.h>
#include <cuda_runtime.h>
#include "common/cufile_utils.h"

#define TEST_SIZE (64 * 1024 * 1024)  // 64 MB — adjust as needed
#define IO_SIZE   (4 * 1024 * 1024)   // 4 MB per I/O

/* ── GPU Verify Kernels ────────────────────────────────────────── */

__global__ void fill_pattern_kernel(unsigned char *buf, size_t size,
                                    unsigned char base) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        buf[idx] = (unsigned char)(base + (idx % 251));
    }
}

__global__ void verify_pattern_kernel(const unsigned char *buf,
                                       size_t size, unsigned char base,
                                       int *error_count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        unsigned char expected = (unsigned char)(base + (idx % 251));
        if (buf[idx] != expected) {
            atomicAdd(error_count, 1);
        }
    }
}

/* ── Synchronous Read with Partial Read Handling ──────────────── */

ssize_t cuFileReadFull(CUfileHandle_t fh, void *bufPtr,
                       size_t size, off_t offset) {
    size_t total = 0;
    while (total < size) {
        ssize_t n = cuFileRead(fh, (char *)bufPtr + total,
                               size - total, offset + total, 0);
        if (n == 0) {
            fprintf(stderr, "  EOF at %zu / %zu bytes\n", total, size);
            break;
        }
        if (n < 0) {
            fprintf(stderr, "  Read error at %zu bytes: %s\n",
                    total, CUFILE_ERRSTR((int)(-n)));
            return -1;
        }
        total += n;
    }
    return (ssize_t)total;
}

ssize_t cuFileWriteFull(CUfileHandle_t fh, const void *bufPtr,
                        size_t size, off_t offset) {
    size_t total = 0;
    while (total < size) {
        ssize_t n = cuFileWrite(fh, (const char *)bufPtr + total,
                                size - total, offset + total, 0);
        if (n < 0) {
            fprintf(stderr, "  Write error at %zu bytes: %s\n",
                    total, CUFILE_ERRSTR((int)(-n)));
            return -1;
        }
        total += n;
    }
    return (ssize_t)total;
}

/* ── Main ─────────────────────────────────────────────────────── */

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <file_path_on_gds_mount>\n", argv[0]);
        return EXIT_FAILURE;
    }
    const char *file_path = argv[1];

    /* ── Step 1: Driver init ────────────────────────────────── */
    printf("=== Sync Read/Write Example ===\n\n");
    CUfileError_t st = cuFileDriverOpen();
    cuFileCheck(st, "cuFileDriverOpen");

    /* ── Step 2: Allocate + register GPU buffer ─────────────── */
    printf("Allocating GPU buffer: %s\n", format_size(TEST_SIZE));
    unsigned char *gpu_buf;
    cudaMalloc(&gpu_buf, TEST_SIZE);
    if (!gpu_buf) {
        fprintf(stderr, "cudaMalloc failed\n");
        return EXIT_FAILURE;
    }

    st = cuFileBufRegister(gpu_buf, TEST_SIZE, 0);
    if (st.err != CU_FILE_SUCCESS) {
        fprintf(stderr, "cuFileBufRegister failed: %s (err=%d)\n",
                CUFILE_ERRSTR(st.err), st.err);
        fprintf(stderr, "(Continuing without registration — compat mode will be used)\n");
    }

    /* ── Step 3: Open + register file ───────────────────────── */
    printf("Opening file: %s\n", file_path);
    int fd = open_direct(file_path, O_CREAT | O_RDWR, 0644);
    preallocate_file(fd, TEST_SIZE);

    CUfileDescr_t descr = {0};
    descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    descr.handle.fd = fd;

    CUfileHandle_t fh;
    st = cuFileHandleRegister(&fh, &descr);
    cuFileCheck(st, "cuFileHandleRegister");

    /* ── Step 4: Write via cuFile (GPU → NVMe) ──────────────── */
    printf("Writing %s to file via cuFile...\n", format_size(TEST_SIZE));

    int block_size = 256;
    int grid_size = (TEST_SIZE + block_size - 1) / block_size;
    fill_pattern_kernel<<<grid_size, block_size>>>(gpu_buf, TEST_SIZE, 0xAB);
    cudaDeviceSynchronize();

    ssize_t written = cuFileWriteFull(fh, gpu_buf, TEST_SIZE, 0);
    printf("  Wrote %zd bytes\n", written);

    /* ── Step 5: Read back via cuFile (NVMe → GPU) ──────────── */
    printf("Reading %s back via cuFile...\n", format_size(TEST_SIZE));
    cudaMemset(gpu_buf, 0, TEST_SIZE);  // Clear buffer before read

    ssize_t nread = cuFileReadFull(fh, gpu_buf, TEST_SIZE, 0);
    printf("  Read %zd bytes\n", nread);

    /* ── Step 6: Verify data on GPU ─────────────────────────── */
    printf("Verifying data on GPU...\n");
    int *d_errors;
    cudaMalloc(&d_errors, sizeof(int));
    cudaMemset(d_errors, 0, sizeof(int));

    verify_pattern_kernel<<<grid_size, block_size>>>(gpu_buf, TEST_SIZE,
                                                      0xAB, d_errors);
    int h_errors;
    cudaMemcpy(&h_errors, d_errors, sizeof(int), cudaMemcpyDeviceToHost);

    if (h_errors == 0) {
        printf("✅ Data verification PASSED (%zd bytes match)\n", nread);
    } else {
        printf("❌ Data verification FAILED (%d mismatches)\n", h_errors);
    }

    /* ── Step 7: Cleanup (reverse order) ────────────────────── */
    cuFileHandleDeregister(fh);
    close(fd);
    cuFileBufDeregister(gpu_buf);
    cudaFree(gpu_buf);
    cudaFree(d_errors);
    cuFileDriverClose();

    printf("\n=== Example Complete ===\n");
    return EXIT_SUCCESS;
}
