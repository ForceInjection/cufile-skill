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

/* ── GPU Verify Kernel ────────────────────────────────────────── */

/* Simple kernel: fill buffer with a pattern */
__global__ void fill_pattern_kernel(unsigned char *buf, size_t size,
                                    unsigned char base) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        buf[idx] = (unsigned char)(base + (idx % 251));
    }
}

/* Simple kernel: verify pattern */
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

ssize_t cuFileReadFull(CUfileDescr_t fh, CUdeviceptr devPtr,
                       size_t size, off_t offset) {
    size_t total = 0;
    while (total < size) {
        ssize_t n = cuFileRead(fh, devPtr + total, size - total,
                               offset + total, 0);
        if (n == 0) {
            fprintf(stderr, "  EOF at %zu / %zu bytes\n", total, size);
            break;
        }
        if (n < 0) {
            fprintf(stderr, "  Read error at %zu bytes\n", total);
            return -1;
        }
        total += n;
    }
    return (ssize_t)total;
}

ssize_t cuFileWriteFull(CUfileDescr_t fh, CUdeviceptr devPtr,
                        size_t size, off_t offset) {
    size_t total = 0;
    while (total < size) {
        ssize_t n = cuFileWrite(fh, devPtr + total, size - total,
                                offset + total, 0);
        if (n < 0) {
            fprintf(stderr, "  Write error at %zu bytes\n", total);
            return -1;
        }
        total += n;
    }
    return (ssize_t)total;
}

/* ── Main ─────────────────────────────────────────────────────── */

int main(int argc, char *argv[]) {
    const char *filepath = (argc > 1) ? argv[1] : "/mnt/nvme/testfile";
    printf("=== Synchronous cuFile Read/Write Example ===\n");
    printf("File: %s\n", filepath);
    printf("Test size: %s\n\n", format_size(TEST_SIZE));

    int thread_count = 256;
    int block_count = (TEST_SIZE + thread_count - 1) / thread_count;

    /* ── 1. Initialize cuFile driver ──────────────────────── */
    printf("1. Initializing cuFile driver...\n");
    CUfileError_t st = cuFileDriverOpen();
    if (st.err != CU_FILE_SUCCESS) {
        fprintf(stderr, "   FAILED: %s\n", cuFileGetErrorString(st));
        return EXIT_FAILURE;
    }
    printf("   OK.\n");
    check_gds_available();  // Print GDS status (informational)

    /* ── 2. Allocate and register GPU buffer ──────────────── */
    printf("\n2. Allocating GPU buffer (%s)...\n", format_size(TEST_SIZE));
    CUdeviceptr devPtr;
    CUresult cr = cuMemAlloc(&devPtr, TEST_SIZE);
    if (cr != CUDA_SUCCESS) {
        fprintf(stderr, "   cuMemAlloc failed: %d\n", cr);
        cuFileDriverClose();
        return EXIT_FAILURE;
    }
    printf("   Allocated at 0x%llx\n", (unsigned long long)devPtr);

    /* Check alignment */
    if (!check_alignment((const void *)(uintptr_t)devPtr, TEST_SIZE, 0)) {
        fprintf(stderr, "   WARNING: Buffer alignment check failed.\n");
    }

    printf("   Registering buffer for GDS...\n");
    st = cuFileBufRegister(devPtr, TEST_SIZE, 0);
    if (st.err != CU_FILE_SUCCESS) {
        fprintf(stderr, "   FAILED: %s\n", cuFileGetErrorString(st));
        cuMemFree(devPtr);
        cuFileDriverClose();
        return EXIT_FAILURE;
    }
    printf("   Registered.\n");

    /* ── 3. Fill GPU buffer with known pattern ────────────── */
    printf("\n3. Filling GPU buffer with test pattern...\n");
    fill_pattern_kernel<<<block_count, thread_count>>>(
        (unsigned char *)devPtr, TEST_SIZE, 0xAB);
    cudaDeviceSynchronize();
    printf("   Pattern written.\n");

    /* ── 4. Open and register file ────────────────────────── */
    printf("\n4. Opening file (O_DIRECT | O_RDWR | O_CREAT)...\n");
    int fd = open(filepath, O_DIRECT | O_RDWR | O_CREAT, 0644);
    if (fd < 0) {
        perror("   open");
        cuFileBufDeregister(devPtr);
        cuMemFree(devPtr);
        cuFileDriverClose();
        return EXIT_FAILURE;
    }
    printf("   Opened (fd=%d).\n", fd);

    /* Pre-allocate file to desired size */
    if (ftruncate(fd, TEST_SIZE) != 0) {
        perror("   ftruncate");
    }
    printf("   File pre-allocated to %s.\n", format_size(TEST_SIZE));

    printf("   Registering file handle...\n");
    CUfileDescr_t fh = {0};
    fh.cookie = (CUfileDriverCookie)(uintptr_t)fd;
    st = cuFileHandleRegister(&fh, NULL);
    if (st.err != CU_FILE_SUCCESS) {
        fprintf(stderr, "   FAILED: %s\n", cuFileGetErrorString(st));
        close(fd);
        cuFileBufDeregister(devPtr);
        cuMemFree(devPtr);
        cuFileDriverClose();
        return EXIT_FAILURE;
    }
    printf("   Registered.\n");

    /* ── 5. Write GPU buffer to file ──────────────────────── */
    printf("\n5. Writing %s to file...\n", format_size(TEST_SIZE));
    struct timespec t1, t2;

    clock_gettime(CLOCK_MONOTONIC, &t1);
    for (size_t off = 0; off < TEST_SIZE; off += IO_SIZE) {
        size_t chunk = (off + IO_SIZE <= TEST_SIZE) ? IO_SIZE : (TEST_SIZE - off);
        ssize_t n = cuFileWriteFull(fh, devPtr + off, chunk, off);
        if (n < 0) {
            fprintf(stderr, "   Write failed at offset %zu\n", off);
            break;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t2);

    double write_elapsed = (t2.tv_sec - t1.tv_sec) +
                           (t2.tv_nsec - t1.tv_nsec) / 1e9;
    double write_bw = measure_bandwidth(TEST_SIZE, write_elapsed);
    printf("   Complete: %.2f GB/s\n", write_bw);

    /* ── 6. Clear buffer and read back from file ──────────── */
    printf("\n6. Clearing GPU buffer...\n");
    cudaMemset((void *)devPtr, 0, TEST_SIZE);

    printf("   Reading %s from file...\n", format_size(TEST_SIZE));
    clock_gettime(CLOCK_MONOTONIC, &t1);
    for (size_t off = 0; off < TEST_SIZE; off += IO_SIZE) {
        size_t chunk = (off + IO_SIZE <= TEST_SIZE) ? IO_SIZE : (TEST_SIZE - off);
        ssize_t n = cuFileReadFull(fh, devPtr + off, chunk, off);
        if (n < 0) {
            fprintf(stderr, "   Read failed at offset %zu\n", off);
            break;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t2);

    double read_elapsed = (t2.tv_sec - t1.tv_sec) +
                          (t2.tv_nsec - t1.tv_nsec) / 1e9;
    double read_bw = measure_bandwidth(TEST_SIZE, read_elapsed);
    printf("   Complete: %.2f GB/s\n", read_bw);

    /* ── 7. Verify data integrity on GPU ──────────────────── */
    printf("\n7. Verifying data integrity on GPU...\n");
    int *d_error_count;
    cudaMalloc(&d_error_count, sizeof(int));
    cudaMemset(d_error_count, 0, sizeof(int));

    verify_pattern_kernel<<<block_count, thread_count>>>(
        (const unsigned char *)devPtr, TEST_SIZE, 0xAB, d_error_count);
    cudaDeviceSynchronize();

    int error_count = 0;
    cudaMemcpy(&error_count, d_error_count, sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaFree(d_error_count);

    if (error_count == 0) {
        printf("   ✅ Data integrity VERIFIED — all %s match.\n",
               format_size(TEST_SIZE));
    } else {
        printf("   ❌ Data CORRUPTION — %d bytes mismatch!\n", error_count);
    }

    /* ── 8. Cleanup (reverse order of setup) ──────────────── */
    printf("\n8. Cleanup...\n");
    printf("   Deregistering file handle...\n");
    cuFileHandleDeregister(fh);
    printf("   Closing file...\n");
    close(fd);
    printf("   Deregistering GPU buffer...\n");
    cuFileBufDeregister(devPtr);
    printf("   Freeing GPU memory...\n");
    cuMemFree(devPtr);
    printf("   Closing driver...\n");
    cuFileDriverClose();

    printf("\n=== Example Complete ===\n");
    printf("Write: %.2f GB/s | Read: %.2f GB/s\n", write_bw, read_bw);
    return (error_count == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
