/*
 * 04-batch-io.cu — cuFile Batch I/O for High Throughput
 *
 * Demonstrates:
 *   1. cuFileBatchIOSetUp — create a batch I/O handle
 *   2. cuFileBatchIOSubmit — submit multiple I/Os in a single call
 *   3. cuFileBatchIOGetStatus — poll or wait for completions
 *   4. cuFileBatchIODestroy — cleanup
 *   5. Performance comparison: batch vs individual synchronous I/O
 *
 * Compile: nvcc -O2 -o 04-batch-io 04-batch-io.cu common/cufile_utils.cu -lcufile -lcuda
 * Run:     sudo ./04-batch-io /mnt/nvme/testfile
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <cufile.h>
#include <cuda_runtime.h>
#include "common/cufile_utils.h"

#define TEST_SIZE   (256 * 1024 * 1024)  // 256 MB
#define BATCH_SIZE  64                    // I/Os per batch
#define IO_SIZE     (256 * 1024)          // 256 KB per IO — small IOs where batch helps

/* ── Fill GPU buffer with pattern ──────────────────────────────── */

__global__ void fill_pattern(unsigned char *buf, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        buf[idx] = (unsigned char)(idx % 256);
    }
}

/* ── Batch Read Benchmark ──────────────────────────────────────── */

double benchmark_batch_read(CUfileDescr_t fh, CUdeviceptr devPtr,
                            size_t total_size, int batch_size) {
    CUfileError_t st;
    int batch_id = 1;
    size_t num_chunks = total_size / IO_SIZE;

    /* Set up batch I/O */
    st = cuFileBatchIOSetUp(fh, batch_id, batch_size, 0);
    if (st.err != CU_FILE_SUCCESS) {
        fprintf(stderr, "  cuFileBatchIOSetUp failed: %s\n",
                cuFileGetErrorString(st));
        return -1;
    }

    /* Allocate parameter and status arrays */
    CUfileIOParams_t *io_params =
        (CUfileIOParams_t *)malloc(batch_size * sizeof(CUfileIOParams_t));
    CUfileIOStatus_t *io_status =
        (CUfileIOStatus_t *)malloc(batch_size * sizeof(CUfileIOStatus_t));

    struct timespec t1, t2;
    clock_gettime(CLOCK_MONOTONIC, &t1);

    size_t processed = 0;
    while (processed < num_chunks) {
        /* Build batch */
        int count = 0;
        for (int i = 0;
             i < batch_size && (processed + i) < num_chunks;
             i++) {
            io_params[i].fh = fh.handle;
            io_params[i].devPtr = devPtr + (processed + i) * IO_SIZE;
            io_params[i].file_offset = (processed + i) * IO_SIZE;
            io_params[i].devPtr_offset = 0;
            io_params[i].size = IO_SIZE;
            io_params[i].cookie = (void *)(uintptr_t)(processed + i);
            io_params[i].opcode = CUFILE_READ;
            count++;
        }

        /* Submit batch */
        st = cuFileBatchIOSubmit(batch_id, count, io_params, 0);
        if (st.err != CU_FILE_SUCCESS) {
            fprintf(stderr, "  cuFileBatchIOSubmit failed: %s\n",
                    cuFileGetErrorString(st));
            break;
        }

        /* Wait for all completions in this batch */
        int num_done = 0;
        st = cuFileBatchIOGetStatus(batch_id, count, count,
                                    io_status, &num_done);
        if (st.err != CU_FILE_SUCCESS) {
            fprintf(stderr, "  cuFileBatchIOGetStatus failed: %s\n",
                    cuFileGetErrorString(st));
            break;
        }

        /* Check individual results */
        for (int i = 0; i < num_done; i++) {
            if (io_status[i].err != CU_FILE_SUCCESS) {
                fprintf(stderr, "  Batch IO [%d] failed: err=%d\n",
                        i, io_status[i].err);
            } else if (io_status[i].ret != IO_SIZE) {
                fprintf(stderr, "  Batch IO [%d] short: %zd / %zu\n",
                        i, io_status[i].ret, (size_t)IO_SIZE);
            }
        }

        processed += count;
    }

    clock_gettime(CLOCK_MONOTONIC, &t2);
    double elapsed = (t2.tv_sec - t1.tv_sec) +
                     (t2.tv_nsec - t1.tv_nsec) / 1e9;

    /* Cleanup */
    free(io_params);
    free(io_status);
    cuFileBatchIODestroy(batch_id);

    return measure_bandwidth(total_size, elapsed);
}

/* ── Individual Sync Read Benchmark (for comparison) ───────────── */

double benchmark_sync_read(CUfileDescr_t fh, CUdeviceptr devPtr,
                           size_t total_size) {
    size_t num_chunks = total_size / IO_SIZE;

    struct timespec t1, t2;
    clock_gettime(CLOCK_MONOTONIC, &t1);

    for (size_t i = 0; i < num_chunks; i++) {
        ssize_t n = cuFileRead(fh, devPtr + i * IO_SIZE, IO_SIZE,
                               i * IO_SIZE, 0);
        if (n != IO_SIZE) {
            fprintf(stderr, "  Sync read failed at chunk %zu\n", i);
            break;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t2);
    double elapsed = (t2.tv_sec - t1.tv_sec) +
                     (t2.tv_nsec - t1.tv_nsec) / 1e9;

    return measure_bandwidth(total_size, elapsed);
}

/* ── Main ─────────────────────────────────────────────────────── */

int main(int argc, char *argv[]) {
    const char *filepath = (argc > 1) ? argv[1] : "/mnt/nvme/testfile";
    printf("=== Batch I/O Example ===\n");
    printf("File: %s\n", filepath);
    printf("Total size: %s\n", format_size(TEST_SIZE));
    printf("IO size: %s × %d per batch\n\n",
           format_size(IO_SIZE), BATCH_SIZE);

    int threads = 256;
    int blocks = (TEST_SIZE + threads - 1) / threads;

    /* ── 1. Setup ─────────────────────────────────────────── */
    printf("1. Setting up...\n");
    cuFileDriverOpen();
    check_gds_available();

    CUdeviceptr devPtr;
    cuMemAlloc(&devPtr, TEST_SIZE);
    cuFileBufRegister(devPtr, TEST_SIZE, 0);

    int fd = open(filepath, O_DIRECT | O_RDWR | O_CREAT, 0644);
    ftruncate(fd, TEST_SIZE);

    CUfileDescr_t fh = {0};
    fh.cookie = (CUfileDriverCookie)(uintptr_t)fd;
    cuFileHandleRegister(&fh, NULL);

    /* Fill buffer with data for initial write */
    fill_pattern<<<blocks, threads>>>((unsigned char *)devPtr, TEST_SIZE);
    cudaDeviceSynchronize();
    printf("   Setup complete.\n\n");

    /* ── 2. Benchmark: Individual sync reads ──────────────── */
    printf("2. Benchmark: Individual synchronous reads...\n");
    double sync_bw = benchmark_sync_read(fh, devPtr, TEST_SIZE);
    printf("   Sync reads: %.2f GB/s\n\n", sync_bw);

    /* ── 3. Benchmark: Batch reads ────────────────────────── */
    printf("3. Benchmark: Batch reads (batch_size=%d)...\n", BATCH_SIZE);
    double batch_bw = benchmark_batch_read(fh, devPtr, TEST_SIZE, BATCH_SIZE);
    printf("   Batch reads: %.2f GB/s\n\n", batch_bw);

    /* ── 4. Results ───────────────────────────────────────── */
    printf("4. Results:\n");
    printf("   Sync:  %.2f GB/s\n", sync_bw);
    printf("   Batch: %.2f GB/s\n", batch_bw);
    if (batch_bw > 0 && sync_bw > 0) {
        printf("   Speedup: %.2f×\n", batch_bw / sync_bw);
    }
    printf("\n   Note: Batch I/O speedup is typically 2-10× for small I/Os.\n");
    printf("   If speedup < 1×, IO size (%s) may be too large for batch benefit.\n",
           format_size(IO_SIZE));

    /* ── 5. Cleanup ───────────────────────────────────────── */
    printf("\n5. Cleanup...\n");
    cuFileHandleDeregister(fh);
    close(fd);
    cuFileBufDeregister(devPtr);
    cuMemFree(devPtr);
    cuFileDriverClose();
    printf("   Done.\n");

    printf("\n=== Example Complete ===\n");
    return EXIT_SUCCESS;
}
